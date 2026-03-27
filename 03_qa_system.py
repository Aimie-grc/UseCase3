# -*- coding: utf-8 -*-
"""
Export du système QA immobilier (logique du notebook 03_qa_system).
Pas de Colab, pas de drive, pas de HF_TOKEN.
"""
from __future__ import annotations

import os
import re
import json
import unicodedata
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

_BASE_DIR = Path(__file__).resolve().parent

# Regex compilées une seule fois pour _normalize_commune
_RE_HYPHEN_APOS = re.compile(r"[-']")
_RE_ST = re.compile(r"\bST\b")
_RE_STE = re.compile(r"\bSTE\b")
_RE_SPACES = re.compile(r"\s+")


def _normalize_commune(name: str) -> str:
    """Normalise pour le matching : majuscules, tirets/apostrophes, St/Ste, suppression des accents."""
    if not name or not isinstance(name, str):
        return ""
    s = name.upper().strip()
    s = _RE_HYPHEN_APOS.sub(" ", s)
    s = _RE_ST.sub("SAINT", s)
    s = _RE_STE.sub("SAINTE", s)
    s = _RE_SPACES.sub(" ", s).strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s


def _is_year_series(d) -> bool:
    """True si d est un dict type {année: valeur} (au moins une clé numérique)."""
    if not isinstance(d, dict):
        return False
    for k in d:
        if isinstance(k, str) and k.isdigit():
            return True
    return False


_IGNORED_COMMUNE_NAMES = {
    "MOYEN", "EU", "LE", "LA", "LES", "UN", "DE", "DU", "AU", "AUX",
    "ET", "OU", "EN", "CE", "SA", "AI", "SO", "DIE", "VIF", "MAISONS",
}


def _phrase_contenant_reponse_bert(context: str, bert_answer: str) -> str:
    """
    Cherche l'extrait renvoyé par BERT dans le contexte (plusieurs stratégies),
    puis retourne la phrase complète (. ! ?) qui le contient.
    """
    a = (bert_answer or "").strip()
    if not a or not context:
        return a
    n = len(context)
    start: int | None = None
    end: int | None = None

    def set_span(s: int, e: int) -> None:
        nonlocal start, end
        start, end = max(0, s), min(n, e)

    i = context.find(a)
    if i >= 0:
        set_span(i, i + len(a))
    else:
        la, lc = a.lower(), context.lower()
        i = lc.find(la)
        if i >= 0:
            set_span(i, i + len(a))
    if start is None:
        parts = [p for p in a.split() if p]
        if parts:
            pat = r"\s+".join(re.escape(p) for p in parts)
            m = re.search(pat, context, re.IGNORECASE)
            if m:
                set_span(m.start(), m.end())
    if start is None:
        chiffres = re.sub(r"[^\d]", "", a)
        if len(chiffres) >= 2:
            i = context.find(chiffres)
            if i >= 0:
                set_span(i, i + len(chiffres))
            else:
                pat_nb = r"\s*".join(re.escape(c) for c in chiffres)
                mnb = re.search(pat_nb, context)
                if mnb:
                    set_span(mnb.start(), mnb.end())
    if start is None:
        for L in range(len(a), 2, -1):
            sub = a[:L]
            i = context.find(sub)
            if i < 0:
                i = context.lower().find(sub.lower())
            if i >= 0:
                set_span(i, i + len(a))
                break
    if start is None:
        for w in sorted(a.split(), key=len, reverse=True):
            if len(w) >= 3:
                i = context.find(w)
                if i < 0:
                    i = context.lower().find(w.lower())
                if i >= 0:
                    set_span(i, min(i + len(a), n))
                    break

    if start is None or end is None:
        return a

    left = start
    while left > 0 and context[left - 1] not in ".!?\n\r":
        left -= 1
    right = end
    while right < n and context[right] not in ".!?\n\r":
        right += 1
    if right < n and context[right] in ".!?":
        right += 1
    phrase = context[left:right].strip()
    return phrase if phrase else a


class DataLoader:
    """Charge les données depuis json_par_ville.json."""

    def __init__(self, json_path: str | Path) -> None:
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self._index = {}
        for key in self.data:
            if "|" in key:
                commune, dep = key.split("|", 1)
                norm = _normalize_commune(commune)
                self._index.setdefault(norm, []).append(key)
        self._sorted_norms = sorted(self._index.keys(), key=len, reverse=True)
        self._year_spectrum = None

    def _get_year_from_series(self, d, year: int | None = None):
        """Pour un dict année->valeur : si year donné et présent retourne d[year] ; si year donné et absent None ; si year None retourne la plus récente."""
        if not d or not isinstance(d, dict):
            return None
        years = [k for k in d if isinstance(k, str) and k.isdigit()]
        if not years:
            return None
        if year is not None:
            if str(year) in years:
                return d[str(year)]
            return None
        return d[max(years, key=int)]

    def find_communes(self, query_text: str) -> list[str]:
        query_norm = _normalize_commune(query_text)
        found_keys = []
        found_set = set()
        matched_spans = []
        for norm in self._sorted_norms:
            if norm in _IGNORED_COMMUNE_NAMES:
                continue
            for m in re.finditer(r"\b" + re.escape(norm) + r"\b", query_norm):
                start, end = m.start(), m.end()
                if any(s <= start and end <= e for s, e in matched_spans):
                    continue
                for k in self._index[norm]:
                    if k not in found_set:
                        found_set.add(k)
                        found_keys.append(k)
                matched_spans.append((start, end))
        return found_keys

    def get_data_by_key(self, key: str, year: int | None = None) -> dict | None:
        """Données pour une clé donnée (ex. 'TOULOUSE|31'), sous forme de dict plat."""
        if key not in self.data:
            return None
        raw = self.data[key]
        commune, code_dep = key.split("|", 1)
        out = {"Commune": commune, "Code_dep": code_dep, "_key": key}
        for var_name, var_values in raw.items():
            if _is_year_series(var_values):
                out[var_name] = self._get_year_from_series(var_values, year)
            else:
                out[var_name] = var_values
        return out

    def get_year_spectrum(self) -> tuple:
        """(année_min, année_max) basé sur Paris ; résultat mis en cache."""
        if self._year_spectrum is not None:
            return self._year_spectrum
        ref_key = "PARIS|75"
        if ref_key not in self.data:
            for k in self.data:
                if "|" in k and "periode_donnees" in self.data[k]:
                    ref_key = k
                    break
            else:
                self._year_spectrum = (None, None)
                return self._year_spectrum
        raw = self.data[ref_key]
        if "periode_donnees" in raw and isinstance(raw["periode_donnees"], dict):
            pd_ = raw["periode_donnees"]
            self._year_spectrum = (pd_.get("debut"), pd_.get("fin"))
        else:
            nv = raw.get("Nb_total_ventes")
            if isinstance(nv, dict):
                years = [int(k) for k in nv if isinstance(k, str) and k.isdigit()]
                self._year_spectrum = (min(years), max(years)) if years else (None, None)
            else:
                self._year_spectrum = (None, None)
        return self._year_spectrum


class BertQAManager:
    """QA par inférence directe (sans pipeline) pour compatibilité transformers."""

    def __init__(self, model_name: str = "CATIE-AQ/QAmemberta") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()
        self._device = next(self.model.parameters()).device
        print("Modèle BERT chargé")

    def extract_answer(self, context: str, question: str, max_answer_len: int = 150) -> dict | None:
        """Inférence directe : tokenize -> model -> start/end_logits -> decode span. Retourne dict avec answer, score, optionnellement start, end."""
        if not context or len(context.strip()) < 10:
            return None
        max_length = 4000
        if len(context) > max_length:
            context = context[:max_length] + "..."

        try:
            inputs = self.tokenizer(
                question,
                context,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            # Logits (1, seq_len) -> (seq_len,)
            start_logits = start_logits[0].cpu()
            end_logits = end_logits[0].cpu()
            start_scores = torch.softmax(start_logits, dim=0)
            end_scores = torch.softmax(end_logits, dim=0)

            # Meilleur span : start puis end >= start et end - start <= max_answer_len
            seq_len = start_logits.shape[0]
            best_score = -1.0
            best_start, best_end = 0, 0
            for start_idx in range(seq_len):
                end_max = min(start_idx + max_answer_len, seq_len)
                for end_idx in range(start_idx, end_max):
                    sc = start_scores[start_idx].item() * end_scores[end_idx].item()
                    if sc > best_score:
                        best_score = sc
                        best_start, best_end = start_idx, end_idx

            input_ids = inputs["input_ids"][0].cpu()
            # Tokens spéciaux (CLS=0, SEP) : si span vide ou impossible, considérer comme pas de réponse
            if best_start == 0 and best_end == 0:
                return None
            decoded = self.tokenizer.decode(
                input_ids[best_start : best_end + 1],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            if not decoded or len(decoded) < 2:
                return None
            sentence = _phrase_contenant_reponse_bert(context, decoded)
            return {
                "answer": decoded,
                "sentence": sentence,
                "score": float(best_score),
                "start": int(best_start),
                "end": int(best_end),
            }
        except Exception as e:
            print(f"Erreur BERT QA: {e}")
            return None


class ImmobilierChatbot:
    def __init__(self, json_path: str | Path, model_name: str = "CATIE-AQ/QAmemberta") -> None:
        self.data_loader = DataLoader(json_path)
        self.bert_qa = BertQAManager(model_name=model_name)
        print("\nChatbot prêt !")

    def detect_question_type(self, question: str) -> str:
        q_lower = question.lower()
        if any(
            x in q_lower
            for x in [
                "general", "toutes les données", "toute les données", "résumé", "tout sur",
                "données complètes", "résume", "en résumé", "vue d ensemble", "vue d'ensemble",
            ]
        ):
            return "general"
        has_prix = "prix" in q_lower
        has_evol = any(
            x in q_lower
            for x in [
                "évolu", "évolution", "évoluent", "evolu", "evoluent", "variation",
                "hausse", "augmenté", "augmente", "baissé", "baisse", "stable", "tendance", "marché",
            ]
        )
        if has_evol:
            return "evolution"
        if has_prix:
            if "appartement" in q_lower or "appart" in q_lower:
                return "prix_appartement"
            if "maison" in q_lower:
                return "prix_maison"
            return "prix"
        if any(x in q_lower for x in ["surface", "m²", "m2", "terrain", "taille"]):
            return "surface"
        if any(x in q_lower for x in ["population", "habitants", "habitent", "démographie", "demographie", "nombre d'habitants"]):
            return "population"
        if any(x in q_lower for x in ["ventes", "combien", "nombre de ventes", "vendus", "vendues", "vendu", "transactions"]):
            return "ventes"
        return "general"

    def extract_year(self, question: str) -> int | None:
        years = re.findall(r"\b(19\d{2}|20\d{2}|2100)\b", question)
        return int(years[0]) if years else None

    def _get_dept_from_question(self, question: str):
        m = re.search(r"\b(département|dept\.?|dpt)\s*(\d{2,3}[AB]?)\b", question.lower())
        if m:
            return m.group(2)
        m = re.search(r"\b(64|75|69|33|59|13|31|44|35|67|68)\b", question)
        return m.group(1) if m else None

    def _safe_float(self, v, default=0):
        if pd.isna(v):
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _safe_int(self, v, default=0):
        if pd.isna(v):
            return default
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default

    def _fallback_answer(self, question: str, context: str, q_type: str) -> dict | str:
        q_lower = question.lower()
        if q_type in ("prix", "prix_appartement", "prix_maison"):
            if "m²" in q_lower or "m2" in q_lower or "au m" in q_lower:
                m = re.search(r"(?:prix (?:moyen|médian) )?au m² est de ([\d\s]+) euros", context)
                if not m:
                    m = re.search(r"prix médian au m² est de ([\d\s]+) euros", context)
                if not m:
                    m = re.search(r"prix moyen au m² est de ([\d\s]+) euros", context)
                if m:
                    return {"answer": m.group(1).replace(" ", "").strip() + " euros", "score": 0.85}
            if "minimum" in q_lower or "min" in q_lower:
                if "m²" in q_lower or "m2" in q_lower:
                    m = re.search(r"(?:prix au m² |prix au m2 )?minimum est de ([\d\s]+) euros", context)
                    if m:
                        return {"answer": m.group(1).replace(" ", "").strip() + " euros le m²", "score": 0.9}
                m = re.search(r"(?:Le )?prix minimum (?:enregistré )?est de ([\d\s]+) euros", context)
                if not m:
                    m = re.search(r"Les prix vont de ([\d\s]+) à", context)
                if m:
                    return {"answer": m.group(1).replace(" ", "").strip() + " euros", "score": 0.85}
            if "maximum" in q_lower or "max" in q_lower:
                if "m²" in q_lower or "m2" in q_lower:
                    m = re.search(r"prix au m² maximum est de ([\d\s]+) euros", context)
                    if m:
                        return {"answer": m.group(1).replace(" ", "").strip() + " euros le m²", "score": 0.9}
                m = re.search(r"(?:Le )?prix maximum est de ([\d\s]+) euros", context)
                if not m:
                    m = re.search(r"Les prix vont de [\d\s]+ à ([\d\s]+) euros", context)
                if m:
                    return {"answer": m.group(1).replace(" ", "").strip() + " euros", "score": 0.85}
            if "fourchette" in q_lower:
                m = re.search(
                    r"Le prix minimum enregistré est de ([\d\s]+) euros et le prix maximum est de ([\d\s]+) euros",
                    context,
                )
                if m:
                    return {
                        "answer": "de " + m.group(1).replace(" ", "").strip() + " à " + m.group(2).replace(" ", "").strip() + " euros",
                        "score": 0.9,
                    }
            m = re.search(r"prix médian (?:d'un bien |des (?:appartements|maisons) )?est de ([\d\s]+) euros", context)
            if not m:
                m = re.search(r"prix moyen d'un bien est de ([\d\s]+) euros", context)
            if m:
                return {"answer": m.group(1).replace(" ", "").strip() + " euros", "score": 0.85}
        if q_type == "evolution":
            m = re.search(r"(?:est en|marché (?:immobilier )?à [A-Za-zÀ-ÿ\s'-]+ est en) ([A-Za-zÀ-ÿ\s]+?)\.", context)
            if m:
                t = m.group(1).strip()
                if t and len(t) < 50:
                    return {"answer": t, "score": 0.85}
            m = re.search(r"évolution du prix au m² sur la période est de ([^.]+\.?)", context)
            if m:
                return {"answer": m.group(1).strip(), "score": 0.85}
            m = re.search(r"plus forte (?:hausse|baisse) des prix est (\d{4})", context)
            if m:
                return {"answer": m.group(1), "score": 0.85}
        if q_type == "surface":
            m = re.search(r"surface (?:intérieure )?(?:moyenne|médiane) [^.]*?est de ([\d\s]+) m²", context)
            if m:
                return {"answer": m.group(1).replace(" ", "").strip() + " m²", "score": 0.85}
        if q_type == "population":
            m = re.search(r"(?:la )?population est de ([\d\s]+) habitants", context)
            if m:
                return {"answer": m.group(1).replace(" ", "").strip() + " habitants", "score": 0.9}
        if q_type == "ventes":
            m = re.search(r"il y a eu ([\d\s]+) ventes", context)
            if not m:
                m = re.search(r"(?:nombre de biens vendus|nombre total de ventes) est de ([\d\s]+)", context)
            if m:
                return {"answer": m.group(1).replace(" ", "").strip(), "score": 0.9}
            if "maisons" in q_lower:
                m = re.search(r"([\d\s]+) maisons et", context)
                if m:
                    return {"answer": m.group(1).replace(" ", "").strip() + " maisons", "score": 0.9}
            if "appartement" in q_lower:
                m = re.search(r"et ([\d\s]+) appartements", context)
                if m:
                    return {"answer": m.group(1).replace(" ", "").strip() + " appartements", "score": 0.9}
        first_sentence = context.split(". ")[0] + "." if context else ""
        if len(first_sentence) > 20:
            return f"Je n'ai pas pu extraire une réponse précise. D'après les données : {first_sentence}"
        return "Je n'ai pas trouvé de réponse précise dans les données. Vous pouvez reformuler votre question ou demander un résumé (ex. : Donnez-moi les données sur Toulouse)."

    def build_context_as_text(
        self,
        question: str,
        question_type: str,
        *,
        found_keys: list[str] | None = None,
        year: int | None = None,
        dept: str | None = None,
    ) -> tuple[str | None, bool]:
        """
        Si found_keys / year / dept sont fournis (ex. depuis answer), évite un second
        find_communes et les regex année/département — même résultat qu'avant.
        """
        if found_keys is not None:
            keys = list(found_keys)
        else:
            keys = self.data_loader.find_communes(question)
        if year is None:
            year = self.extract_year(question)
        if dept is None:
            dept = self._get_dept_from_question(question)
        if dept:
            keys = [k for k in keys if k.endswith("|" + str(dept).zfill(2))]
        if not keys:
            keys = (
                list(found_keys)
                if found_keys is not None
                else self.data_loader.find_communes(question)
            )

        sentences = []
        loc = lambda c, y: f"En {y} à {c}, " if y else f"À {c}, "

        if question_type == "prix":
            for key in keys:
                data = self.data_loader.get_data_by_key(key, year=year)
                if not data:
                    continue
                pmoy = self._safe_float(data.get("Prix_moy"))
                pmed = self._safe_float(data.get("Prix_med"))
                pmin = self._safe_float(data.get("Prix_min"))
                pmax = self._safe_float(data.get("Prix_max"))
                p2m = self._safe_float(data.get("Prix_m2_moy"))
                p2med = self._safe_float(data.get("Prix_m2_med"))
                p2min = self._safe_float(data.get("Prix_m2_min"))
                p2max = self._safe_float(data.get("Prix_m2_max"))
                ecart = data.get("Ecart_type_prix")
                ecart_str = f"{self._safe_float(ecart):.0f} euros"
                prefix = loc(data["Commune"], year)
                # Une information par phrase (plus clair + meilleur pour BERT/fallback)
                sentences.extend(
                    [
                        prefix + f"le prix moyen d'un bien est de {pmoy:.0f} euros.",
                        prefix + f"le prix médian d'un bien est de {pmed:.0f} euros.",
                        prefix + f"le prix minimum enregistré est de {pmin:.0f} euros.",
                        prefix + f"le prix maximum est de {pmax:.0f} euros.",
                        prefix + f"les prix vont de {pmin:.0f} à {pmax:.0f} euros.",
                        prefix + f"le prix moyen au m² est de {p2m:.0f} euros.",
                        prefix + f"le prix médian au m² est de {p2med:.0f} euros.",
                        prefix + f"le prix au m² minimum est de {p2min:.0f} euros le m².",
                        prefix + f"le prix au m² maximum est de {p2max:.0f} euros le m².",
                        # Phrase demandée explicitement
                        prefix + f"La fourchette des prix au m² va de {p2min:.0f} à {p2max:.0f} euros le m².",
                        prefix + f"L'écart-type des prix est de {ecart_str}.",
                    ]
                )

        elif question_type == "prix_appartement":
            for key in keys:
                data = self.data_loader.get_data_by_key(key, year=year)
                if not data:
                    continue
                pa_med = self._safe_float(data.get("Prix_med_appartement"))
                pa_moy = self._safe_float(data.get("Prix_moy_appartement"))
                pam2_med = self._safe_float(data.get("Prix_m2_med_appartement"))
                pam2_moy = self._safe_float(data.get("Prix_m2_moy_appartement"))
                surf = self._safe_float(data.get("Surface_moy_appartement"))
                nb_a = self._safe_int(data.get("Nb_ventes_appartements"))
                prefix = loc(data["Commune"], year)
                sentences.extend(
                    [
                        prefix + f"le prix médian des appartements est de {pa_med:.0f} euros.",
                        prefix + f"le prix moyen des appartements est de {pa_moy:.0f} euros.",
                        prefix + f"le prix médian au m² des appartements est de {pam2_med:.0f} euros.",
                        prefix + f"le prix moyen au m² des appartements est de {pam2_moy:.0f} euros.",
                        prefix + f"la surface moyenne des appartements vendus est de {surf:.0f} m².",
                        prefix + f"le nombre de ventes d'appartements est de {nb_a}.",
                    ]
                )

        elif question_type == "prix_maison":
            for key in keys:
                data = self.data_loader.get_data_by_key(key, year=year)
                if not data:
                    continue
                pm_med = self._safe_float(data.get("Prix_med_maison"))
                pm_moy = self._safe_float(data.get("Prix_moy_maison"))
                pmm2_med = self._safe_float(data.get("Prix_m2_med_maison"))
                pmm2_moy = self._safe_float(data.get("Prix_m2_moy_maison"))
                surf = self._safe_float(data.get("Surface_moy_maison"))
                terr = self._safe_float(data.get("Terrain_moy_maison"))
                nb_m = self._safe_int(data.get("Nb_ventes_maisons"))
                prefix = loc(data["Commune"], year)
                sentences.extend(
                    [
                        prefix + f"le prix médian des maisons est de {pm_med:.0f} euros.",
                        prefix + f"le prix moyen des maisons est de {pm_moy:.0f} euros.",
                        prefix + f"le prix médian au m² des maisons est de {pmm2_med:.0f} euros.",
                        prefix + f"le prix moyen au m² des maisons est de {pmm2_moy:.0f} euros.",
                        prefix + f"la surface moyenne des maisons vendues est de {surf:.0f} m².",
                        prefix + f"le terrain moyen est de {terr:.0f} m².",
                        prefix + f"le nombre de ventes de maisons est de {nb_m}.",
                    ]
                )

        elif question_type == "surface":
            for key in keys:
                data = self.data_loader.get_data_by_key(key, year=year)
                if not data:
                    continue
                si = self._safe_float(data.get("Surface_moy_interieur"))
                si_med = self._safe_float(data.get("Surface_med_interieur"))
                st = self._safe_float(data.get("Surface_moy_terrain"))
                st_med = self._safe_float(data.get("Surface_med_terrain"))
                prefix = loc(data["Commune"], year)
                sentences.extend(
                    [
                        prefix + f"la surface intérieure moyenne des biens vendus est de {si:.0f} m².",
                        prefix + f"la surface intérieure médiane des biens vendus est de {si_med:.0f} m².",
                        prefix + f"la surface de terrain moyenne est de {st:.0f} m².",
                        prefix + f"la surface de terrain médiane est de {st_med:.0f} m².",
                    ]
                )

        elif question_type == "population":
            for key in keys:
                data = self.data_loader.get_data_by_key(key, year=year)
                if not data:
                    continue
                pop = data.get("population")
                seg = data.get("Segment_Commune")
                ev_pop = data.get("evolution_population")
                coords = data.get("coordonnees")
                debut, fin = self.data_loader.get_year_spectrum()
                prefix = loc(data["Commune"], year)
                lat, lon = coords.get("latitude_mairie"), coords.get("longitude_mairie")
                sentences.extend(
                    [
                        prefix + f"la population est de {self._safe_int(pop)} habitants.",
                        prefix + f"La commune fait partie du segment \"{seg}\" (classification par taille de population).",
                        prefix + f"L'évolution de la population sur la période {debut}-{fin} est de {ev_pop.get('pct'):+.1f}%.",
                        prefix + f"Les coordonnées de la mairie sont {lat}° de latitude et {lon}° de longitude.",
                    ]
                )

        elif question_type == "ventes":
            for key in keys:
                data = self.data_loader.get_data_by_key(key, year=year)
                if not data:
                    continue
                nb = self._safe_int(data.get("Nb_total_ventes"))
                nm = self._safe_int(data.get("Nb_ventes_maisons"))
                na = self._safe_int(data.get("Nb_ventes_appartements"))
                part = (nm / nb * 100) if nb else 0
                prefix = loc(data["Commune"], year)
                if year:
                    sentences.append(prefix + f"il y a eu {nb} ventes.")
                else:
                    sentences.append(prefix + f"le nombre total de ventes est de {nb}.")
                sentences.append(prefix + f"le nombre de ventes de maisons est de {nm}.")
                sentences.append(prefix + f"le nombre de ventes d'appartements est de {na}.")
                sentences.append(prefix + f"la part de maisons est de {part:.0f}%.")

        elif question_type == "evolution":
            for key in keys:
                data = self.data_loader.get_data_by_key(key, year=year)
                if not data:
                    continue
                t = data.get("tendance") or "N/A"
                ev = data.get("evolution_prix_m2")
                pct = ev.get("pct")
                pctstr = f"{pct:+.1f}%"
                ann_hausse = data.get("annee_plus_forte_hausse_prix")
                ann_baisse = data.get("annee_plus_forte_baisse_prix")
                ev_ventes = data.get("evolution_ventes")
                pct_ventes = ev_ventes.get("pct")
                var_ventes = data.get("variation_annuelle_ventes")
                debut, fin = self.data_loader.get_year_spectrum()
                vv_str = f"{self._safe_float(var_ventes):+.1f}%"
                if year:
                    prefix = loc(data["Commune"], year)
                    sentences.append(prefix + f"le marché immobilier est en {t}.")
                else:
                    prefix = f"À {data['Commune']}, "
                    sentences.append(prefix + f"le marché immobilier est en {t}.")
                sentences.append(prefix + f"sur la période {debut}-{fin}, l'évolution du prix au m² est de {pctstr}.")
                sentences.append(prefix + f"l'année avec la plus forte hausse des prix est {ann_hausse}.")
                sentences.append(prefix + f"l'année avec la plus forte baisse des prix est {ann_baisse}.")
                sentences.append(prefix + f"la variation annuelle des ventes est de {vv_str}.")
                sentences.append(prefix + f"l'évolution du nombre de ventes sur la période est de {pct_ventes:+.1f}%.")

        elif question_type == "general":
            for key in keys:
                data = self.data_loader.get_data_by_key(key, year=year)
                if not data:
                    continue
                pmed = self._safe_float(data.get("Prix_med"))
                p2med = self._safe_float(data.get("Prix_m2_med"))
                p2med_grp = self._safe_float(data.get("Prix_med_groupe_pop"))
                nb = self._safe_int(data.get("Nb_total_ventes"))
                var_p2 = data.get("variation_annuelle_prix_m2")
                var_v = data.get("variation_annuelle_ventes")
                t = data.get("tendance") or "N/A"
                debut, fin = self.data_loader.get_year_spectrum()
                vp2 = f"{self._safe_float(var_p2):+.1f}%" if var_p2 is not None else "N/A"
                vv = f"{self._safe_float(var_v):+.1f}%" if var_v is not None else "N/A"
                s = (loc(data["Commune"], year) if year else f"Pour {data['Commune']}, ") + (f"avec les données de {fin}, " if not year else "")
                s += f"le prix médian d'un bien est de {pmed:.0f} euros et le prix médian au m² est de {p2med:.0f} euros. "
                s += f"La tendance du marché est en {t} entre {debut} à {fin}. La variation annuelle du prix au m² entre {fin-1} et {fin} est de {vp2} et celle des ventes est de {vv}. "
                s += f"Le nombre total de ventes est de {nb}. Le nombre de biens vendus est de {nb}. "
                pop = data.get("population")
                seg = data.get("Segment_Commune")
                ecart_pct = data.get("Ecart_pct_prix_m2_vs_groupe")
                s += f"La population est de {self._safe_int(pop)} habitants. "
                s += f"La commune fait partie du segment \"{seg}\". "
                s += f"Par rapport aux villes de même population, le prix médian au m² est à {self._safe_float(ecart_pct):+.1f}% . "
                s += f"Le prix médian au m² du groupe de population (référence) est de {p2med_grp:.0f} euros. "
                sentences.append(s)

        year_unavailable = year is not None and len(keys) > 0 and len(sentences) == 0
        return (" ".join(sentences) if sentences else None, year_unavailable)

    def answer(self, question: str, verbose: bool = True):
        debut, fin = self.data_loader.get_year_spectrum()
        q_type = self.detect_question_type(question)
        if verbose:
            print(f"\nQuestion : {question}")
            print(f"Type détecté : {q_type}")

        keys = self.data_loader.find_communes(question)
        dept = self._get_dept_from_question(question)
        keys_by_commune = {}
        for k in keys:
            c = k.split("|")[0]
            keys_by_commune.setdefault(c, []).append(k)
        homonym_communes = [c for c, kk in keys_by_commune.items() if len(kk) > 1]
        if homonym_communes and dept is None:
            parts = [
                f"{c} (départements {', '.join(kk.split('|')[1] for kk in keys_by_commune[c])})"
                for c in homonym_communes
            ]
            msg = "La commune suivante existe dans plusieurs départements : " + "; ".join(parts) + ". Précisez le département dans la question (ex: département/dept/dpt 01)."
            if verbose:
                print("De quel département souhaitez-vous les informations ?")
            return msg

        year = self.extract_year(question)
        if year is not None:
            if debut is not None and fin is not None and (year < debut or year > fin):
                return f"Année incorrecte, choisissez une année dans le spectre {debut} - {fin}"

        context, year_unavailable = self.build_context_as_text(
            question,
            q_type,
            found_keys=keys,
            year=year,
            dept=dept,
        )
        if year_unavailable:
            return f"Données indisponible, essayer une autre année dans le spectre {debut} - {fin}"
        if context is None:
            return "Désolé, je n'ai pas trouvé de commune dans votre question. Vérifiez l'orthographe (les accents sont reconnus, ex. Orléans). Si la commune n'est pas dans notre jeu de données (certaines villes peuvent être absentes), nous ne pourrons pas répondre."

        if verbose:
            print(f"Contexte : {context}")

        if q_type == "general":
            return {
                "answer": context,
                "context_used": context,
                "source": "données",
            }

        result = self.bert_qa.extract_answer(context, question)
        if result and result.get("answer") and len(str(result["answer"]).strip()) > 2:
            result["source"] = "BERT"
            result["context_used"] = context
            return result

        if q_type == "ventes":
            m = re.search(r"(?:nombre total de ventes|nombre de biens vendus à [A-Za-zÀ-ÿ\s'-]+) est de (\d+)", context)
            if m:
                return {
                    "answer": m.group(1),
                    "score": 0.95,
                    "source": "fallback",
                    "context_used": context,
                }

        fb = self._fallback_answer(question, context, q_type)
        if isinstance(fb, dict):
            fb["source"] = "fallback"
            fb["context_used"] = context
            return fb
        return {"answer": fb, "context_used": context, "source": "fallback"}