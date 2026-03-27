"""
04_resumes.py - Résumé automatique de textes et PDF (export de Script/04_Resumes.ipynb)

Fonctions : extraire_texte_pdf, resumer_texte, resumer_pdf.
Utilisable en CLI, par 06_FastAPI (POST /summarize, /summarize_pdf) ou tout autre script.
"""
from __future__ import annotations

from io import BytesIO

import fitz  # PyMuPDF

# Modèle chargé à la demande (lourd)
_summarizer = None

# --- Configuration (alignée sur le notebook) ---
SUM_MODEL = "plguillou/t5-base-fr-sum-cnndm"
# Pourcentages de la taille du texte d'entrée (en tokens) pour la longueur du résumé
LONGUEURS = {
    "court": 0.25,   # 10%
    "moyen": 0.50,   # 25%
    "long": 0.75,    # 50%
}
MAX_INPUT_TOKENS = 512
MIN_OUTPUT_TOKENS = 15
MAX_OUTPUT_TOKENS = 350
# Génération : num_beams=1 (rapide) ou 2-4 (meilleure qualité). 2 = bon compromis vitesse/qualité.
NUM_BEAMS = 2


def _get_summarizer():
    """Charge le modèle de résumé (T5) sans pipeline 'summarization' (absent en transformers récent)."""
    global _summarizer
    if _summarizer is None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(SUM_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL)
        # FP16 sur GPU : ~1.5-2x plus rapide
        if torch.cuda.is_available():
            model = model.to("cuda").half()
        _summarizer = (tokenizer, model)
    return _summarizer


def extraire_texte_pdf(pdf_source: str | BytesIO | bytes) -> str:
    """
    Extrait le texte brut d'un PDF.
    pdf_source : chemin fichier (str), BytesIO, ou bytes.
    """
    if isinstance(pdf_source, str):
        doc = fitz.open(pdf_source)
    else:
        raw = pdf_source.getvalue() if hasattr(pdf_source, "getvalue") else pdf_source
        doc = fitz.open(stream=raw, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text.strip()


def resumer_texte(
    texte: str,
    longueur: str = "moyen",
    max_length: int | None = None,
    min_length: int | None = None,
) -> str:
    """
    Résume un texte.
    - texte : str
    - longueur : "court", "moyen" ou "long" de la taille du texte d'entrée
    - max_length, min_length : override optionnel (en tokens), désactivent le calcul par pourcentage
    """
    if not texte or not texte.strip():
        return ""

    texte = texte.strip()
    if len(texte) > MAX_INPUT_TOKENS * 4:
        texte = texte[: MAX_INPUT_TOKENS * 4]

    try:
        tokenizer, model = _get_summarizer()
        inputs = tokenizer(
            texte,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(model.device)
        input_tokens = inputs["input_ids"].shape[1]

        if min_length is not None and max_length is not None:
            min_l, max_l = min_length, max_length
        else:
            ratio = LONGUEURS.get((longueur or "moyen").lower().strip(), LONGUEURS["moyen"])
            target = max(MIN_OUTPUT_TOKENS, int(input_tokens * ratio))
            target = min(MAX_OUTPUT_TOKENS, target)
            min_l = max(MIN_OUTPUT_TOKENS, int(target * 0.5))
            max_l = max(min_l + 10, target)
            if min_length is not None:
                min_l = min_length
            if max_length is not None:
                max_l = max_length

        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_l,
            min_new_tokens=min_l,
            do_sample=False,
            num_beams=NUM_BEAMS,
            early_stopping=True,
        )
        out = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        return out or texte[:max_l]
    except Exception as e:
        return f"[Erreur de résumé : {e}]"


def resumer_pdf(
    pdf_source: str | BytesIO | bytes,
    longueur: str = "moyen",
    max_length: int | None = None,
    min_length: int | None = None,
) -> str:
    """
    Extrait le texte du PDF puis le résume.
    - pdf_source : chemin fichier, BytesIO ou bytes
    - longueur : "court", "moyen" ou "long"
    """
    try:
        texte = extraire_texte_pdf(pdf_source)
    except Exception as e:
        return f"[Erreur d'extraction PDF : {e}]"

    if not texte:
        return "[Aucun texte extrait du PDF (image ?)]"

    return resumer_texte(texte, longueur=longueur, max_length=max_length, min_length=min_length)


if __name__ == "__main__":
    texte_exemple = """
    Le marché immobilier français a connu une évolution significative en 2023. Les prix au m² ont continué d'augmenter dans les grandes métropoles, notamment à Paris et Lyon, tandis que certaines zones rurales ont vu une stabilisation. Les taux d'intérêt élevés ont ralenti les transactions, avec une baisse d'environ 15 % du nombre de ventes par rapport à 2022. Les experts prévoient une reprise progressive en 2024, sous réserve d'une baisse des taux et d'une amélioration du pouvoir d'achat.
    """
    print("--- Court (10%) ---")
    print(resumer_texte(texte_exemple, longueur="court"))
    print("\n--- Moyen (25%) ---")
    print(resumer_texte(texte_exemple, longueur="moyen"))
    print("\n--- Long (50%) ---")
    print(resumer_texte(texte_exemple, longueur="long"))
