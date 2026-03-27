"""
02_pdfs.py - RAG sur PDFs immobiliers (export de Script/02_PDFs.ipynb)

Fonctions : extraction PDF (PyMuPDF), chunking (tiktoken), index FAISS, recherche sémantique.
Utilisable en CLI, par 06_FastAPI (endpoint /ask_pdf) ou tout autre script.

Usage :
  - Charger un index existant : engine = PDFRagEngine(); engine.search("question")
  - Construire l'index : voir build_index_from_sources() ou traite_pdfs().
"""
from __future__ import annotations

import os
import re
import pickle
import unicodedata
from io import BytesIO
from pathlib import Path
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import fitz  # PyMuPDF
import tiktoken
import numpy as np
import faiss

# SentenceTransformer chargé à la demande (lourd)
_embedding_model = None

# ---------------------------------------------------------------------------
# LLM Hugging Face (réponse structurée a partir des extraits RAG)
# ---------------------------------------------------------------------------
# Modèle relativement petit par défaut pour limiter la charge.
# Vous pouvez le changer via variable d'environnement `PDF_ANSWER_LLM_MODEL_NAME`.
_pdf_answer_llm = None
_PDF_ANSWER_LLM_MODEL_NAME = os.environ.get("PDF_ANSWER_LLM_MODEL_NAME", "google/flan-t5-base")
_PDF_ANSWER_LLM_MAX_NEW_TOKENS = int(os.environ.get("PDF_ANSWER_LLM_MAX_NEW_TOKENS", "256"))
_PDF_ANSWER_LLM_MIN_NEW_TOKENS = int(os.environ.get("PDF_ANSWER_LLM_MIN_NEW_TOKENS", "32"))
_PDF_ANSWER_LLM_MAX_INPUT_CHARS = int(os.environ.get("PDF_ANSWER_LLM_MAX_INPUT_CHARS", "3500"))

# --- Configuration : racine du projet = dossier de ce fichier ---
_BASE_DIR = Path(__file__).resolve().parent

CHUNK_MAX_TOKENS = 200
CHUNK_OVERLAP_RATIO = 0.15
MIN_FRAGMENT_LENGTH = 2
RAG_TOP_K = 5
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", str(_BASE_DIR / "index.faiss"))
CHUNKS_PATH = os.environ.get("CHUNKS_PATH", str(_BASE_DIR / "chunks.pkl"))


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


# ---------------------------------------------------------------------------
# Récupération des liens PDF (sources du notebook)
# ---------------------------------------------------------------------------

def get_pdf_links_data_gouv() -> list[str]:
    """Liens Data.gouv (API datasets)."""
    return [
        "https://www.data.gouv.fr/api/1/datasets/r/d573456c-76eb-4276-b91c-e6b9c89d6656",
        "https://www.data.gouv.fr/api/1/datasets/r/087ec735-74fd-48a7-a82e-0b1cd3ea6fe9",
    ]


def get_pdf_links_nafu() -> list[str]:
    """Liens Observatoire NAFU (page HTML)."""
    page_url = "https://observatoire-nafu.fr/espaces_nafu/marche-immobilier/"
    pdf_links = []
    try:
        response = requests.get(page_url, timeout=30)
        soup = BeautifulSoup(response.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf"):
                pdf_links.append(urljoin(page_url, href))
    except Exception as e:
        print(f"[AVERTISSEMENT] NAFU : {e}")
    return pdf_links


def get_pdf_links_notaire() -> list[str]:
    """Liens Notaires (Calameo API)."""
    pdf_links = []
    content = True
    page = 0
    try:
        while content:
            api_url = (
                f"https://d.calameo.com/pinwheel/library/get?order=0&way=desc&page={page}&step=8"
                f"&type=subscription&id=5169166"
            )
            r = requests.get(api_url, timeout=30)
            data = r.json()
            if not data.get("content", {}).get("books"):
                content = False
            else:
                page += 1
                for book in data["content"]["books"]:
                    try:
                        date = datetime.strptime(book["date"], "%Y-%m-%d")
                        if date.year >= 2020:
                            pdf_links.append(f"https://www.calameo.com/download/{book['id']}")
                    except (KeyError, ValueError):
                        pass
    except Exception as e:
        print(f"[AVERTISSEMENT] Notaire/Calameo : {e}")
    return pdf_links


def get_all_pdf_links() -> list[str]:
    """Agrège toutes les sources de liens PDF."""
    links = []
    links.extend(get_pdf_links_data_gouv())
    links.extend(get_pdf_links_nafu())
    links.extend(get_pdf_links_notaire())
    return links


def download_pdfs(pdf_links: list[str]) -> list[tuple[BytesIO, str]]:
    """Télécharge chaque URL en PDF et retourne une liste de (BytesIO, url_source)."""
    pdfs = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
    }
    for pdf_url in pdf_links:
        h = dict(headers)
        if "calameo.com/download" in pdf_url:
            h["Referer"] = pdf_url.replace("download", "read")
        try:
            response = requests.get(pdf_url, headers=h, timeout=30)
            if response.status_code == 200:
                pdf_file = BytesIO(response.content)
                pdfs.append((pdf_file, pdf_url))
            else:
                print(f"[ERREUR téléchargement] {pdf_url} → code HTTP {response.status_code}")
        except requests.RequestException as e:
            print(f"[ERREUR téléchargement] {pdf_url} → {e}")
    return pdfs


# ---------------------------------------------------------------------------
# Normalisation et chunking
# ---------------------------------------------------------------------------

def normalize_text(text: str, min_fragment_length: int = MIN_FRAGMENT_LENGTH) -> str:
    """Normalise le texte pour le chunking (phrases, accents, espaces)."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    fragments = re.split(r"\.\s+", text)
    clean = [f.strip() for f in fragments if len(f.strip()) > min_fragment_length]
    text = ". ".join(clean)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_tokens(
    text: str,
    source: str,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_ratio: float = CHUNK_OVERLAP_RATIO,
) -> list[dict]:
    """Découpe un texte en chunks de tokens avec overlap. Retourne [{"text": ..., "source": ...}, ...]."""
    try:
        encoder = tiktoken.get_encoding("cl100k_base")
    except Exception:
        encoder = tiktoken.get_encoding("gpt2")
    overlap_tokens = int(max_tokens * overlap_ratio)
    chunks = []
    tokens = encoder.encode(normalize_text(text))
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tok = tokens[start:end]
        chunk_text = encoder.decode(chunk_tok).strip()
        if chunk_text:
            chunks.append({"text": chunk_text, "source": source})
        start = end - overlap_tokens if end - overlap_tokens > start else end
    return chunks


# ---------------------------------------------------------------------------
# Réponse structurée (LLM) a partir des résultats FAISS
# ---------------------------------------------------------------------------
def _get_pdf_answer_llm():
    """
    Charge un LLM Hugging Face de facon lazy (seq2seq type T5/FLAN).
    Retourne un tuple (tokenizer, model).
    """
    global _pdf_answer_llm
    if _pdf_answer_llm is not None:
        return _pdf_answer_llm

    # Import local pour éviter de charger Transformers si on ne l'utilise pas.
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_PDF_ANSWER_LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(_PDF_ANSWER_LLM_MODEL_NAME)
    _pdf_answer_llm = (tokenizer, model)
    return _pdf_answer_llm


def answer_from_faiss_results(
    question: str,
    results: list[dict],
    *,
    max_new_tokens: int | None = None,
) -> str | None:
    """
    Transforme les extraits FAISS en une réponse structurée (en francais).

    `results` attend : [{"text": ..., "source": ..., "distance": ...}, ...]
    """
    if not results:
        return "[LLM indisponible] Aucun extrait RAG trouvé pour répondre à la question."

    max_new_tokens = _PDF_ANSWER_LLM_MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens

    # Contexte concis (FLAN-T5 gère mal les prompts trop longs)
    blocks: list[str] = []
    for i, r in enumerate(results[:3]):
        text = (r.get("text") or "").strip().replace("\n", " ")
        text = text[:400]
        if text:
            blocks.append(text)

    context = " ".join(blocks)[:_PDF_ANSWER_LLM_MAX_INPUT_CHARS]

    # Prompt simple : format que FLAN-T5 maîtrise mieux
    prompt = f"""Lis le contexte et réponds à la question en français. Sois concis et précis. Utilise uniquement les informations du contexte.

Contexte: {context}

Question: {question}

Réponse:"""

    try:
        tokenizer, model = _get_pdf_answer_llm()
        import torch

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=16,
                do_sample=False,
                num_beams=3,
                repetition_penalty=1.2,
                length_penalty=0.8,
            )
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if not decoded:
            return "[LLM indisponible] Réponse vide."
        return decoded
    except Exception as e:
        # On renvoie un message plutôt que de casser l'endpoint
        return f"[LLM indisponible] {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Index FAISS et recherche
# ---------------------------------------------------------------------------

def index_faiss(chunks: list[dict], model=None):
    """Construit l'index FAISS à partir des chunks. Retourne l'index faiss."""
    if model is None:
        model = _get_embedding_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    return index


def question_faiss(
    query: str,
    index,
    chunks: list[dict],
    k: int = RAG_TOP_K,
    model=None,
    verbose: bool = False,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """
    Recherche les k chunks les plus proches de la question.
    Retourne (résultats, distances, indices) avec résultats = [{"text", "source", "distance"}, ...].
    """
    if model is None:
        model = _get_embedding_model()
    query_norm = normalize_text(query)
    query_vec = model.encode([query_norm]).astype(np.float32)
    distances, indices = index.search(query_vec, min(k, index.ntotal))
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        c = chunks[idx]
        dist = float(distances[0][i])
        results.append({"text": c["text"], "source": c["source"], "distance": dist})
        if verbose:
            print(f"Résultat {i+1}:", c["text"][:200], "... | Source:", c["source"], "| Distance:", dist)
    return results, distances, indices


# ---------------------------------------------------------------------------
# Traitement PDF et sauvegarde / chargement
# ---------------------------------------------------------------------------

def traite_pdfs(pdf_list: list[tuple]) -> tuple:
    """
    pdf_list : liste de (fichier_pdf, source).
    Fichier peut être BytesIO ou file-like avec .read().
    Retourne (index_faiss, chunks).
    """
    all_chunks = []
    for pdf_path, source in pdf_list:
        try:
            raw = pdf_path.getvalue() if hasattr(pdf_path, "getvalue") else pdf_path.read()
            with fitz.open(stream=raw, filetype="pdf") as doc:
                all_text = ""
                for page in doc:
                    try:
                        text = page.get_text()
                        if text:
                            all_text += text + "\n"
                    except Exception as e:
                        print(f"[ERREUR page] {source} → {e}")
                if not all_text.strip():
                    print(f"[AVERTISSEMENT] {source} → aucun texte extrait")
                    continue
            chunks = chunk_tokens(all_text, source)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"[ERREUR extraction] {source} → {e}")
            continue
    if not all_chunks:
        raise ValueError("Aucun chunk produit : tous les PDFs ont échoué ou sont vides.")
    index = index_faiss(all_chunks)
    return index, all_chunks


def save_faiss_index(index, chunks: list[dict], index_path: str = FAISS_INDEX_PATH, chunks_path: str = CHUNKS_PATH) -> None:
    """Sauvegarde l'index FAISS et les chunks sur le disque."""
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Index sauvegardé : {index_path} | Chunks : {chunks_path}")


def load_faiss_index(index_path: str = FAISS_INDEX_PATH, chunks_path: str = CHUNKS_PATH) -> tuple:
    """Charge l'index FAISS et les chunks. Retourne (index, chunks) ou (None, None) si absent."""
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    return None, None


# ---------------------------------------------------------------------------
# Classe pour l'API : chargement unique, recherche à la demande
# ---------------------------------------------------------------------------

class PDFRagEngine:
    """
    Moteur RAG PDF : charge l'index et les chunks depuis le disque,
    expose search(query, k) pour l'API ou tout autre script.
    """

    def __init__(
        self,
        index_path: str = FAISS_INDEX_PATH,
        chunks_path: str = CHUNKS_PATH,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.embedding_model_name = embedding_model_name
        self._index, self._chunks = load_faiss_index(index_path, chunks_path)
        self._model = None

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._chunks is not None and len(self._chunks) > 0

    def search(self, query: str, k: int = RAG_TOP_K) -> list[dict]:
        """
        Recherche sémantique. Retourne [{"text", "source", "distance"}, ...].
        Si l'index n'est pas chargé, retourne [].
        """
        if not self.is_ready:
            return []
        if self._model is None:
            self._model = _get_embedding_model()
        results, _, _ = question_faiss(query, self._index, self._chunks, k=k, model=self._model, verbose=False)
        return results


def build_index_from_sources(index_path: str = FAISS_INDEX_PATH, chunks_path: str = CHUNKS_PATH) -> bool:
    """
    Récupère tous les liens PDF, télécharge, traite et sauvegarde l'index.
    Retourne True si succès, False sinon.
    """
    links = get_all_pdf_links()
    if not links:
        print("Aucun lien PDF trouvé.")
        return False
    pdfs = download_pdfs(links)
    if not pdfs:
        print("Aucun PDF téléchargé.")
        return False
    print(f"Téléchargés : {len(pdfs)} / {len(links)} PDF(s).")
    try:
        index, chunks = traite_pdfs(pdfs)
        save_faiss_index(index, chunks, index_path, chunks_path)
        return True
    except Exception as e:
        print(f"Erreur construction index : {e}")
        return False


if __name__ == "__main__":
    # Charger ou construire l'index, puis exécuter une question de test
    index, chunks = load_faiss_index()
    if index is None:
        print("Index absent. Construction depuis les sources (peut être long)...")
        if build_index_from_sources():
            index, chunks = load_faiss_index()
    if index is not None and chunks is not None:
        engine = PDFRagEngine()
        q = "Prix au m² médian des appartements anciens au 2e trimestre 2022 Orléans"
        for r in engine.search(q, k=3):
            print("---")
            print("Source:", r["source"])
            print("Distance:", r["distance"])
            print("Texte:", r["text"][:300], "...")
    else:
        print("Impossible de charger ou construire l'index.")
