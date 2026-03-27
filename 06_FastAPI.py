"""
06_FastAPI.py - API REST pour l'Assistant Immobilier

Utilise 03_qa_system.py (Q&A DVF), 02_pdfs.py (RAG PDF), 04_resumes.py (résumé), 05_sentiment.py (sentiment) à la racine.

Lancement : depuis la racine du projet :
  uvicorn 06_FastAPI:app --reload --host 0.0.0.0 --port 8000
"""
import base64
import os
from io import BytesIO
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import importlib
_mod_qa = importlib.import_module("03_qa_system")
ImmobilierChatbot = _mod_qa.ImmobilierChatbot

# Modules à la racine (noms commençant par chiffre → importlib)
_mod_pdf = importlib.import_module("02_pdfs")
PDFRagEngine = _mod_pdf.PDFRagEngine
answer_from_faiss_results = _mod_pdf.answer_from_faiss_results

_mod_resumes = importlib.import_module("04_resumes")
resumer_texte = _mod_resumes.resumer_texte
resumer_pdf = _mod_resumes.resumer_pdf

_mod_sentiment = importlib.import_module("05_sentiment")
analyser_sentiment = _mod_sentiment.analyser_sentiment

BASE_DIR = Path(__file__).resolve().parent
# Par défaut : json_par_ville.json (base de données du projet)
JSON_PATH = os.environ.get("JSON_PAR_VILLE_PATH", str(BASE_DIR / "json_par_ville.json"))

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)

class AskResponse(BaseModel):
    question: str
    answer: str
    score: float | None = None
    source: str | None = None
    context_used: str | None = Field(
        default=None,
        description="Texte construit depuis le JSON DVF (contexte envoyé à BERT / affiché en détail)",
    )
    sentence: str | None = Field(
        default=None,
        description="Phrase du contexte contenant la réponse BERT (recherche textuelle)",
    )

class AskPdfResultItem(BaseModel):
    text: str
    source: str
    distance: float

class AskPdfResponse(BaseModel):
    question: str
    results: list[AskPdfResultItem]
    pdf_rag_ready: bool
    answer: str | None = None

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=10)
    longueur: str = Field(default="moyen", description="court | moyen | long")

class SummarizePdfRequest(BaseModel):
    """PDF encodé en base64 (évite python-multipart si pip install échoue)."""
    pdf_base64: str = Field(..., description="Contenu du PDF encodé en base64")
    longueur: str = Field(default="moyen", description="court | moyen | long")

class SummarizeResponse(BaseModel):
    summary: str
    longueur: str

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Texte à analyser (ex. annonce immobilière)")

class SentimentDetailItem(BaseModel):
    label: str
    score: float

class SentimentResponse(BaseModel):
    label: str = Field(..., description="positive | negative | neutral")
    score: float = Field(..., description="Confiance 0–1")
    details: list[SentimentDetailItem] = Field(..., description="Scores par classe")

class HealthResponse(BaseModel):
    status: str
    json_loaded: bool
    chatbot_ready: bool
    pdf_rag_ready: bool
    json_path: str

_chatbot = None
_pdf_rag: PDFRagEngine | None = None

def get_chatbot():
    global _chatbot
    if _chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot non initialisé")
    return _chatbot

def get_pdf_rag():
    """Retourne le moteur RAG PDF s'il est prêt, sinon None."""
    global _pdf_rag
    return _pdf_rag if (_pdf_rag is not None and _pdf_rag.is_ready) else None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chatbot, _pdf_rag
    if os.path.exists(JSON_PATH):
        try:
            _chatbot = ImmobilierChatbot(JSON_PATH)
            # Eviter les caractères Unicode qui peuvent casser l'encodage terminal (Windows CP1252)
            print(f"OK Chatbot charge : {JSON_PATH}")
        except Exception as e:
            import traceback
            print(f"WARN Erreur au chargement du chatbot : {e}")
            traceback.print_exc()
            _chatbot = None
    else:
        print(f"WARN Fichier introuvable : {JSON_PATH}")
        _chatbot = None
    try:
        _pdf_rag = PDFRagEngine()
        if _pdf_rag.is_ready:
            print("OK RAG PDF charge (index.faiss / chunks.pkl)")
        else:
            _pdf_rag = None
            print("INFO RAG PDF : index non trouve (optionnel). Lancez 02_pdfs.py pour construire l'index.")
    except Exception as e:
        print(f"WARN RAG PDF : {e}")
        _pdf_rag = None

    # Précharger les modèles ML au démarrage (évite la latence au 1er appel)
    print("Préchargement des modèles (résumé, sentiment)...")
    try:
        _mod_resumes._get_summarizer()
        print("OK Modèle résumé chargé")
    except Exception as e:
        print(f"WARN Préchargement résumé : {e}")
    try:
        _mod_sentiment._get_analyzer()
        print("OK Modèle sentiment chargé")
    except Exception as e:
        print(f"WARN Préchargement sentiment : {e}")
    if _pdf_rag is not None and _pdf_rag.is_ready:
        try:
            _mod_pdf._get_embedding_model()
            _mod_pdf._get_pdf_answer_llm()
            print("OK Modèles RAG PDF (embedding + LLM) chargés")
        except Exception as e:
            print(f"WARN Préchargement RAG PDF : {e}")

    yield
    _chatbot = None
    _pdf_rag = None
    print("API arrêtée.")

app = FastAPI(title="API Immobilier - Assistant Q&A + RAG PDF", version="1.1.0", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "message": "API Assistant Immobilier",
        "docs": "/docs",
        "health": "/health",
        "ask": "POST /ask",
        "ask_pdf": "POST /ask_pdf",
        "summarize": "POST /summarize",
        "summarize_pdf": "POST /summarize_pdf",
        "sentiment": "POST /sentiment",
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    json_exists = os.path.exists(JSON_PATH)
    chatbot_ok = _chatbot is not None
    pdf_rag_ok = _pdf_rag is not None and _pdf_rag.is_ready
    status = "ok" if (json_exists and chatbot_ok) else "degraded"
    return HealthResponse(
        status=status,
        json_loaded=json_exists,
        chatbot_ready=chatbot_ok,
        pdf_rag_ready=pdf_rag_ok,
        json_path=JSON_PATH,
    )

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    chatbot = get_chatbot()
    result = chatbot.answer(req.question, verbose=False)
    if isinstance(result, str):
        return AskResponse(
            question=req.question,
            answer=result,
            score=None,
            source=None,
            context_used=None,
            sentence=None,
        )
    if isinstance(result, dict):
        return AskResponse(
            question=req.question,
            answer=str(result.get("answer", "Réponse non disponible")),
            score=result.get("score"),
            source=result.get("source"),
            context_used=result.get("context_used"),
            sentence=result.get("sentence"),
        )
    return AskResponse(
        question=req.question,
        answer=str(result),
        score=None,
        source=None,
        context_used=None,
        sentence=None,
    )


@app.post("/ask_pdf", response_model=AskPdfResponse)
async def ask_pdf(req: AskRequest):
    """Recherche sémantique dans les PDFs indexés (RAG). Retourne les k extraits les plus pertinents."""
    pdf_rag = get_pdf_rag()
    if pdf_rag is None:
        return AskPdfResponse(question=req.question, results=[], pdf_rag_ready=False, answer=None)
    results = pdf_rag.search(req.question, k=5)
    structured_answer = None
    # Enrichit la liste des extraits avec une réponse structurée via LLM HF.
    try:
        structured_answer = answer_from_faiss_results(req.question, results)
    except Exception as e:
        print(f"WARN answer_from_faiss_results: {e}")
    return AskPdfResponse(
        question=req.question,
        results=[AskPdfResultItem(text=r["text"], source=r["source"], distance=r["distance"]) for r in results],
        pdf_rag_ready=True,
        answer=structured_answer,
    )


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):
    """Résume un texte (longueur : court, moyen, long)."""
    summary = resumer_texte(req.text, longueur=req.longueur)
    return SummarizeResponse(summary=summary, longueur=req.longueur)


@app.post("/summarize_pdf", response_model=SummarizeResponse)
async def summarize_pdf(req: SummarizePdfRequest):
    """
    Résume un PDF envoyé en base64 (JSON).
    Pas besoin de python-multipart. Exemple : {"pdf_base64": "<contenu base64>", "longueur": "moyen"}.
    """
    try:
        content = base64.b64decode(req.pdf_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Contenu base64 invalide : {e}")
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Le contenu ne semble pas être un PDF valide")
    summary = resumer_pdf(BytesIO(content), longueur=req.longueur)
    return SummarizeResponse(summary=summary, longueur=req.longueur)


@app.post("/sentiment", response_model=SentimentResponse)
async def sentiment(req: SentimentRequest):
    """Analyse le sentiment d'un texte (ex. annonce immobilière). Retourne label (positive/negative/neutral), score et détails."""
    label, score, details = analyser_sentiment(req.text)
    if not details and label.startswith("[Erreur"):
        raise HTTPException(status_code=500, detail=label)
    return SentimentResponse(
        label=label,
        score=score,
        details=[SentimentDetailItem(label=d["label"], score=d["score"]) for d in details],
    )
