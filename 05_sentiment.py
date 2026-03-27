"""
05_sentiment.py - Analyse de sentiment (export de Script/05_Sentiment.ipynb)

Fonction : analyser_sentiment(texte) → label, score, details.
Utilisable en CLI, par 06_FastAPI (POST /sentiment) ou tout autre script.
"""
from __future__ import annotations

# Modèle chargé à la demande (lourd)
_analyzer = None

SENTIMENT_MODEL = "cardiffnlp/camembert-base-tweet-sentiment-fr"
MAX_INPUT_LENGTH = 2000


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        from transformers import pipeline
        _analyzer = pipeline(
            "text-classification",
            model=SENTIMENT_MODEL,
            top_k=None,
        )
    return _analyzer


def analyser_sentiment(texte: str) -> tuple[str, float, list[dict]]:
    """
    Analyse le sentiment d'un texte (ex. annonce immobilière).

    Retourne : (label, score, details)
    - label : 'positive', 'negative' ou 'neutral'
    - score : probabilité (0-1) du label dominant
    - details : liste [{"label": "...", "score": float}, ...] pour chaque classe
    """
    if not texte or not texte.strip():
        return "", 0.0, []

    texte = texte.strip()
    if len(texte) > MAX_INPUT_LENGTH:
        texte = texte[:MAX_INPUT_LENGTH]

    try:
        analyzer = _get_analyzer()
        result = analyzer(texte)[0]
        label = result[0]["label"].lower()
        score = float(result[0]["score"])
        details = [{"label": r["label"].lower(), "score": round(r["score"], 4)} for r in result]
        return label, score, details
    except Exception as e:
        return f"[Erreur : {e}]", 0.0, []


if __name__ == "__main__":
    annonce = """
    Magnifique appartement rénové au cœur du centre-ville, à deux pas des commerces et transports.
    Séjour lumineux, cuisine équipée, deux chambres. Calme et pièce en plus. Idéal pour famille !
    """
    label, score, details = analyser_sentiment(annonce)
    labels_fr = {"positive": "positif", "negative": "négatif", "neutral": "neutre"}
    print(f"Sentiment dominant : {labels_fr.get(label, label).upper()} (confiance : {score*100:.1f} %)")
    for d in details:
        print(f"  - {labels_fr.get(d['label'], d['label'])} : {d['score']*100:.1f} %")
