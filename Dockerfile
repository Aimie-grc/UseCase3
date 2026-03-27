# API Assistant Immobilier - Dockerfile
# Build : docker build -t api-immobilier .
# Run   : docker run -p 8001:8001 api-immobilier

FROM python:3.11-slim

WORKDIR /app

# Dépendances système (PyMuPDF, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python - 2 étapes pour profiter du cache et limiter les erreurs I/O
COPY requirements.txt .
# Étape 1 : PyTorch CPU (le plus lourd) - si OK, cette couche est en cache
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# Étape 2 : le reste (torch déjà installé sera ignoré)
RUN pip install --no-cache-dir -r requirements.txt

# Code source
COPY *.py .
COPY json_par_ville.json .

# Fichiers RAG optionnels (montés via volume si présents)
# index.faiss et chunks.pkl doivent être au même niveau que les .py

EXPOSE 8001

# Les modèles HuggingFace sont téléchargés au 1er lancement
# (stockés dans ~/.cache/huggingface)
ENV HF_HOME=/app/.cache/huggingface

# Préchargement des modèles au démarrage (lifespan FastAPI)
CMD ["uvicorn", "06_FastAPI:app", "--host", "0.0.0.0", "--port", "8001"]
