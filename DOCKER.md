# Docker - Assistant Immobilier

## Fonctionnement

1. **Build** : L'image inclut Python, les dépendances (`pip install`) et le code.
2. **Run** : Au démarrage, uvicorn lance l'API. Les modèles Hugging Face sont téléchargés au **1er lancement** (ou préchargés par le lifespan).
3. **Cache** : Le volume `hf_cache` conserve les modèles entre les recréations de conteneur (plus de téléchargement après la 1ère fois).
4. **RAG PDF** : Si `index.faiss` et `chunks.pkl` sont présents dans le dossier, ils sont utilisés. Sinon, l'API tourne sans recherche sémantique PDF.

## Commandes

### API seule
```bash
docker build -t api-immobilier .
docker run -p 8001:8001 -v hf_cache:/app/.cache/huggingface api-immobilier
```

### API + Streamlit (recommandé)
```bash
docker-compose up --build
```
- API : http://localhost:8001
- Streamlit : http://localhost:8501

### Build en 2 étapes (si erreur I/O)
Le Dockerfile installe les deps en 2 RUN. Si l'étape 2 échoue, relancez le build : l'étape 1 (PyTorch) sera en cache.
```bash
docker-compose build
# Si erreur à l'étape 2, relancer :
docker-compose build
```

### Premier démarrage
Le 1er `docker-compose up` peut prendre **2 à 5 minutes** (téléchargement des modèles). Les suivants sont rapides grâce au cache.

## Variables d'environnement

| Variable | Défaut | Description |
|----------|--------|-------------|
| `JSON_PAR_VILLE_PATH` | `/app/json_par_ville.json` | Chemin du JSON DVF |
| `API_IMMOBILIER_URL` | `http://api:8001` | URL de l'API (pour Streamlit) |
| `FAISS_INDEX_PATH` | `./index.faiss` | Index FAISS RAG |
| `HF_HOME` | `/app/.cache/huggingface` | Cache Hugging Face |

## Ressources

Les modèles occupent ~3–4 Go en RAM. Le `docker-compose` limite à 4 Go ; ajuster si besoin.

## GPU

Pour utiliser un GPU (CUDA) et accélérer résumé/sentiment :
- Remplacer l'image de base par `nvidia/cuda:...` + PyTorch GPU
- Ou utiliser `--gpus all` avec une image compatible
