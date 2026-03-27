"""
07_streamlit_app.py - Interface utilisateur de l'Assistant Immobilier

Utilise l'API FastAPI (06_FastAPI) via requests pour le Q&A.

Lancement :
  1. Démarrer l'API : uvicorn 06_FastAPI:app --host 0.0.0.0 --port 8001
  2. Lancer Streamlit : streamlit run 07_streamlit_app.py

Ou depuis la racine du projet :
  streamlit run 07_streamlit_app.py
"""
import os
import base64
import html
import time
import requests
import streamlit as st

# --- Configuration ---
API_URL = os.environ.get("API_IMMOBILIER_URL", "http://127.0.0.1:8001")
TIMEOUT = 30
# Contourner le proxy pour localhost (souvent la cause de "impossible de joindre l'API")
_REQ_KW = {"timeout": TIMEOUT, "proxies": {"http": None, "https": None}}

# Page par défaut
if "page" not in st.session_state:
    st.session_state["page"] = "accueil"


def ask_api(question: str) -> dict | None:
    """Envoie une question à l'API et retourne la réponse."""
    try:
        r = requests.post(f"{API_URL}/ask", json={"question": question}, **_REQ_KW)
        if r.ok:
            return r.json()
        return {"error": f"Erreur {r.status_code}", "detail": r.text}
    except requests.RequestException as e:
        return {"error": "Impossible de joindre l'API", "detail": str(e)}


def sentiment_api(text: str) -> dict | None:
    """Envoie le texte de l'annonce à l'API (POST /sentiment)."""
    try:
        r = requests.post(f"{API_URL}/sentiment", json={"text": text}, **_REQ_KW)
        if r.ok:
            return r.json()
        return {"error": f"Erreur {r.status_code}", "detail": r.text}
    except requests.RequestException as e:
        return {"error": "Impossible de joindre l'API", "detail": str(e)}


def ask_pdf_api(question: str) -> dict | None:
    """Envoie une question pour recherche sémantique dans les PDFs (POST /ask_pdf)."""
    try:
        # Timeout plus long : FAISS + LLM prennent du temps
        kw = {**_REQ_KW, "timeout": 90}
        r = requests.post(f"{API_URL}/ask_pdf", json={"question": question}, **kw)
        if r.ok:
            return r.json()
        return {"error": f"Erreur {r.status_code}", "detail": r.text}
    except requests.RequestException as e:
        return {"error": "Impossible de joindre l'API", "detail": str(e)}


def summarize_api(text: str, longueur: str) -> dict | None:
    """Résume un texte (POST /summarize)."""
    try:
        r = requests.post(
            f"{API_URL}/summarize",
            json={"text": text, "longueur": longueur},
            **_REQ_KW,
        )
        if r.ok:
            return r.json()
        return {"error": f"Erreur {r.status_code}", "detail": r.text}
    except requests.RequestException as e:
        return {"error": "Impossible de joindre l'API", "detail": str(e)}


def summarize_pdf_api(pdf_base64: str, longueur: str) -> dict | None:
    """Résume un PDF encodé en base64 (POST /summarize_pdf)."""
    try:
        r = requests.post(
            f"{API_URL}/summarize_pdf",
            json={"pdf_base64": pdf_base64, "longueur": longueur},
            **_REQ_KW,
        )
        if r.ok:
            return r.json()
        return {"error": f"Erreur {r.status_code}", "detail": r.text}
    except requests.RequestException as e:
        return {"error": "Impossible de joindre l'API", "detail": str(e)}


st.set_page_config(
    page_title="Assistant Immobilier",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Thème boutons : bleu saphir (remplace le rouge des boutons primaires par défaut)
_SAPHIR = "#0F52BA"
_SAPHIR_FONCE = "#0A3570"
st.markdown(
    f"""
<style>
    /* Boutons primaires (type="primary") */
    button[kind="primary"],
    div[data-testid="stButton"] > button[kind="primary"],
    button[data-testid="baseButton-primary"] {{
        background-color: {_SAPHIR} !important;
        border-color: {_SAPHIR_FONCE} !important;
        color: #ffffff !important;
    }}
    button[kind="primary"]:hover,
    div[data-testid="stButton"] > button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover {{
        background-color: {_SAPHIR_FONCE} !important;
        border-color: #082a5c !important;
        color: #ffffff !important;
    }}
    /* Boutons secondaires (navigation, etc.) : style bleu saphir */
    button[kind="secondary"],
    div[data-testid="stButton"] > button[kind="secondary"],
    button[data-testid="baseButton-secondary"] {{
        border-color: {_SAPHIR} !important;
        color: {_SAPHIR} !important;
        background-color: transparent !important;
    }}
    button[kind="secondary"]:hover,
    div[data-testid="stButton"] > button[kind="secondary"]:hover,
    button[data-testid="baseButton-secondary"]:hover {{
        border-color: {_SAPHIR_FONCE} !important;
        color: {_SAPHIR_FONCE} !important;
        background-color: rgba(15, 82, 186, 0.08) !important;
    }}
</style>
""",
    unsafe_allow_html=True,
)

# --- Sidebar : navigation (toujours visible) ---
with st.sidebar:
    st.markdown(
        f"""
<div style="text-align:center;font-weight:800;font-size:1.75rem;color:{_SAPHIR};
letter-spacing:0.12em;margin-bottom:0.6rem;padding-bottom:0.5rem;
border-bottom:2px solid {_SAPHIR};">
SAFIR
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("### 🧭 Navigation")
    if st.button("🏠 Accueil", use_container_width=True):
        st.session_state["page"] = "accueil"
        st.rerun()
    if st.button("📊 Questions numériques", use_container_width=True):
        st.session_state["page"] = "questions"
        st.rerun()
    if st.button("🧾 Analyse d'annonce", use_container_width=True):
        st.session_state["page"] = "sentiment"
        st.rerun()
    if st.button("🔎 Recherche sémantique", use_container_width=True):
        st.session_state["page"] = "ask_pdf"
        st.rerun()
    if st.button("✍️ Résumé de texte", use_container_width=True):
        st.session_state["page"] = "summarize"
        st.rerun()
    if st.button("🗂️ Résumé de PDF", use_container_width=True):
        st.session_state["page"] = "summarize_pdf"
        st.rerun()


# --- Contenu selon la page ---
if st.session_state["page"] == "accueil":
    st.markdown("## 🏠 SAFIR, votre assistant immobilier")
    st.markdown(
        """
**SAFIR** vous aide à exploiter l'information immobilière en un seul endroit.

- **Données DVF** : interrogez les ventes et indicateurs par commune (prix, volumes, tendances, surfaces, contexte démographique).
- **Annonces** : analysez le ton et le sentiment d'un texte d'annonce.
- **Documents** : recherchez des passages pertinents dans vos PDF, ou obtenez des **résumés** de texte ou de fichiers PDF.

Utilisez le menu à gauche ou les raccourcis ci-dessous pour commencer.
"""
    )
    st.markdown("---")
    if st.button("📊 Accéder aux questions numériques", type="primary", use_container_width=True):
        st.session_state["page"] = "questions"
        st.rerun()

    st.markdown("### Accès rapide")
    btn_cols = st.columns(2)
    with btn_cols[0]:
        if st.button("🧾 Analyser une annonce", use_container_width=True):
            st.session_state["page"] = "sentiment"
            st.rerun()
    with btn_cols[0]:
        if st.button("🔎 Rechercher des informations textuelles", use_container_width=True):
            st.session_state["page"] = "ask_pdf"
            st.rerun()
    with btn_cols[1]:
        if st.button("✍️ Résumer un texte", use_container_width=True):
            st.session_state["page"] = "summarize"
            st.rerun()
    with btn_cols[1]:
        if st.button("🗂️ Résumer un fichier PDF", use_container_width=True):
            st.session_state["page"] = "summarize_pdf"
            st.rerun()

    st.markdown("---")
    st.caption("Projet 3 - Use Case")
else:
    if st.session_state["page"] == "sentiment":
        st.markdown("### 📝 Analyse d'annonce")
        st.markdown("Collez le texte de l'annonce immobilière ci-dessous, puis lancez l'analyse du sentiment de l'annonce.")
        st.markdown("---")

        annonce = st.text_area(
            "Texte de l'annonce",
            value=st.session_state.get("annonce", ""),
            placeholder="Ex : Magnifique appartement rénové au cœur du centre-ville...",
            height=220,
            max_chars=2000,
        )
        st.session_state["annonce"] = annonce

        analyser = st.button("Analyser l'annonce", type="primary")
        if analyser:
            if not annonce.strip():
                st.warning("Veuillez saisir un texte.")
            else:
                with st.spinner("Analyse en cours..."):
                    result = sentiment_api(annonce.strip())
                if not result:
                    st.error("Erreur inconnue lors de l'appel à l'API.")
                elif "error" in result:
                    st.error(f"**{result['error']}**")
                else:
                    label = result.get("label", "")
                    score = result.get("score")
                    details = result.get("details") or []
                    if score is not None:
                        st.success(f"Résultat : {label} (score={score:.2f})")
                    else:
                        st.success(f"Résultat : {label}")

                    if details:
                        st.markdown("Détails par classe :")
                        st.table(details)

        st.markdown("---")
        st.stop()

    if st.session_state["page"] == "ask_pdf":
        st.markdown("### 🔎 Recherche sémantique")
        st.markdown("Posez une question pour retrouver les passages pertinents dans les PDFs stockés dans notre base de données.")
        st.markdown("---")

        # Exemples : au clic, pré-remplit la zone de recherche
        ex1 = "C'est quoi les données DVF?"
        ex2 = "C'est quoi une disposition ?"
        ex_cols = st.columns(2)
        with ex_cols[0]:
            if st.button(ex1, key="ask_pdf_ex1", use_container_width=True):
                st.session_state["ask_pdf_question"] = ex1
                st.rerun()
        with ex_cols[1]:
            if st.button(ex2, key="ask_pdf_ex2", use_container_width=True):
                st.session_state["ask_pdf_question"] = ex2
                st.rerun()

        question = st.text_input(
            "Votre question",
            value=st.session_state.get("ask_pdf_question", ""),
            placeholder="Ex : C'est quoi les données DVF ?",
            max_chars=500,
        )
        st.session_state["ask_pdf_question"] = question

        run = st.button("Rechercher dans les PDFs", type="primary")
        if run:
            if not question.strip():
                st.warning("Veuillez saisir une question.")
            else:
                with st.spinner("Recherche en cours..."):
                    result = ask_pdf_api(question.strip())

                if not result:
                    st.error("Erreur inconnue lors de l'appel à l'API.")
                elif "error" in result:
                    st.error(f"**{result['error']}**")
                else:
                    if not result.get("pdf_rag_ready", True):
                        st.warning("RAG PDF pas prêt (index/faiss introuvable).")

                    answer = result.get("answer")
                    if answer:
                        st.markdown(answer)
                    else:
                        st.error("LLM indisponible ou réponse non fournie.")

        st.markdown("---")
        st.stop()

    if st.session_state["page"] == "summarize":
        st.markdown("### 📝 Résumer texte")
        st.markdown("Collez un texte (ou un extrait) à résumer.")
        st.markdown("---")

        longueur = st.radio(
            "Longueur du résumé (selon la taille du texte)",
            options=["Court", "Moyen", "Long"],
            index=1,
            horizontal=True,
        )
        # Mapping affichage → valeur API
        long_map = {"Court": "court", "Moyen": "moyen", "Long": "long"}
        long_val = long_map.get(longueur, "moyen")

        text = st.text_area(
            "Texte à résumer",
            value=st.session_state.get("summarize_text", ""),
            placeholder="Collez ici le texte à résumer...",
            height=220,
            max_chars=4000,
        )
        st.session_state["summarize_text"] = text

        run = st.button("Résumer", type="primary")
        if run:
            if not text.strip():
                st.warning("Veuillez saisir un texte.")
            else:
                with st.spinner("Résumé en cours..."):
                    result = summarize_api(text.strip(), long_val)

                if not result:
                    st.error("Erreur inconnue lors de l'appel à l'API.")
                elif "error" in result:
                    st.error(f"**{result['error']}**")
                else:
                    summary = result.get("summary", "")
                    st.markdown("---")
                    st.markdown("### Résumé")
                    st.write(summary)
                    st.caption(f"Longueur : {result.get('longueur', long_val)}")

        st.markdown("---")
        st.stop()

    if st.session_state["page"] == "summarize_pdf":
        st.markdown("### 📄 Résumer PDF")
        st.markdown("Chargez un PDF pour résumer son contenu.")
        st.markdown("---")

        longueur = st.radio(
            "Longueur du résumé (selon la taille du texte)",
            options=["Court", "Moyen", "Long"],
            index=1,
            horizontal=True,
        )
        long_map_pdf = {"Court": "court", "Moyen": "moyen", "Long": "long"}
        long_val_pdf = long_map_pdf.get(longueur, "moyen")

        uploaded = st.file_uploader("Choisir un fichier PDF", type=["pdf"])
        run = st.button("Résumer le PDF", type="primary", disabled=(uploaded is None))

        if run and uploaded is not None:
            with st.spinner("Résumé en cours..."):
                pdf_bytes = uploaded.read()
                pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
                result = summarize_pdf_api(pdf_b64, long_val_pdf)

            if not result:
                st.error("Erreur inconnue lors de l'appel à l'API.")
            elif "error" in result:
                st.error(f"**{result['error']}**")
            else:
                summary = result.get("summary", "")
                st.markdown("---")
                st.markdown("### Résumé")
                st.write(summary)
                st.caption(f"Longueur : {result.get('longueur', long_val_pdf)}")

        st.markdown("---")
        st.stop()

    # Page Questions numériques
    st.markdown("### 📊 Questions numériques")

    examples = [
        "Quel est le prix moyen au m² à Lyon ?",
        "Combien de ventes à Toulouse en 2023 ?",
        "Comment évoluent les prix à Bordeaux ?",
        "Quelle est la population de Nantes ?",
        "Donne-moi les données sur Paris",
    ]

    st.markdown("**💡 Exemples de questions :**")
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        with cols[i % 5]:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state["question"] = ex
                st.rerun()

    st.markdown("---")

    question = st.text_input(
        "Votre question",
        value=st.session_state.get("question", ""),
        placeholder="Ex : Quel est le prix moyen au m² à Marseille ?",
        max_chars=500,
    )
    st.session_state["question"] = question
    submit = st.button("Envoyer", type="primary")

    if "show_context_numeric" not in st.session_state:
        st.session_state["show_context_numeric"] = False

    if submit and question.strip():
        if len(question.strip()) < 3:
            st.warning("La question doit contenir au moins 3 caractères.")
        else:
            with st.spinner("Recherche en cours..."):
                start_ts = time.perf_counter()
                result = ask_api(question.strip())
                response_time = time.perf_counter() - start_ts
            if "error" in result:
                st.error(f"**{result['error']}**")
                if "detail" in result and "Chatbot non initialisé" in str(result.get("detail", "")):
                    st.info("L'API est lancée mais le chatbot n'a pas pu se charger. Vérifiez les logs uvicorn.")
                st.session_state["last_numeric_result"] = None
            else:
                st.session_state["last_numeric_result"] = result
                st.session_state["last_numeric_response_time"] = response_time
                st.session_state["show_context_numeric"] = False
    elif submit and not question.strip():
        st.warning("Veuillez saisir une question.")

    last_result = st.session_state.get("last_numeric_result")
    if last_result:
        answer = last_result.get("answer", "Pas de réponse.")
        sentence = last_result.get("sentence")
        context_used = last_result.get("context_used")
        score = last_result.get("score")
        source = last_result.get("source")
        response_time = st.session_state.get("last_numeric_response_time")

        safe_answer = html.escape(str(answer)).replace("\n", "<br>")
        safe_sentence = html.escape(str(sentence)).replace("\n", "<br>") if sentence else ""
        box_html = (
            f'<div style="background:#f0f7ff; border-left:4px solid #1f77b4; padding:1rem; margin:1rem 0; border-radius:0 8px 8px 0;">'
            f'<strong>Réponse brute :</strong><br><br>{safe_answer}'
            + (f"<br><br><strong>Réponse phrase :</strong><br>{safe_sentence}" if sentence else "")
            + f'<p style="font-size:0.75rem; color:#666; margin-top:0.5rem;">Source : {source or "N/A"}'
            + (f" • Score : {score:.2f}" if score is not None else "")
            + (f" • Temps : {response_time:.2f}s" if response_time is not None else "")
            + "</p></div>"
        )
        st.markdown(box_html, unsafe_allow_html=True)

        if context_used:
            label = "Masquer le contexte" if st.session_state["show_context_numeric"] else "Afficher le contexte"
            if st.button(label, key="toggle_context_numeric"):
                st.session_state["show_context_numeric"] = not st.session_state["show_context_numeric"]
                st.rerun()
            if st.session_state["show_context_numeric"]:
                st.markdown("**Contexte utilisé :**")
                st.write(context_used)
