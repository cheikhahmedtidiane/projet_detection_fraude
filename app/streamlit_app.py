"""Démo Streamlit — détection de fraude Mobile Money.

Lancement, depuis la racine du projet :

    streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="Détection de fraude Mobile Money", page_icon="🛡️", layout="wide")

pages = [
    st.Page("vues/accueil.py", title="Accueil", icon="🏠", default=True),
    st.Page("vues/analyser.py", title="Analyser une transaction", icon="🔎"),
    st.Page("vues/simuler.py", title="Simuler une période", icon="📊"),
    st.Page("vues/a_propos.py", title="À propos du modèle", icon="ℹ️"),
]
st.navigation(pages).run()
