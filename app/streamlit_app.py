"""Point d'entrée du dashboard (fichier principal pour Streamlit Community Cloud)."""
from pathlib import Path

import streamlit as st

ICI = Path(__file__).resolve().parent
st.set_page_config(page_title="Tarification non vie · TEAM", page_icon=str(ICI.parent / "assets" / "logo_team.png"),
                   layout="wide", initial_sidebar_state="expanded")

pages = {
    "Vue d'ensemble": [
        st.Page(ICI / "vues" / "0_Accueil.py", url_path="accueil", title="Accueil", icon="🏛️", default=True),
        st.Page(ICI / "vues" / "1_Portefeuille.py", url_path="portefeuille", title="Portefeuille", icon="📊"),
        st.Page(ICI / "vues" / "2_Analyses_univariees.py", url_path="analyses", title="Analyses de sinistralité", icon="🔎"),
    ],
    "Modélisation": [
        st.Page(ICI / "vues" / "3_GLM_Frequence.py", url_path="glm_frequence", title="GLM fréquence", icon="📈"),
        st.Page(ICI / "vues" / "4_Cout_et_Prime_pure.py", url_path="cout_prime_pure", title="Coût et prime pure", icon="💶"),
        st.Page(ICI / "vues" / "5_Simulateur_de_prime.py", url_path="simulateur", title="Simulateur de prime", icon="🧮"),
    ],
    "Marché et risques": [
        st.Page(ICI / "vues" / "6_Marche_concurrentiel.py", url_path="marche", title="Marché concurrentiel", icon="⚔️"),
        st.Page(ICI / "vues" / "7_Sinistres_graves.py", url_path="sinistres_graves", title="Sinistres graves", icon="🌪️"),
    ],
}
st.navigation(pages).run()
