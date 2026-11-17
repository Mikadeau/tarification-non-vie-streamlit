"""Page d'accueil du dashboard de tarification non vie."""
import commun as C
import streamlit as st

from src import graphiques as GR
from src import theme as T
from src.indicateurs import indicateurs_globaux, table_par_exercice

C.configurer("Accueil")
C.bandeau("Tarification non vie · Assurance habitation",
          "Modèles GLM fréquence × coût moyen sur 1,1 million de contrats · garanties Dégât des eaux et Vol · 2006-2013")

bd = C.base()
g_dde, g_vol = indicateurs_globaux(bd, "DDE"), indicateurs_globaux(bd, "Vol")
C.kpis([
    ("Contrats", T.fmt_nb(g_dde["Nb contrats"]), "lignes de la base"),
    ("Exposition", T.fmt_nb(g_dde["Exposition"]), "années-polices"),
    ("Sinistres DDE", T.fmt_nb(g_dde["Nb sinistres"]), f"fréquence {T.fmt_pct(g_dde['Fréquence'])}"),
    ("Coût moyen DDE", T.fmt_eur(g_dde["Coût moyen"]), f"charge {T.fmt_eur(g_dde['Charge'] / 1e6, 1)[:-2]} M€"),
    ("Prime pure DDE", T.fmt_eur(g_dde["Prime pure"], 2), "par an et par contrat"),
    ("Prime pure Vol", T.fmt_eur(g_vol["Prime pure"], 2), f"fréquence {T.fmt_pct(g_vol['Fréquence'])}"),
])

col1, col2 = st.columns([3, 2])
with col1:
    C.graphe(GR.kpi_exercice(table_par_exercice(bd, "DDE"), "DDE"))
with col2:
    st.markdown("### La démarche")
    st.markdown("""
<div class='insight'><span class='tag'>Prime pure = Fréquence × Coût moyen</span>
On modélise séparément <b>le nombre de sinistres</b> (GLM de Poisson, l'exposition en <i>offset</i>)
et <b>le coût moyen d'un sinistre</b> (GLM Gamma, lien log, pondéré par le nombre de sinistres).
Le produit des deux donne la <b>prime pure</b>, la prime commerciale ajoute la marge.</div>
""", unsafe_allow_html=True)
    st.markdown("""
| Page | Contenu |
|---|---|
| 📊 **Portefeuille** | Indicateurs par exercice, sinistres extrêmes |
| 🔎 **Analyses univariées** | Fréquence et CM par variable, âge, carte, V de Cramer, lois |
| 📈 **GLM Fréquence** | Comparaison de modèles, relativités, sélection de variables |
| 💶 **Coût et prime pure** | GLM Gamma, validation, lift, Lorenz |
| 🧮 **Simulateur** | Tarif d'un logement en temps réel |
| ⚔️ **Marché concurrentiel** | 3 assureurs, antisélection, marge optimale |
| 🌪️ **Sinistres graves** | Écrêtement et mutualisation |
""")

C.encadre("Le découpage apprentissage / test suit l'énoncé du TD : <b>np.random.seed(2016)</b> puis "
          "<b>np.random.choice(len(bd), 300 000)</b>. Les modèles sont estimés sur les 300 000 lignes d'apprentissage "
          "et validés sur les 812 095 lignes de test.", "Méthodologie")
C.pied()
