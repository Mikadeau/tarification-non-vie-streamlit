"""Ex I et II.1 : le portefeuille, sa sinistralité par exercice et ses sinistres extrêmes."""
import commun as C
import numpy as np
import streamlit as st

from src import graphiques as GR
from src import theme as T
from src.indicateurs import couts_unitaires, part_expo_annuelle, table_par_exercice

C.configurer("Portefeuille")
C.bandeau("Le portefeuille", "Exercice I · volumes, exposition, fréquence et coût moyen · sinistralité extrême")

bd = C.base()
garantie = st.radio("Garantie", ["DDE", "Vol"], horizontal=True,
                    format_func=lambda g: "💧 Dégât des eaux" if g == "DDE" else "🔓 Vol")
tab = table_par_exercice(bd, garantie)
tot = tab.iloc[-1]
C.kpis([
    ("Contrats souscrits", T.fmt_nb(tot["Nb_contrats"]), "lignes de la base"),
    ("Contrats annuels", T.fmt_nb(tot["Exposition"]), "exposition en années"),
    (f"Sinistres {garantie}", T.fmt_nb(tot["Nb_sinistres"]), f"charge {T.fmt_eur(tot['Charge'])}"),
    ("Fréquence", T.fmt_pct(tot["Fréquence"]), "sinistres par année-police"),
    ("Coût moyen", T.fmt_eur(tot["Coût moyen"]), "par sinistre"),
    ("Prime pure", T.fmt_eur(tot["Prime pure"], 2), "charge / exposition"),
])

t1, t2, t3 = st.tabs(["📅 Par exercice", "🧭 Stabilité du portefeuille", "🌋 Sinistralité extrême"])
with t1:
    c1, c2 = st.columns(2)
    with c1:
        C.graphe(GR.kpi_exercice(tab, garantie))
    with c2:
        C.graphe(GR.cout_moyen_exercice(tab, garantie))
    aff = tab.rename(columns={"Nb_contrats": "Contrats", "Nb_sinistres": "Sinistres"})[
        ["Exercice", "Contrats", "Exposition", "Sinistres", "Charge", "Fréquence", "Coût moyen", "Prime pure"]]
    st.dataframe(aff.style.format({"Contrats": "{:,.0f}", "Exposition": "{:,.0f}", "Sinistres": "{:,.0f}",
                                   "Charge": "{:,.0f} €", "Fréquence": "{:.2%}", "Coût moyen": "{:,.0f} €",
                                   "Prime pure": "{:,.2f} €"}, thousands=" ", decimal=","), width="stretch", hide_index=True)
    C.encadre("Le portefeuille croît régulièrement (de 114 000 à 151 000 lignes) tandis que la fréquence et le coût moyen "
              "restent stables d'un exercice à l'autre : il n'y a pas de dérive temporelle nécessitant une variable "
              "« exercice » dans le tarif. Une ligne ne vaut pas un contrat annuel : 16 % des lignes ont plus d'un an "
              "d'exposition, d'où l'usage systématique de l'exposition au dénominateur.")
with t2:
    var = st.selectbox("Variable", ["Couverture", "Nbpieces", "Qualite", "Local", "Zonier_DDE", "Kassu"],
                       format_func=lambda v: GR.LIB(v))
    C.graphe(GR.stabilite_modalites(part_expo_annuelle(bd, var), var))
    C.encadre("La structure du portefeuille est stable : la part d'exposition de C1 (≈ 37 %), C2 (≈ 63 %) et C3 (≈ 0,3 %) "
              "varie de moins d'un demi-point par an. Les modèles estimés sur l'ensemble des exercices sont donc "
              "représentatifs du portefeuille actuel.")
with t3:
    cu = couts_unitaires(bd, garantie)
    q = cu.quantile([0.5, 0.9, 0.99, 0.999])
    C.kpis([("Médiane", T.fmt_eur(q[0.5]), "coût par sinistre"), ("Quantile 90 %", T.fmt_eur(q[0.9]), ""),
            ("Quantile 99 %", T.fmt_eur(q[0.99]), ""), ("Quantile 99,9 %", T.fmt_eur(q[0.999]), ""),
            ("Maximum", T.fmt_eur(cu.max()), f"{cu.max() / cu.mean():.0f} × le coût moyen")])
    q1, q3 = cu.quantile([0.25, 0.75])
    borne = q3 + 1.5 * (q3 - q1)
    c1, c2 = st.columns([3, 2])
    with c1:
        C.graphe(GR.histogramme_couts(cu, {"Q3 + 1,5 IQR": borne, "Q 99 %": q[0.99]}, garantie=garantie))
    with c2:
        C.graphe(GR.top_sinistres(cu, 15, q[0.99], garantie))
    C.graphe(GR.boite_couts(cu, garantie))
    part = cu[cu > q[0.99]].sum() / cu.sum()
    C.encadre(f"La distribution est très asymétrique à droite : la moitié des sinistres coûte moins de "
              f"<b>{T.fmt_eur(q[0.5])}</b>, mais les 1 % les plus graves (au-delà de {T.fmt_eur(q[0.99])}) "
              f"représentent <b>{T.fmt_pct(part, 1)}</b> de la charge. Ces sinistres restent dans un ordre de grandeur plausible "
              f"(maximum {T.fmt_eur(cu.max())}) : pas de retraitement nécessaire en DDE, contrairement au Vol de l'exercice VII.")
C.pied()
