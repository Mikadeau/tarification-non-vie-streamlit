"""Ex VI : marché concurrentiel entre 3 assureurs, antisélection et marge optimale."""
import commun as C
import numpy as np
import pandas as pd
import streamlit as st

from src import graphiques as GR
from src import theme as T
from src.marche import balayage_cellules, repartition_cellules, simuler_cellules

C.configurer("Marché concurrentiel")
C.bandeau("Marché concurrentiel", "Exercice VI · 3 assureurs, prime commerciale = prime pure × (1 + marge) · "
                                  "chaque assuré de la base test choisit le moins cher")

cell = C.lire("ex6_cellules_test")
NOMS = ["Assureur 1", "Assureur 2", "Assureur 3"]

st.markdown("""
| | Assureur 1 | Assureur 2 | Assureur 3 |
|---|---|---|---|
| **Variables (fréquence et coût)** | aucune | Nbpieces, Nbpers, Qualite | Nbpieces, Nbpers, Qualite, **Local** |
""")
c1, c2, c3 = st.columns(3)
marges = {"Assureur 1": c1.slider("Marge assureur 1", 0, 60, 30, 1, format="%d %%") / 100,
          "Assureur 2": c2.slider("Marge assureur 2", 0, 60, 15, 1, format="%d %%") / 100,
          "Assureur 3": c3.slider("Marge assureur 3", 0, 60, 15, 1, format="%d %%") / 100}
sim = simuler_cellules(cell, marges)
s = sim["synthese"].set_index("Assureur")

C.kpis([(n, T.fmt_eur(s.loc[n, "Résultat"]), f"part de marché {T.fmt_pct(s.loc[n, 'Part de marché'], 1)} · S/P {T.fmt_pct(s.loc[n, 'S/P'], 0)}")
        for n in NOMS])

t0, t1, t2, t3 = st.tabs(["0️⃣ Seul sur le marché", "1️⃣ Marché simulé", "2️⃣ Antisélection", "4️⃣ Marge optimale"])
with t0:
    C.graphe(GR.resultats_assureurs(s["Résultat seul"].to_dict(), s["Résultat"].to_dict()))
    C.encadre("Seul sur le marché, chaque assureur encaisse toute la base test : son résultat vaut la marge appliquée à une "
              "prime pure bien calibrée en moyenne. L'assureur 1, avec sa marge de 30 %, serait le plus rentable. "
              "La segmentation n'a aucune valeur tant qu'il n'y a pas de concurrence.", "Question 0")
with t1:
    c1, c2 = st.columns([2, 3])
    with c1:
        C.graphe(GR.parts_de_marche(s["Part de marché"].to_dict()))
    with c2:
        C.graphe(GR.repartition_variable(repartition_cellules(cell, sim["choix"], "Local", NOMS), "Local"))
    c1, c2 = st.columns(2)
    with c1:
        C.graphe(GR.repartition_variable(repartition_cellules(cell, sim["choix"], "Nbpieces", NOMS), "Nbpieces"))
    with c2:
        C.graphe(GR.ratio_sp(s["S/P"].to_dict()))
    st.dataframe(sim["synthese"].style.format({"Marge": "{:.0%}", "Nb contrats": "{:,.0f}", "Part de marché": "{:.1%}",
                                               "Primes encaissées": "{:,.0f} €", "Sinistres payés": "{:,.0f} €",
                                               "Résultat": "{:,.0f} €", "S/P": "{:.1%}", "Résultat seul": "{:,.0f} €",
                                               "S/P seul": "{:.1%}"}, thousands=" ", decimal=","), width="stretch", hide_index=True)
with t2:
    st.markdown("#### Pourquoi l'antisélection apparaît")
    C.encadre("L'<b>antisélection</b> est la situation où l'assureur attire les risques qu'il tarife trop bas, parce que "
              "l'assuré (ou le marché) connaît mieux le risque que lui. Dans un marché concurrentiel, chaque assuré choisit "
              "le prix le plus bas : un assureur qui segmente moins finement que ses concurrents récupère mécaniquement les "
              "<b>mauvais risques sous-tarifés</b> et perd les bons risques, partis chez le concurrent qui les a vus "
              "moins chers. Son portefeuille se dégrade, il doit monter ses prix, perd encore ses meilleurs clients : "
              "c'est la spirale de l'antisélection.", "Question 2")
    bd = C.base()
    uni = bd.groupby("Local", observed=True).agg(Expo=("Expo", "sum"), Nb=("Sini_DDE", "sum"), Ch=("Cout_DDE", "sum"))
    uni["Prime pure"] = uni["Ch"] / uni["Expo"]
    uni = uni.reset_index()
    uni["Fréquence"] = uni["Nb"] / uni["Expo"]
    uni["Coût moyen"] = uni["Ch"] / uni["Nb"]
    uni["Part expo"] = uni["Expo"] / uni["Expo"].sum()
    uni["Nb_sinistres"] = uni["Nb"]; uni["Exposition"] = uni["Expo"]; uni["Charge"] = uni["Ch"]
    C.graphe(GR.univarie(uni, "Local", "DDE", "Prime pure"))
    rep = repartition_cellules(cell, sim["choix"], "Local", NOMS)
    m2 = rep.loc["Assureur 2", "Maison"] if "Maison" in rep.columns else np.nan
    m3 = rep.loc["Assureur 3", "Maison"] if "Maison" in rep.columns else np.nan
    C.encadre(f"<b>Lecture de la question 1 à la lumière de l'analyse univariée de Local</b> : les maisons ont la fréquence "
              f"la plus basse mais le coût moyen le plus élevé ; leur prime pure brute paraît plus forte, mais c'est un effet de "
              f"structure (maisons grandes et occupées par des propriétaires). À taille, composition et qualité égales, le GLM "
              f"de l'assureur 3 trouve une maison environ 20 % moins risquée qu'un appartement. L'assureur 2 ignore Local : il "
              f"<b>surtarife les maisons et sous-tarife les appartements</b>. L'assureur 3 capte donc les maisons "
              f"({T.fmt_pct(m3, 0)} de son portefeuille) et l'assureur 2 garde les appartements sous-tarifés "
              f"({T.fmt_pct(m2, 0)} de maisons seulement), d'où sa perte malgré une marge identique. L'assureur 1, sans "
              f"segmentation, ne gagne que les profils dont le risque dépasse son prix moyen majoré de 30 % : il subit le plus "
              f"l'antisélection.", "Question 3")
with t3:
    grille = np.round(np.arange(0.0, 0.601, 0.005), 3)
    res3, parts3 = balayage_cellules(cell, {**marges}, "Assureur 3", grille)
    j = int(res3.argmax())
    grille_pct = np.arange(0, 61) / 100
    res_pct, _ = balayage_cellules(cell, {**marges}, "Assureur 3", grille_pct)
    m_pct = grille_pct[int(res_pct.argmax())]
    C.graphe(GR.courbe_marge(grille, res3, parts3, (m_pct, res_pct.max())))
    C.kpis([("Marge optimale (au % près)", f"{T.fmt_pct(m_pct, 0)}", f"résultat {T.fmt_eur(res_pct.max())}"),
            ("Marge actuelle", T.fmt_pct(marges['Assureur 3'], 0), f"résultat {T.fmt_eur(s.loc['Assureur 3', 'Résultat'])}"),
            ("Gain potentiel", T.fmt_eur(res_pct.max() - s.loc["Assureur 3", "Résultat"]), "en ajustant la seule marge")])
    C.encadre("Le résultat de l'assureur 3 est une fonction en escalier de sa marge : à chaque hausse, il gagne en prime "
              "unitaire mais perd des cellules au profit des concurrents. L'optimum réalise le meilleur compromis "
              "<b>prix × volume</b>. Au-delà, l'assureur passe au-dessus du prix de l'assureur 2 sur des segments entiers "
              "et son résultat chute. Le calcul suppose des concurrents figés : en pratique, ils réagiraient.", "Question 4")
C.pied()
