"""Ex III et IV : GLM de Poisson sur la fréquence DDE, importance des variables et sélection."""
import commun as C
import pandas as pd
import streamlit as st

from src import graphiques as GR
from src import theme as T

C.configurer("GLM Fréquence")
C.bandeau("GLM fréquence DDE", "Exercices III et IV · Poisson, lien log, offset log(exposition) · sélection de variables")

res = C.lire_json("resume")
crit = C.lire("ex3_criteres")
rel = C.lire("ex3_relativites_complet")

C.kpis([
    ("Fréquence observée", T.fmt_pct(res["freq_observee_apprentissage"], 3), "base d'apprentissage"),
    ("exp(intercept) modèle nul", T.fmt_pct(res["freq_nulle"], 3), "identique : le GLM nul = la moyenne"),
    ("Variables (backward)", str(len(res["selection_backward"])), "sur 12 candidates"),
    ("Variables (forward)", str(len(res["selection_forward"])), "même ensemble ?" if
     set(res["selection_backward"]) == set(res["selection_forward"]) else "ensemble différent"),
])

t1, t2, t3, t4, t5 = st.tabs(["⚖️ Comparaison des modèles", "🎚️ Relativités", "🏅 Importance des variables",
                              "🔀 Interactions", "🧭 Backward / Forward"])
with t1:
    C.graphe(GR.comparaison_modeles(crit, ("AIC", "BIC", "Déviance", "RMSE test")))
    st.dataframe(crit.style.format({"Déviance": "{:,.1f}", "AIC": "{:,.1f}", "BIC": "{:,.1f}", "RMSE test": "{:.5f}",
                                    "Déviance test": "{:,.1f}"}, thousands=" ", decimal=","), width="stretch", hide_index=True)
    C.encadre("Trois critères : l'<b>AIC</b> (−2 log L + 2k) arbitre ajustement et parcimonie, le <b>BIC</b> (−2 log L + k log n) "
              "pénalise davantage les paramètres, la <b>RMSE / déviance sur la base test</b> mesure la qualité prédictive hors "
              "échantillon. L'interaction Nbpieces × Nbpers fait à peine baisser l'AIC pour 12 paramètres de plus et le BIC la "
              "rejette : elle n'apporte rien. Le modèle complet domine nettement sur tous les critères.")
with t2:
    st.markdown("#### Modèle complet (12 variables)")
    vars_dispo = list(dict.fromkeys(rel["Variable"]))
    C.graphe(GR.relativites_grille(rel, vars_dispo, "Relativités du GLM Poisson complet"))
    c1, c2 = st.columns(2)
    with c1:
        bd = C.base()
        expo_k = bd.groupby("Kassu", observed=True)["Expo"].sum()
        expo_k.index = expo_k.index.astype(str)
        C.graphe(GR.courbe_kassu(rel, expo_k))
    with c2:
        st.dataframe(C.lire("ex3_kassu_criteres")[["Modèle", "Nb paramètres", "AIC", "BIC", "RMSE test"]]
                     .style.format({"AIC": "{:,.1f}", "BIC": "{:,.1f}", "RMSE test": "{:.5f}"}, thousands=" ", decimal=","),
                     width="stretch", hide_index=True)
        C.encadre("<b>Anomalies opérationnelles</b> : la relativité du capital assuré n'est pas monotone (48 000 € plus "
                  "risqué que 64 000 €) et 256 000 € a un intervalle énorme (1 088 lignes). Un tarif ne peut pas faire baisser "
                  "la prime quand le capital augmente : on regroupe 48-64 000 € et 128-256 000 € (variable "
                  "<i>Kassu_reg</i>). Autres signaux : <b>Direct</b> n'a aucun effet (p = 0,98), <b>C3</b> est fragile, "
                  "et <b>DMG2</b> réduit la fréquence de 23 % sans que son sens métier soit documenté.", "Anomalies")
with t3:
    imp = C.lire("ex4_importance_retrait")
    wald = C.lire("ex3_wald_complet")
    c1, c2 = st.columns(2)
    with c1:
        C.graphe(GR.importance_barres(imp, "Variable", "Δ AIC", "Importance par retrait : hausse de l'AIC",
                                      "On retire chaque variable du modèle complet · sous 0, la variable est inutile", seuil=0))
    with c2:
        C.graphe(GR.importance_barres(imp.assign(**{"Δ Déviance": imp["Δ Déviance"]}), "Variable", "Δ Déviance",
                                      "Importance par retrait : hausse de la déviance", "Perte d'ajustement si on retire la variable"))
    st.dataframe(imp.style.format({"Δ Déviance": "{:,.1f}", "Δ AIC": "{:,.1f}", "p-value": "{:.2e}"}, thousands=" ", decimal=","),
                 width="stretch", hide_index=True)
    top = ", ".join(imp["Variable"].head(4))
    inutiles = ", ".join(imp.loc[imp["Δ AIC"] < 0, "Variable"]) or "aucune"
    C.encadre(f"Les variables les plus importantes sont <b>{top}</b>. Retirer <b>{inutiles}</b> fait baisser l'AIC : "
              "ces variables n'améliorent pas le modèle et peuvent être écartées du tarif.")
with t4:
    inter = C.lire("ex4_interactions")
    C.graphe(GR.importance_barres(inter.assign(gain=-inter["Δ AIC"]), "Interaction", "gain",
                                  "Gain d'AIC apporté par chaque interaction",
                                  "Ajoutée une à une au modèle complet · au-delà de 0, l'interaction améliore l'AIC", seuil=0))
    st.dataframe(inter.style.format({"Δ Déviance": "{:,.1f}", "Δ AIC": "{:,.1f}", "p-value": "{:.2e}"}, thousands=" ", decimal=","),
                 width="stretch", hide_index=True)
    C.encadre("Les gains d'AIC des interactions sont faibles au regard de leur coût en paramètres et de la complexité "
              "opérationnelle (une grille croisée par couple). Le tarif final reste <b>multiplicatif sans interaction</b>, "
              "plus lisible pour le réseau commercial et plus stable.")
with t5:
    hb, hf = C.lire("ex4_backward"), C.lire("ex4_forward")
    C.graphe(GR.trajectoire_selection(hb, hf))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Backward** (on part du modèle complet)")
        st.dataframe(hb.style.format({"Critère": "{:,.1f}"}, thousands=" ", decimal=","), width="stretch", hide_index=True)
    with c2:
        st.markdown("**Forward** (on part du modèle nul)")
        st.dataframe(hf.style.format({"Critère": "{:,.1f}"}, thousands=" ", decimal=","), width="stretch", hide_index=True)
    meme = set(res["selection_backward"]) == set(res["selection_forward"])
    C.encadre(("Les deux méthodes aboutissent au <b>même ensemble de variables</b>. " if meme else
               "Les deux méthodes n'aboutissent pas au même ensemble. ") +
              f"Variables retenues : <b>{', '.join(res['selection_backward'])}</b>. Le backward part de toute l'information "
              "et évalue chaque variable en présence des autres (plus fiable quand les variables sont corrélées) ; le forward "
              "est plus rapide mais peut écarter une variable utile seulement en combinaison. Dans les deux cas, la sélection "
              "automatique ne remplace pas le jugement métier (stabilité, sens, faisabilité commerciale).")
C.pied()
