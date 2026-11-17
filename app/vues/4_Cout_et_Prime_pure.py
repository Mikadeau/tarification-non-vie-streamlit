"""Ex V : GLM Gamma sur le coût moyen DDE, prime pure et validation sur la base test."""
import commun as C
import pandas as pd
import streamlit as st

from src import graphiques as GR
from src import theme as T

C.configurer("Coût et prime pure")
C.bandeau("Coût moyen et prime pure DDE", "Exercice V · GLM Gamma lien log pondéré par le nombre de sinistres · "
                                          "prime pure = fréquence × coût moyen")

res = C.lire_json("resume")
pp = res["pp_dde"]
C.kpis([
    ("CM observé", T.fmt_eur(res["cm_observe_apprentissage"]), "apprentissage"),
    ("exp(intercept) Gamma nul", T.fmt_eur(res["cm_modele_nul"]), "retrouve le CM observé"),
    ("Charge prédite / observée", T.fmt_pct(pp['cout_predit_test'] / pp['cout_observe_test'], 2), "base test"),
    ("Prime pure annuelle moyenne", T.fmt_eur(pp["pp_annuelle_moyenne_test"], 2), "base test"),
    ("Gini modèle final", T.fmt_nb(res['gini']['Modèle final'], 3), f"vs {T.fmt_nb(res['gini']['3 variables'], 3)} avec 3 variables"),
])

t1, t2, t3 = st.tabs(["💶 Modèle de coût", "🧩 Modèle de fréquence final", "✅ Validation de la prime pure"])
with t1:
    wald = C.lire("ex5_wald_cout_complet")
    rel_c = C.lire("final_relativites_cout_dde")
    c1, c2 = st.columns([2, 3])
    with c1:
        C.graphe(GR.importance_barres(wald, "Variable", "Statistique", "Gamma complet : test de Wald par variable",
                                      "Statistique du test de nullité · or = significatif à 5 %",
                                      seuil=None))
    with c2:
        C.graphe(GR.relativites_grille(rel_c, res["var_cout_dde"], "Relativités du coût moyen DDE (modèle retenu)", cols=2))
    C.encadre(f"Le coût moyen est beaucoup moins segmentant que la fréquence : seules <b>{', '.join(res['var_cout_dde'])}</b> "
              "sont significatives. Les grands logements et les maisons ont des sinistres plus chers (plus de surface "
              "endommagée), les propriétaires aussi (ils déclarent aussi les dommages au bâti). Le lien log rend le modèle "
              "multiplicatif et compatible avec la fréquence ; le lien inverse (canonique) du cours donne des coefficients "
              "non interprétables en relativités.")
with t2:
    rel_f = C.lire("final_relativites_freq_dde")
    C.graphe(GR.relativites_grille(rel_f, res["var_freq_dde"], "Relativités de la fréquence DDE (modèle retenu)"))
    C.encadre("Le modèle de fréquence final reprend les variables retenues par la sélection backward, avec le capital assuré "
              "regroupé pour supprimer la non-monotonie. Lecture : un logement en zone Z4 a une fréquence multipliée par la "
              "relativité indiquée par rapport à la zone Z1, toutes choses égales par ailleurs.")
with t3:
    c1, c2 = st.columns(2)
    with c1:
        C.graphe(GR.lift(C.lire("ex5_lift_dde")))
    with c2:
        lor = C.lire("ex5_lorenz_dde")
        C.graphe(GR.lorenz({m: g for m, g in lor.groupby("Modèle", sort=False)}, res["gini"]))
    ovp = C.lire("ex5_obs_pred_dde")
    var = st.selectbox("Observé vs prédit par variable", list(dict.fromkeys(ovp["Variable"])), format_func=GR.LIB)
    C.graphe(GR.observe_predit(ovp[ovp["Variable"] == var].rename(columns={"Modalité": var}), var))
    tab = pd.DataFrame({"Base": ["Apprentissage", "Test"],
                        "Charge observée": [pp["cout_observe_app"], pp["cout_observe_test"]],
                        "Charge prédite": [pp["cout_predit_app"], pp["cout_predit_test"]]})
    tab["Écart"] = tab["Charge prédite"] / tab["Charge observée"] - 1
    st.dataframe(tab.style.format({"Charge observée": "{:,.0f} €", "Charge prédite": "{:,.0f} €", "Écart": "{:+.2%}"}, thousands=" ", decimal=","),
                 width="stretch", hide_index=True)
    C.encadre("Sur l'apprentissage, la charge prédite est quasi égale à la charge observée (−0,1 % : le GLM de Poisson à lien "
              "canonique équilibre exactement les sinistres, le lien log de la Gamma n'est pas canonique). Sur la base test, "
              "l'écart est de +0,01 % : le modèle généralise. La courbe de lift "
              "montre une prime observée croissante du décile 1 au décile 10 et suivie par la prédiction : le tarif hiérarchise "
              "correctement les risques. Le Gini mesure ce pouvoir de segmentation, nettement supérieur à celui d'un modèle "
              "à 3 variables : c'est ce qui fera la différence en marché concurrentiel.")
C.pied()
