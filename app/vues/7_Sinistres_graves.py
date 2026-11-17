"""Ex VII : sinistres extrêmes en Vol, écrêtement, mutualisation et effet en marché."""
import commun as C
import numpy as np
import pandas as pd
import streamlit as st

from src import graphiques as GR
from src import theme as T
from src.data import VAR_TARIF_VOL
from src.graves import mean_excess
from src.modeles import glm_cout

C.configurer("Sinistres graves")
C.bandeau("Sinistres graves", "Exercice VII · deux sinistres Vol de 2 M€ et 3 M€ injectés · écrêtement, mutualisation, "
                              "comparaison en marché")

res = C.lire_json("resume")
scen = st.radio("Scénario de découpage", ["2021", "2016"], horizontal=True,
                format_func=lambda s: "Seed 2021 (énoncé du TD)" if s == "2021" else "Seed 2016 (corrigé du professeur)")
info = res["ex7"][scen]
cu = C.lire("ex7_couts_unitaires_vol")["Cout_unitaire"]
lignes = C.lire(f"ex7_cout_app_seed{scen}")
cells = C.lire(f"ex7_cellules_test_seed{scen}")

pos = info["graves_dans_apprentissage"]
C.encadre(("Avec ce découpage, <b>aucun des deux sinistres graves n'est dans la base d'apprentissage</b> : ils tombent tous "
           "les deux dans la base test. Le modèle ne peut pas les « voir » ; ils matérialisent le risque que l'assureur "
           "doit pourtant financer, d'où la nécessité d'une charge grave mutualisée estimée à part." if not pos else
           f"Avec ce découpage, le sinistre grave en position {pos} est dans la base d'apprentissage : il déforme les "
           "coefficients du modèle de coût de sa cellule. L'autre est en base test."), "Où sont les graves ?")

t1, t2, t3, t4 = st.tabs(["🌋 Détection", "✂️ Écrêtement", "📐 Impact sur les modèles", "⚔️ Effet en marché"])
with t1:
    s, me = mean_excess(cu[cu < 1e6])
    c1, c2 = st.columns(2)
    with c1:
        C.graphe(GR.top_sinistres(cu, 15, 20_000, "Vol"))
    with c2:
        C.graphe(GR.mean_excess(s, me, 20_000))
    C.kpis([("Coût moyen Vol", T.fmt_eur(cu[cu < 1e6].mean()), "hors graves injectés"),
            ("Plus gros sinistre naturel", T.fmt_eur(cu[cu < 1e6].max()), ""),
            ("Sinistres injectés", "2 M€ et 3 M€", f"{T.fmt_pct(5e6 / cu[cu < 1e6].sum(), 0)} de la charge Vol historique")])
    C.encadre("Les deux sinistres injectés valent 25 à 40 fois le plus gros sinistre naturel et pèsent à eux seuls "
              "une part énorme de la charge Vol de huit exercices. <b>Il faut les retraiter</b> : un GLM Gamma estime une "
              "moyenne, un seul point extrême fait exploser le coefficient de sa cellule (ici une combinaison "
              "Couverture × Qualite × Zonier précise), si bien que ses assurés paieraient des primes absurdes et partiraient "
              "à la concurrence, tandis que le reste du portefeuille serait sous-tarifé. Sans retraitement, le tarif "
              "dépend du hasard de l'endroit où le sinistre est tombé, pas du risque.", "Question 1")

with t2:
    c1, c2 = st.columns([1, 2])
    seuil = c1.select_slider("Seuil d'écrêtement", options=[5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 50_000, 100_000],
                             value=20_000, format_func=T.fmt_eur)
    base_k = c1.radio("Surcharge grave estimée sur", ["Historique complet", "Base d'apprentissage"])
    n = lignes["Sini_Vol"]; cm = lignes["Cout_Vol"] / n
    lignes = lignes.assign(CM=cm, CM_ecr=np.minimum(cm, seuil))
    if base_k == "Base d'apprentissage":
        exces = ((lignes["CM"] - lignes["CM_ecr"]) * n).sum(); ecr = (lignes["CM_ecr"] * n).sum()
    else:
        exces = (np.maximum(cu - seuil, 0)).sum(); ecr = np.minimum(cu, seuil).sum()
    k = exces / ecr
    c1.markdown(f"<div class='prime-box'><div class='lbl'>Taux de surcharge</div><div class='val'>+{T.fmt_pct(k, 1)}</div>"
                f"<div class='sub'>charge au-delà de {T.fmt_eur(seuil)} / charge écrêtée</div></div>", unsafe_allow_html=True)
    with c2:
        C.graphe(GR.histogramme_couts(cu, {"Seuil": seuil}, garantie="Vol"))
    C.encadre("<b>Méthode</b> : (1) on choisit un seuil au début de la queue de distribution (rupture du mean excess plot, "
              "≈ quantile 99,5 %), (2) on écrête chaque sinistre au seuil et on modélise le coût moyen écrêté avec le GLM "
              "Gamma, (3) la charge au-delà du seuil est <b>mutualisée</b> : elle est répartie sur tous les assurés par un "
              "coefficient multiplicatif (1 + taux de surcharge). Comme les graves sont rares, la surcharge doit être "
              "estimée sur l'historique le plus long possible (voire avec une loi de Pareto généralisée), pas seulement "
              "sur l'échantillon d'apprentissage.", "Question 2 · méthodologie")


@st.cache_data(show_spinner=False)
def ajuster(scen: str, seuil: float):
    lg = C.lire(f"ex7_cout_app_seed{scen}").copy()
    lg["CM_brut"] = lg["Cout_Vol"] / lg["Sini_Vol"]
    lg["CM_ecr"] = np.minimum(lg["CM_brut"], seuil)
    propre = lg[lg["Cout_Vol"] < 1e6].copy()
    fits = {"Sans graves (référence)": glm_cout(propre, VAR_TARIF_VOL, "Vol", cible="CM_brut"),
            "Avec graves, non retraité": glm_cout(lg, VAR_TARIF_VOL, "Vol", cible="CM_brut"),
            "Avec graves, écrêté": glm_cout(lg, VAR_TARIF_VOL, "Vol", cible="CM_ecr")}
    return fits


fits = ajuster(scen, float(seuil))
coefs = pd.DataFrame({nom: np.exp(f.params) for nom, f in fits.items()})
coefs.index = coefs.index.str.replace("[T.", " = ", regex=False).str.replace("]", "", regex=False)
coefs = coefs.reset_index(names="Terme")

with t3:
    C.graphe(GR.comparaison_coefficients(coefs[coefs["Terme"] != "Intercept"], "Relativités du coût moyen Vol"))
    aff = coefs.copy()
    st.dataframe(aff.style.format({c: "{:.3f}" for c in aff.columns if c != "Terme"}, thousands=" ", decimal=","), width="stretch",
                 hide_index=True)
    pred = cells[VAR_TARIF_VOL].copy()
    obs = cells["Charge"].sum()
    lig = []
    for nom, f in fits.items():
        mult = (1 + k) if "écrêté" in nom else 1.0
        ch = (cells["Freq annuelle"] * f.predict(pred) * mult * cells["Exposition"]).sum()
        lig.append({"Modèle": nom, "Charge prédite test": ch, "Charge observée test": obs, "Ratio": ch / obs,
                    "AIC (coût)": f.aic})
    st.dataframe(pd.DataFrame(lig).style.format({"Charge prédite test": "{:,.0f} €", "Charge observée test": "{:,.0f} €",
                                                 "Ratio": "{:.1%}", "AIC (coût)": "{:,.0f}"}),
                 width="stretch", hide_index=True)
    C.encadre("Le modèle écrêté retrouve des relativités proches de la référence sans graves : la segmentation n'est plus "
              "polluée. Le modèle non retraité, lui, attribue au hasard des graves un effet tarifaire à une modalité "
              "(cas du seed 2016) et son ajustement se dégrade fortement. La mutualisation (1 + k) permet au modèle écrêté "
              "de financer la charge grave en moyenne.", "Qualité des modèles")

with t4:
    marge = st.slider("Marge commune aux deux assureurs", 0, 40, 15, 1, format="%d %%") / 100
    f_nr, f_ec = fits["Avec graves, non retraité"], fits["Avec graves, écrêté"]
    p_nr = cells["Freq annuelle"] * f_nr.predict(pred) * (1 + marge)
    p_ec = cells["Freq annuelle"] * f_ec.predict(pred) * (1 + k) * (1 + marge)
    choix_ec = p_ec <= p_nr
    lig = []
    for nom, m, p in [("Non retraité", ~choix_ec, p_nr), ("Écrêté + mutualisé", choix_ec, p_ec)]:
        primes = (p[m] * cells["Exposition"][m]).sum(); sin = cells["Charge"][m].sum()
        lig.append({"Assureur": nom, "Part de marché (expo)": cells["Exposition"][m].sum() / cells["Exposition"].sum(),
                    "Primes": primes, "Sinistres": sin, "Résultat": primes - sin})
    mk = pd.DataFrame(lig).set_index("Assureur")
    c1, c2 = st.columns(2)
    with c1:
        import plotly.graph_objects as go
        fig = go.Figure(go.Pie(labels=mk.index, values=mk["Part de marché (expo)"], hole=0.6,
                               marker=dict(colors=[T.ROUGE, T.OR], line=dict(color=T.NUIT, width=4)),
                               textinfo="label+percent"))
        fig.update_layout(title=T.titre("Parts de marché (exposition)"), height=420, showlegend=False)
        C.graphe(fig)
    with c2:
        fig = go.Figure(go.Bar(x=mk.index, y=mk["Résultat"], marker_color=[T.VERT if v >= 0 else T.ROUGE for v in mk["Résultat"]],
                               text=[T.fmt_eur(v) for v in mk["Résultat"]], textposition="outside"))
        fig.add_hline(y=0, line=dict(color=T.CHAMPAGNE))
        fig.update_layout(title=T.titre("Résultat technique en concurrence", "Base test, graves inclus"), height=420)
        C.graphe(fig)
    if mk.loc["Écrêté + mutualisé", "Part de marché (expo)"] < 0.01:
        C.encadre("Ici, les graves ne sont pas dans l'apprentissage : les deux modèles de coût sont presque identiques et seul "
                  "l'assureur prudent ajoute la surcharge grave. Il est plus cher partout et ne vend rien, mais celui qui prend "
                  "tout le marché <b>perd de l'argent</b> quand les graves surviennent : mieux vaut ne pas vendre que vendre à "
                  "perte. Choisis le scénario 2016 pour voir le cas où le grave déforme la segmentation.", "Lecture du scénario")
    C.encadre("En concurrence, l'assureur qui ne retraite pas surtarife la (ou les) cellule(s) touchée(s) par un grave et "
              "sous-tarifie les autres : il perd les premières et attire les secondes, avec une charge grave qu'il n'a "
              "pas financée. L'assureur qui écrête et mutualise propose un tarif cohérent avec le risque et finance les "
              "graves par la surcharge. <b>Conclusion</b> : les sinistres extrêmes se traitent hors du GLM (écrêtement, "
              "charge grave mutualisée, éventuellement réassurance en excédent de sinistre), jamais en les laissant "
              "piloter la segmentation.", "Interprétation et conclusion")
C.pied()
