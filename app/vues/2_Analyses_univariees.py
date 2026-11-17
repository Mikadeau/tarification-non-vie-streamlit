"""Ex II : analyses univariées et bivariées, âge, département, corrélations, adéquation des lois."""
import commun as C
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from src import graphiques as GR
from src import theme as T
from src.data import VAR_TARIF_DDE
from src.indicateurs import couts_unitaires, matrice_cramer, table_indicateurs

C.configurer("Analyses")
C.bandeau("Analyses de la sinistralité", "Exercice II · univarié, bivarié, âge, géographie, corrélations et lois")

bd = C.base()
garantie = st.radio("Garantie", ["DDE", "Vol"], horizontal=True,
                    format_func=lambda g: "💧 Dégât des eaux" if g == "DDE" else "🔓 Vol")
VARS = ["Couverture", "Nbpieces", "Nbpers", "Qualite", "Kassu", "Age2", "Usage", "Local", "Direct", "DMG",
        f"Zonier_{garantie}", f"Ante_{garantie}"]


@st.cache_data(show_spinner=False)
def tab_ind(by, g):
    return table_indicateurs(C.base(), list(by), g)


@st.cache_data(show_spinner="Calcul des V de Cramer…")
def cramer():
    return matrice_cramer(C.base(), VAR_TARIF_DDE + ["Zonier_Vol", "Ante_Vol"])


@st.cache_data(show_spinner="Ajustement des lois…")
def lois(g):
    b = C.base()
    N = b[f"Sini_{g}"].astype(int).to_numpy(); e = b["Expo"].to_numpy()
    lam = N.sum() / e.sum(); k = np.arange(0, 7)
    p_obs = pd.Series(N).value_counts(normalize=True).reindex(k, fill_value=0).to_numpy()
    p_th = np.array([stats.poisson.pmf(j, lam * e).mean() for j in k])
    cu = couts_unitaires(b, g).to_numpy(); cu = cu[cu >= 1]
    m, v = cu.mean(), cu.var()
    gam = stats.gamma(m ** 2 / v, scale=v / m)
    s, _, sc = stats.lognorm.fit(cu, floc=0)
    ln = stats.lognorm(s, scale=sc)
    ks = {"Gamma": stats.kstest(cu, gam.cdf).statistic, "Log-normale": stats.kstest(cu, ln.cdf).statistic}
    return k, p_obs, p_th, lam, cu, {"Gamma": gam, "Log-normale": ln}, ks, N.var() / N.mean()


tabs = st.tabs(["📊 Univarié", "📈 Stabilité annuelle", "🫧 Bivarié", "👤 Âge", "🗺️ Département",
                "🔗 Corrélations", "📐 Adéquation des lois"])

with tabs[0]:
    var = st.selectbox("Variable tarifaire", VARS, index=0, format_func=GR.LIB, key="uni")
    t = tab_ind((var,), garantie)
    c1, c2 = st.columns(2)
    with c1:
        C.graphe(GR.univarie(t, var, garantie, "Fréquence"))
    with c2:
        C.graphe(GR.univarie(t, var, garantie, "Coût moyen"))
    st.dataframe(t[[var, "Exposition", "Part expo", "Nb_sinistres", "Fréquence", "Coût moyen", "Prime pure"]]
                 .style.format({"Exposition": "{:,.0f}", "Part expo": "{:.1%}", "Nb_sinistres": "{:,.0f}",
                                "Fréquence": "{:.2%}", "Coût moyen": "{:,.0f} €", "Prime pure": "{:,.2f} €"}, thousands=" ", decimal=","),
                 width="stretch", hide_index=True)
    if var == "Couverture":
        C.encadre("C1 et C2 ont un coût moyen proche (≈ 1 315 €) mais C2 a une fréquence plus élevée. <b>C3</b> est la plus "
                  "risquée en fréquence comme en coût, mais elle ne pèse que 0,3 % de l'exposition : son estimation est "
                  "fragile et sa relativité devra être lue avec son intervalle de confiance.")
    else:
        C.encadre("Comparer chaque losange à la ligne pointillée (moyenne du portefeuille) : l'écart mesure le pouvoir "
                  "discriminant brut de la variable. Une modalité très peu exposée (barre basse) donne une estimation "
                  "instable : c'est un candidat au regroupement.")

with tabs[1]:
    var = st.selectbox("Variable", VARS, index=0, format_func=GR.LIB, key="stab")
    t = tab_ind(("Exercice", var), garantie)
    c1, c2 = st.columns(2)
    with c1:
        C.graphe(GR.evolution_par_modalite(t, var, "Fréquence", garantie))
    with c2:
        C.graphe(GR.evolution_par_modalite(t, var, "Coût moyen", garantie))
    C.encadre("Les modalités bien exposées ont des fréquences et des coûts moyens stables (la hiérarchie entre modalités "
              "ne s'inverse pas), condition nécessaire pour qu'une relativité estimée sur le passé reste valable dans le tarif.")

with tabs[2]:
    c0, c00 = st.columns(2)
    vx = c0.selectbox("Variable en abscisse", VARS, index=1, format_func=GR.LIB)
    vy = c00.selectbox("Variable en ordonnée", VARS, index=2, format_func=GR.LIB)
    if vx == vy:
        st.warning("Choisis deux variables différentes.")
    else:
        t = tab_ind((vx, vy), garantie)
        c1, c2 = st.columns(2)
        with c1:
            C.graphe(GR.bulles_croisement(t, vx, vy, garantie))
        with c2:
            C.graphe(GR.heatmap_croisement(t, vx, vy, "Fréquence"))
        C.encadre("Pour Nbpieces × Nbpers, la fréquence augmente surtout avec le <b>nombre de pièces</b> (de ≈ 1,6 % à plus "
                  "de 3 % pour 9 pièces et plus), le nombre de personnes jouant à la marge. Si l'effet d'une variable est "
                  "le même dans chaque ligne, un modèle additif (sans interaction) suffit.")

with tabs[3]:
    t = tab_ind(("Age_num",), garantie)
    C.graphe(GR.sinistralite_age(t, garantie))
    C.encadre("La fréquence croît jusqu'à 35-50 ans puis se stabilise, le coût moyen augmente avec l'âge. Les courbes "
              "brutes sont bruitées aux âges extrêmes. <b>Contraintes opérationnelles d'une variable quantitative</b> : "
              "un GLM impose une forme (linéaire en log) qui ne colle pas à une courbe non monotone, chaque âge isolé est "
              "peu exposé, et le tarif doit rester lisible et stable d'une année sur l'autre (pas de saut de prime à "
              "chaque anniversaire). On regroupe donc en classes (Age2) ou on lisse (polynôme, splines).")

with tabs[4]:
    ind = st.radio("Indicateur", ["Fréquence", "Coût moyen", "Prime pure"], horizontal=True, key="dep")
    t = tab_ind(("Dep",), garantie)
    c1, c2 = st.columns([3, 2])
    with c1:
        C.graphe(GR.carte_departements(t, C.geojson(), ind, garantie))
    with c2:
        C.graphe(GR.univarie(tab_ind((f"Zonier_{garantie}",), garantie), f"Zonier_{garantie}", garantie, "Fréquence"))
    C.graphe(GR.classement_departements(t, ind))
    C.encadre("La sinistralité varie fortement selon le département (Paris et la petite couronne, la Côte d'Azur en tête). "
              "Avec 96 modalités, le département ne peut pas entrer tel quel dans un GLM (sur-paramétrage, départements "
              "peu exposés). <b>On construit un zonier</b> : on lisse la sinistralité géographique (résidus d'un GLM hors "
              "géographie, crédibilité, voisinage), puis on regroupe les départements en 4 ou 5 zones homogènes. "
              "C'est exactement le rôle des variables Zonier_DDE et Zonier_Vol.")

with tabs[5]:
    m = cramer()
    C.graphe(GR.heatmap_cramer(m))
    mm = m.where(~np.eye(len(m), dtype=bool)).stack().sort_values(ascending=False)
    a, b = mm.index[0]
    C.encadre(f"La liaison la plus forte est <b>{a} × {b}</b> (V = {mm.iloc[0]:.2f}) : les deux zoniers reposent sur la même "
              "géographie. Viennent ensuite DMG et les zoniers, puis Qualite × Local (les propriétaires habitent surtout des "
              "maisons). Toutes les liaisons restent modérées : pas de redondance qui rendrait les coefficients du GLM "
              "instables, mais l'effet propre de Qualite et de Local se lit « toutes choses égales par ailleurs ».")

with tabs[6]:
    k, p_obs, p_th, lam, cu, lois_aj, ks, disp = lois(garantie)
    c1, c2 = st.columns([2, 3])
    with c1:
        C.graphe(GR.adequation_poisson(k, p_obs, p_th, garantie))
    with c2:
        C.graphe(GR.adequation_couts(cu, lois_aj, garantie))
    C.kpis([("λ estimé", T.fmt_nb(lam, 4), "sinistres par année-police"), ("Variance / moyenne", T.fmt_nb(disp, 3),
            "≈ 1 sous Poisson"), ("KS Gamma", T.fmt_nb(ks['Gamma'], 3), "distance de Kolmogorov"),
            ("KS Log-normale", T.fmt_nb(ks['Log-normale'], 3), "plus petit = meilleur")])
    C.encadre("Le nombre de sinistres suit bien une <b>loi de Poisson</b> pondérée par l'exposition, avec une légère "
              "surdispersion (variance / moyenne ≈ 1,10) due à l'hétérogénéité entre assurés, que les variables du GLM captent. Pour les coûts, les deux lois sont rejetées par les tests "
              "(échantillon énorme), la log-normale épouse mieux la queue ; on garde pourtant le <b>GLM Gamma</b> pour le "
              "coût moyen : il modélise l'espérance avec une variance proportionnelle au carré de la moyenne, ce qui "
              "suffit pour tarifer, et reste robuste.")
C.pied()
