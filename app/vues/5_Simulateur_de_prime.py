"""Simulateur : tarif d'un logement en temps réel à partir des modèles GLM retenus."""
import commun as C
import plotly.graph_objects as go
import streamlit as st

from src import graphiques as GR
from src import theme as T
from src.data import LIBELLES, ORDRE_MODALITES, REGROUPEMENT_KASSU
from src.modeles import appliquer_tarif

C.configurer("Simulateur")
C.bandeau("Simulateur de prime", "Décrivez le logement : fréquence, coût moyen et prime pure DDE + Vol calculés "
                                 "instantanément par les GLM retenus")

tarif = C.lire_json("tarif")
moyennes = C.lire_json("resume")
vars_utiles = []
for m in tarif.values():
    for v in m["variables"]:
        v = "Kassu" if v == "Kassu_reg" else v
        if v not in vars_utiles:
            vars_utiles.append(v)

AIDE = {"Kassu": "Capital mobilier assuré", "Local": "Maison, rez-de-chaussée, étage intermédiaire ou dernier étage",
        "Zonier_DDE": "Zone de risque dégât des eaux (Z1 la moins risquée)", "Zonier_Vol": "Zone de risque vol",
        "Ante_DDE": "Nombre de sinistres DDE antérieurs", "DMG": "Indicateur DMG de la base"}
FORMAT = {"Kassu": lambda m: f"{int(m[:-1]):,} €".replace(",", " ")}

st.markdown("### 🏠 Le logement")
profil = {}
cols = st.columns(4)
defauts = {"Nbpieces": "4-5", "Nbpers": "2", "Qualite": "Prop", "Kassu": "36000e", "Local": "Maison",
           "Zonier_DDE": "Z2", "Zonier_Vol": "Z2", "Couverture": "C2"}
for i, v in enumerate(vars_utiles):
    mods = ORDRE_MODALITES[v]
    profil[v] = cols[i % 4].selectbox(LIBELLES.get(v, v), mods, index=mods.index(defauts.get(v, mods[0])),
                                      format_func=FORMAT.get(v, str), help=AIDE.get(v))
profil["Kassu_reg"] = REGROUPEMENT_KASSU[profil.get("Kassu", "36000e")]

c_m, c_e = st.columns(2)
marge = c_m.slider("Marge commerciale", 0, 60, 15, 1, format="%d %%") / 100
expo = c_e.slider("Durée de couverture (années)", 0.25, 1.0, 1.0, 0.25)

f_dde, d_fdde = appliquer_tarif(tarif["freq_DDE"], profil)
cm_dde, d_cdde = appliquer_tarif(tarif["cout_DDE"], profil)
f_vol, d_fvol = appliquer_tarif(tarif["freq_Vol"], profil)
cm_vol_ecr, d_cvol = appliquer_tarif(tarif["cout_Vol"], profil)
k = tarif["cout_Vol"]["infos"]["taux_surcharge_graves"]
cm_vol = cm_vol_ecr * (1 + k)
pp_dde, pp_vol = f_dde * cm_dde, f_vol * cm_vol
pp = (pp_dde + pp_vol) * expo
prime_com = pp * (1 + marge)

st.markdown("### 💎 Le tarif")
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    st.markdown(f"<div class='prime-box'><div class='lbl'>Prime commerciale</div>"
                f"<div class='val'>{T.fmt_eur(prime_com, 2)}</div>"
                f"<div class='sub'>prime pure {T.fmt_eur(pp, 2)} × (1 + {T.fmt_pct(marge, 0)}) · {expo:g} an</div></div>",
                unsafe_allow_html=True)
with c2:
    C.kpis([("Fréquence DDE", T.fmt_pct(f_dde), "sinistres / an"), ("Coût moyen DDE", T.fmt_eur(cm_dde), "par sinistre"),
            ("Prime pure DDE", T.fmt_eur(pp_dde, 2), "par an")])
with c3:
    C.kpis([("Fréquence Vol", T.fmt_pct(f_vol), "sinistres / an"),
            ("Coût moyen Vol", T.fmt_eur(cm_vol), f"dont surcharge graves +{T.fmt_pct(k, 1)}"),
            ("Prime pure Vol", T.fmt_eur(pp_vol, 2), "par an")])

pp_moy = moyennes["pp_dde"]["pp_annuelle_moyenne_test"]
ecart = pp_dde / pp_moy - 1
C.encadre(f"La prime pure DDE de ce logement est <b>{T.fmt_pct(abs(ecart), 0)} {'au-dessus' if ecart > 0 else 'en dessous'}</b> "
          f"de la moyenne du portefeuille ({T.fmt_eur(pp_moy, 2)}). Le Vol est tarifé sur coûts écrêtés à "
          f"{T.fmt_eur(tarif['cout_Vol']['infos']['seuil_ecretement'])}, la charge des sinistres graves étant mutualisée "
          f"par une surcharge de {T.fmt_pct(k, 1)} appliquée à tous les assurés.", "Positionnement")

c1, c2 = st.columns(2)
with c1:
    det = [("Base", d_fdde[0][1] * d_cdde[0][1])] + [(n, r) for n, r in d_fdde[1:]] + [(f"{n} (coût)", r) for n, r in d_cdde[1:]]
    C.graphe(GR.waterfall_prime(det, "Construction de la prime pure DDE", "€"))
with c2:
    fig = go.Figure(go.Pie(labels=["Prime pure DDE", "Prime pure Vol", "Marge"],
                           values=[pp_dde * expo, pp_vol * expo, prime_com - pp], hole=0.6, sort=False,
                           marker=dict(colors=[T.BLEU_VIF, T.BRONZE, T.OR], line=dict(color=T.NUIT, width=4)),
                           textinfo="label+percent", hovertemplate="%{label}<br>%{value:,.2f} €<extra></extra>"))
    fig.add_annotation(text=f"<b>{T.fmt_eur(prime_com, 0)}</b><br>par contrat", showarrow=False,
                       font=dict(color=T.OR_CLAIR, size=18))
    fig.update_layout(title=T.titre("Décomposition de la prime commerciale"), height=460, showlegend=False)
    C.graphe(fig)

with st.expander("📋 Détail des relativités appliquées"):
    lignes = [(tarif[m]["nom"], LIBELLES.get(n, n), profil.get(n, "-") if n != "Base" else "-", r)
              for m, d in [("freq_DDE", d_fdde), ("cout_DDE", d_cdde), ("freq_Vol", d_fvol), ("cout_Vol", d_cvol)]
              for n, r in d]
    import pandas as pd
    st.dataframe(pd.DataFrame(lignes, columns=["Modèle", "Variable", "Modalité", "Coefficient"])
                 .style.format({"Coefficient": "{:.4f}"}, thousands=" ", decimal=","), width="stretch", hide_index=True)
C.pied()
