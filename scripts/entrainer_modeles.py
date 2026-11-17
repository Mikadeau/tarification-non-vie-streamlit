"""Entraîne les modèles du TD et sauvegarde tout ce dont le dashboard a besoin dans models/.

Lancement (depuis la racine du projet) :  python -m scripts.entrainer_modeles
Durée : quelques minutes (sélection backward / forward sur 300 000 lignes).
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import (VAR_INTERACTIONS, VAR_TARIF_DDE, VAR_TARIF_VOL, base_cout, charger_base, decouper,
                      injecter_graves)
from src import graves as G
from src import modeles as M

SORTIE = Path(__file__).resolve().parents[1] / "models"
SORTIE.mkdir(exist_ok=True)
SEUIL_VOL = 20_000.0  # seuil d'écrêtement Vol (≈ quantile 99,5 %, rupture du mean excess plot)
CELLULES_MARCHE = ["Nbpieces", "Nbpers", "Qualite", "Local"]
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def sauver(df: pd.DataFrame, nom: str):
    df.to_parquet(SORTIE / f"{nom}.parquet", index=False)


bd = charger_base()
bda, bdt, samp = decouper(bd, seed=2016)
bdc = base_cout(bda, "DDE")
resume: dict = {"seed": 2016, "n_apprentissage": len(bda), "n_test": len(bdt)}
log("Base chargée")

# ---------------------------------------------------------------- Ex III
m0 = M.glm_freq(bda, [])
m1 = M.glm_freq(bda, ["Nbpieces"])
m2 = M.glm_freq(bda, ["Nbpieces", "Nbpers"])
m3 = M.glm_freq(bda, ["Nbpieces", "Nbpers"], interactions=[("Nbpieces", "Nbpers")])
complet = M.glm_freq(bda, VAR_TARIF_DDE)
crit3 = M.criteres({"M0 · nul": m0, "M1 · Nbpieces": m1, "M2 · + Nbpers": m2,
                    "M3 · + interaction": m3, "M4 · complet": complet}, bdt)
sauver(crit3, "ex3_criteres")
sauver(M.relativites(complet), "ex3_relativites_complet")
sauver(M.importance_wald(complet), "ex3_wald_complet")
resume["freq_nulle"] = float(np.exp(m0.params["Intercept"]))
resume["freq_observee_apprentissage"] = float(bda["Sini_DDE"].sum() / bda["Expo"].sum())
del m1, m2, m3
gc.collect()
log("Ex III terminé")

# Anomalie Kassu : variante regroupée
var_kreg = [v if v != "Kassu" else "Kassu_reg" for v in VAR_TARIF_DDE]
complet_kreg = M.glm_freq(bda, var_kreg)
sauver(M.criteres({"Kassu (8 modalités)": complet, "Kassu regroupé (6 modalités)": complet_kreg}, bdt), "ex3_kassu_criteres")
sauver(M.relativites(complet_kreg, ["Kassu_reg"]), "ex3_relativites_kassu_reg")
del complet, complet_kreg
gc.collect()
log("Variante Kassu terminée")

# ---------------------------------------------------------------- Ex IV
sauver(M.importance_retrait(bda, VAR_TARIF_DDE), "ex4_importance_retrait")
log("Importance par retrait terminée")
sauver(M.importance_interactions(bda, VAR_TARIF_DDE, VAR_INTERACTIONS), "ex4_interactions")
log("Interactions terminées")
sel_b, hist_b = M.selection_pas_a_pas(bda, VAR_TARIF_DDE, sens="backward")
sauver(hist_b, "ex4_backward")
log(f"Backward : {sel_b}")
sel_f, hist_f = M.selection_pas_a_pas(bda, VAR_TARIF_DDE, sens="forward")
sauver(hist_f, "ex4_forward")
log(f"Forward : {sel_f}")
resume["selection_backward"] = sel_b
resume["selection_forward"] = sel_f

# ---------------------------------------------------------------- Modèle final DDE
var_freq_dde = [v if v != "Kassu" else "Kassu_reg" for v in sel_b]
fit_f_dde = M.glm_freq(bda, var_freq_dde)
cout_complet = M.glm_cout(bdc, VAR_TARIF_DDE, "DDE")
wald_c = M.importance_wald(cout_complet)
sauver(wald_c, "ex5_wald_cout_complet")
var_cout_dde = [v for v in VAR_TARIF_DDE if v in set(wald_c.loc[wald_c["p-value"] < 0.05, "Variable"])]
fit_c_dde = M.glm_cout(bdc, var_cout_dde, "DDE")
sauver(M.relativites(fit_f_dde), "final_relativites_freq_dde")
sauver(M.relativites(fit_c_dde), "final_relativites_cout_dde")
cout_nul = M.glm_cout(bdc, [], "DDE")
resume["cm_modele_nul"] = float(np.exp(cout_nul.params["Intercept"]))
resume["cm_observe_apprentissage"] = float(bdc["Cout_DDE"].sum() / bdc["Sini_DDE"].sum())
resume["var_freq_dde"] = var_freq_dde
resume["var_cout_dde"] = var_cout_dde
log(f"Final DDE : fréquence {var_freq_dde} | coût {var_cout_dde}")

# Validation de la prime pure DDE
pp_a = M.prime_pure(fit_f_dde, fit_c_dde, bda)
pp_t = M.prime_pure(fit_f_dde, fit_c_dde, bdt)
resume["pp_dde"] = {"cout_observe_app": float(bda["Cout_DDE"].sum()), "cout_predit_app": float(pp_a.sum()),
                    "cout_observe_test": float(bdt["Cout_DDE"].sum()), "cout_predit_test": float(pp_t.sum()),
                    "pp_annuelle_moyenne_test": float(pp_t.sum() / bdt["Expo"].sum())}
sauver(M.table_lift(pp_t, bdt["Cout_DDE"], bdt["Expo"]), "ex5_lift_dde")
# Comparaison du pouvoir discriminant : modèle nul, modèle « assureur 2 », modèle final
fit_f2 = M.glm_freq(bda, ["Nbpieces", "Nbpers", "Qualite"])
fit_c2 = M.glm_cout(bdc, ["Nbpieces", "Nbpers", "Qualite"])
preds = {"Modèle nul": M.prime_pure(m0, cout_nul, bdt), "3 variables": M.prime_pure(fit_f2, fit_c2, bdt),
         "Modèle final": pp_t}
lor = []
resume["gini"] = {}
for nom, p in preds.items():
    c = M.courbe_lorenz(p, bdt["Cout_DDE"], bdt["Expo"]); c["Modèle"] = nom; lor.append(c)
    resume["gini"][nom] = M.gini(p, bdt["Cout_DDE"], bdt["Expo"])
sauver(pd.concat(lor), "ex5_lorenz_dde")
ovp = []
for v in ["Nbpieces", "Local", "Qualite", "Zonier_DDE", "Kassu_reg", "Age2"]:
    t = M.observe_vs_predit(bdt, pp_t, v); t = t.rename(columns={v: "Modalité"}); t["Variable"] = v
    t["Modalité"] = t["Modalité"].astype(str); ovp.append(t)
sauver(pd.concat(ovp), "ex5_obs_pred_dde")
log("Validation DDE terminée")

# ---------------------------------------------------------------- Modèle Vol (tarif)
fit_f_vol = M.glm_freq(bda, VAR_TARIF_VOL, "Vol")
bdc_vol_ecr = G.base_cout_ecretee(bda, SEUIL_VOL, "Vol")
fit_c_vol = M.glm_cout(bdc_vol_ecr, VAR_TARIF_VOL, "Vol", cible="CM_Vol_ecr")
k_vol = G.taux_surcharge(bd, SEUIL_VOL, "Vol")  # surcharge estimée sur tout l'historique
resume["vol"] = {"seuil": SEUIL_VOL, "taux_surcharge": k_vol}
sauver(M.relativites(fit_f_vol), "final_relativites_freq_vol")
sauver(M.relativites(fit_c_vol), "final_relativites_cout_vol")

# ---------------------------------------------------------------- Export du tarif (simulateur)
tarif = {
    "freq_DDE": M.exporter_modele(fit_f_dde, var_freq_dde, "Fréquence DDE (Poisson)", "DDE", "frequence"),
    "cout_DDE": M.exporter_modele(fit_c_dde, var_cout_dde, "Coût moyen DDE (Gamma)", "DDE", "cout"),
    "freq_Vol": M.exporter_modele(fit_f_vol, VAR_TARIF_VOL, "Fréquence Vol (Poisson)", "Vol", "frequence"),
    "cout_Vol": M.exporter_modele(fit_c_vol, VAR_TARIF_VOL, "Coût moyen Vol écrêté (Gamma)", "Vol", "cout",
                                  infos={"seuil_ecretement": SEUIL_VOL, "taux_surcharge_graves": k_vol}),
}
M.sauver_json(tarif, SORTIE / "tarif.json")
log("Tarif exporté")

# ---------------------------------------------------------------- Ex VI : marché concurrentiel
bdc_dde = bdc
assureurs = {"Assureur 1": [], "Assureur 2": ["Nbpieces", "Nbpers", "Qualite"],
             "Assureur 3": ["Nbpieces", "Nbpers", "Qualite", "Local"]}
cell = (bdt.groupby(CELLULES_MARCHE, observed=True)
           .agg(Exposition=("Expo", "sum"), Charge=("Cout_DDE", "sum"), Nb_sinistres=("Sini_DDE", "sum"),
                Nb_contrats=("Expo", "size")).reset_index())
proto = cell[CELLULES_MARCHE].copy(); proto["Expo"] = 1.0
for nom, vs in assureurs.items():
    ff = M.glm_freq(bda, vs); fc = M.glm_cout(bdc_dde, vs)
    cell[f"PP {nom}"] = (M.predire_freq(ff, proto) * fc.predict(proto)).to_numpy()  # PP annuelle de la cellule
    # Base Vol sur la base d'apprentissage pour la vérification
for c in CELLULES_MARCHE:
    cell[c] = cell[c].astype(str)
sauver(cell, "ex6_cellules_test")
log("Ex VI préparé")

# ---------------------------------------------------------------- Ex VII : sinistres graves (2 scénarios)
cells_vol = VAR_TARIF_VOL
ex7 = {}
for seed in (2021, 2016):
    bd2 = injecter_graves(bd)
    a2, t2, s2 = decouper(bd2, seed=seed)
    dans_app = [int(p) for p in (2344, 100788) if p in set(s2)]
    lignes_cout = a2.loc[a2["Cout_Vol"] > 0, cells_vol + ["Sini_Vol", "Cout_Vol", "Expo"]].copy()
    for c in cells_vol:
        lignes_cout[c] = lignes_cout[c].astype(str)
    sauver(lignes_cout, f"ex7_cout_app_seed{seed}")
    ff = M.glm_freq(a2, cells_vol, "Vol")
    ct = (t2.groupby(cells_vol, observed=True)
            .agg(Exposition=("Expo", "sum"), Charge=("Cout_Vol", "sum"), Nb_sinistres=("Sini_Vol", "sum"))
            .reset_index())
    pr = ct[cells_vol].copy(); pr["Expo"] = 1.0
    ct["Freq annuelle"] = M.predire_freq(ff, pr).to_numpy()
    for c in cells_vol:
        ct[c] = ct[c].astype(str)
    sauver(ct, f"ex7_cellules_test_seed{seed}")
    rel = M.relativites(ff); sauver(rel, f"ex7_relativites_freq_seed{seed}")
    ex7[str(seed)] = {"graves_dans_apprentissage": dans_app,
                      "charge_app": float(a2["Cout_Vol"].sum()), "charge_test": float(t2["Cout_Vol"].sum()),
                      "nb_sin_app": float(a2["Sini_Vol"].sum()), "expo_app": float(a2["Expo"].sum())}
    log(f"Ex VII seed {seed} : graves en apprentissage aux positions {dans_app}")
# coûts unitaires Vol de la base complète avec graves (pour le mean excess plot)
bd2 = injecter_graves(bd)
cu = (bd2.loc[bd2["Sini_Vol"] > 0, "Cout_Vol"] / bd2.loc[bd2["Sini_Vol"] > 0, "Sini_Vol"]).rename("Cout_unitaire")
sauver(cu.to_frame(), "ex7_couts_unitaires_vol")
resume["ex7"] = ex7

(SORTIE / "resume.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")
log("Terminé")
