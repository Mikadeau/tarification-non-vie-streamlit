"""GLM fréquence (Poisson) et coût moyen (Gamma), sélection de variables, prime pure, export du tarif."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

from .data import ORDRE_MODALITES, base_cout


# ----------------------------------------------------------------------------
# Ajustement
# ----------------------------------------------------------------------------
def formule(cible: str, variables: list[str], interactions: list[tuple[str, str]] | None = None) -> str:
    termes = list(variables) + [f"{a}:{b}" for a, b in (interactions or [])]
    return f"{cible} ~ " + (" + ".join(termes) if termes else "1")


def glm_freq(df: pd.DataFrame, variables: list[str], garantie: str = "DDE",
             interactions: list[tuple[str, str]] | None = None):
    """GLM Poisson, lien log, offset = log(Expo) (comme la correction du cours)."""
    return smf.glm(formule(f"Sini_{garantie}", variables, interactions), data=df,
                   family=sm.families.Poisson(sm.families.links.Log()),
                   offset=np.log(df["Expo"])).fit()


def glm_cout(df_cout: pd.DataFrame, variables: list[str], garantie: str = "DDE",
             lien: str = "log", cible: str | None = None):
    """GLM Gamma sur le coût moyen par sinistre, pondéré par le nombre de sinistres.

    On utilise var_weights partout (Ex V du corrigé) : la variance du coût moyen d'une ligne
    à n sinistres est divisée par n. (Le corrigé de l'Ex VI utilise freq_weights, ce qui
    revient à dupliquer les lignes et donne les mêmes coefficients mais des écarts-types faux.)
    """
    link = sm.families.links.Log() if lien == "log" else sm.families.links.InversePower()
    cible = cible or f"CM_{garantie}"
    return smf.glm(formule(cible, variables), data=df_cout,
                   family=sm.families.Gamma(link),
                   var_weights=df_cout[f"Sini_{garantie}"]).fit()


def _resume_fit(df, variables, garantie, interactions=None) -> dict:
    """Ajuste un GLM fréquence, garde seulement les indicateurs utiles et libère la mémoire
    (indispensable dans les boucles de sélection : chaque ajustement pèse plusieurs centaines de Mo)."""
    f = glm_freq(df, variables, garantie, interactions)
    r = {"deviance": f.deviance, "aic": f.aic, "bic": f.bic_llf, "df_resid": f.df_resid}
    del f
    gc.collect()
    return r


def predire_freq(fit, df: pd.DataFrame) -> pd.Series:
    """Nombre de sinistres attendu sur la période d'exposition de la ligne."""
    return fit.predict(df, offset=np.log(df["Expo"]))


# ----------------------------------------------------------------------------
# Critères de qualité
# ----------------------------------------------------------------------------
def rmse_freq(fit, test: pd.DataFrame, garantie: str = "DDE") -> float:
    """RMSE pondérée par l'exposition sur la base test (définition du cours)."""
    mu = predire_freq(fit, test)
    return float(np.sqrt(np.sum((mu - test[f"Sini_{garantie}"]) ** 2 * test["Expo"]) / len(test)))


def deviance_poisson_test(fit, test: pd.DataFrame, garantie: str = "DDE") -> float:
    y = test[f"Sini_{garantie}"].to_numpy()
    mu = predire_freq(fit, test).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        terme = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2 * np.sum(terme - (y - mu)))


def criteres(fits: dict, test: pd.DataFrame | None = None, garantie: str = "DDE") -> pd.DataFrame:
    lignes = []
    for nom, f in fits.items():
        d = {"Modèle": nom, "Nb paramètres": int(f.df_model + 1), "Déviance": f.deviance,
             "AIC": f.aic, "BIC": f.bic_llf}
        if test is not None:
            d["RMSE test"] = rmse_freq(f, test, garantie)
            d["Déviance test"] = deviance_poisson_test(f, test, garantie)
        lignes.append(d)
    return pd.DataFrame(lignes)


def test_rapport_vraisemblance(petit, grand) -> dict:
    stat = petit.deviance - grand.deviance
    ddl = int(petit.df_resid - grand.df_resid)
    return {"Statistique": stat, "ddl": ddl, "p-value": float(chi2.sf(stat, ddl))}


# ----------------------------------------------------------------------------
# Relativités
# ----------------------------------------------------------------------------
def relativites(fit, variables: list[str] | None = None) -> pd.DataFrame:
    """exp(coef) avec IC à 95 %, y compris la modalité de référence (relativité 1)."""
    params, ic, pv = fit.params, fit.conf_int(), fit.pvalues
    if variables is None:
        variables = [v for v in ORDRE_MODALITES if any(k.startswith(f"{v}[") for k in params.index)]
    lignes = []
    for var in variables:
        for i, mod in enumerate(ORDRE_MODALITES[var]):
            cle = f"{var}[T.{mod}]"
            if i == 0:
                lignes.append({"Variable": var, "Modalité": mod, "Relativité": 1.0,
                               "IC bas": 1.0, "IC haut": 1.0, "p-value": np.nan, "Référence": True})
            elif cle in params.index:
                lignes.append({"Variable": var, "Modalité": mod, "Relativité": np.exp(params[cle]),
                               "IC bas": np.exp(ic.loc[cle, 0]), "IC haut": np.exp(ic.loc[cle, 1]),
                               "p-value": pv[cle], "Référence": False})
    return pd.DataFrame(lignes)


def importance_wald(fit) -> pd.DataFrame:
    w = fit.wald_test_terms(scalar=True).table.drop("Intercept", errors="ignore")
    w.columns = ["Statistique", "p-value", "ddl"]
    return w.sort_values("Statistique", ascending=False).reset_index(names="Variable")


# ----------------------------------------------------------------------------
# Ex IV : importance par retrait, interactions, backward / forward
# ----------------------------------------------------------------------------
def importance_retrait(df, variables, garantie="DDE") -> pd.DataFrame:
    """Retire les variables une à une du modèle complet et mesure la perte (déviance, AIC)."""
    complet = _resume_fit(df, variables, garantie)
    lignes = []
    for v in variables:
        f = _resume_fit(df, [x for x in variables if x != v], garantie)
        stat, ddl = f["deviance"] - complet["deviance"], int(f["df_resid"] - complet["df_resid"])
        lignes.append({"Variable": v, "Δ Déviance": stat, "Δ AIC": f["aic"] - complet["aic"], "ddl": ddl,
                       "p-value": float(chi2.sf(stat, ddl))})
    return pd.DataFrame(lignes).sort_values("Δ AIC", ascending=False).reset_index(drop=True)


def importance_interactions(df, variables, var_inter, garantie="DDE") -> pd.DataFrame:
    from itertools import combinations
    base = _resume_fit(df, variables, garantie)
    lignes = []
    for a, b in combinations(var_inter, 2):
        f = _resume_fit(df, variables, garantie, interactions=[(a, b)])
        stat, ddl = base["deviance"] - f["deviance"], int(base["df_resid"] - f["df_resid"])
        lignes.append({"Interaction": f"{a} × {b}", "Δ Déviance": stat, "Δ AIC": f["aic"] - base["aic"],
                       "ddl": ddl, "p-value": float(chi2.sf(stat, ddl)) if ddl > 0 else np.nan})
    return pd.DataFrame(lignes).sort_values("Δ AIC").reset_index(drop=True)


def selection_pas_a_pas(df, variables, garantie="DDE", sens="backward", critere="aic"):
    """Sélection pas à pas sur l'AIC (ou le BIC). Retourne (variables retenues, historique)."""
    crit = (lambda f: f["aic"]) if critere == "aic" else (lambda f: f["bic"])
    courant = list(variables) if sens == "backward" else []
    f_courant = _resume_fit(df, courant, garantie)
    historique = [{"Étape": 0, "Action": "Départ (modèle complet)" if sens == "backward" else "Départ (modèle nul)",
                   "Variable": "-", "Critère": crit(f_courant), "Nb variables": len(courant)}]
    etape = 0
    while True:
        candidats = []
        options = courant if sens == "backward" else [v for v in variables if v not in courant]
        for v in options:
            vars_test = [x for x in courant if x != v] if sens == "backward" else courant + [v]
            candidats.append((crit(_resume_fit(df, vars_test, garantie)), v, vars_test))
        if not candidats:
            break
        meilleur = min(candidats, key=lambda c: c[0])
        if meilleur[0] >= crit(f_courant) - 1e-9:
            break
        etape += 1
        courant = meilleur[2]
        f_courant = {"aic": meilleur[0], "bic": meilleur[0]}
        historique.append({"Étape": etape, "Action": "Retrait" if sens == "backward" else "Ajout",
                           "Variable": meilleur[1], "Critère": meilleur[0], "Nb variables": len(courant)})
    return courant, pd.DataFrame(historique)


# ----------------------------------------------------------------------------
# Prime pure et validation
# ----------------------------------------------------------------------------
def prime_pure(fit_f, fit_c, df: pd.DataFrame) -> pd.Series:
    """Prime pure sur la période d'exposition = fréquence × exposition × coût moyen."""
    return predire_freq(fit_f, df) * fit_c.predict(df)


def table_lift(pred: pd.Series, obs: pd.Series, expo: pd.Series, n: int = 10) -> pd.DataFrame:
    """Courbe de lift : on classe les contrats par prime pure annuelle prédite, en déciles d'exposition."""
    d = pd.DataFrame({"pred": pred.to_numpy(), "obs": obs.to_numpy(), "expo": expo.to_numpy()})
    d["pp_an"] = d["pred"] / d["expo"]
    d = d.sort_values("pp_an")
    d["cum"] = d["expo"].cumsum() / d["expo"].sum()
    d["Décile"] = np.minimum((d["cum"] * n).apply(np.ceil).astype(int), n)
    t = d.groupby("Décile").agg(Exposition=("expo", "sum"), Prédit=("pred", "sum"), Observé=("obs", "sum"))
    t["PP prédite"] = t["Prédit"] / t["Exposition"]
    t["PP observée"] = t["Observé"] / t["Exposition"]
    return t.reset_index()


def gini(pred: pd.Series, obs: pd.Series, expo: pd.Series) -> float:
    """Indice de Gini (courbe de Lorenz ordonnée par prime pure annuelle prédite)."""
    d = pd.DataFrame({"p": (pred / expo).to_numpy(), "o": obs.to_numpy(), "e": expo.to_numpy()}).sort_values("p")
    x = np.concatenate([[0], d["e"].cumsum() / d["e"].sum()])
    y = np.concatenate([[0], d["o"].cumsum() / d["o"].sum()])
    return float(1 - 2 * np.trapezoid(y, x))


def courbe_lorenz(pred, obs, expo, n_points: int = 200) -> pd.DataFrame:
    d = pd.DataFrame({"p": (pred / expo).to_numpy(), "o": obs.to_numpy(), "e": expo.to_numpy()}).sort_values("p")
    x = d["e"].cumsum() / d["e"].sum()
    y = d["o"].cumsum() / d["o"].sum()
    idx = np.linspace(0, len(d) - 1, n_points).astype(int)
    return pd.DataFrame({"Part exposition": np.r_[0, x.iloc[idx]], "Part charge": np.r_[0, y.iloc[idx]]})


def observe_vs_predit(df: pd.DataFrame, pred: pd.Series, var: str, garantie: str = "DDE") -> pd.DataFrame:
    d = df[[var, "Expo", f"Cout_{garantie}"]].copy()
    d["pred"] = pred.to_numpy()
    t = d.groupby(var, observed=True).agg(Exposition=("Expo", "sum"),
                                          Observé=(f"Cout_{garantie}", "sum"), Prédit=("pred", "sum"))
    t["PP observée"] = t["Observé"] / t["Exposition"]
    t["PP prédite"] = t["Prédit"] / t["Exposition"]
    return t.reset_index()


# ----------------------------------------------------------------------------
# Export / chargement du tarif (utilisé par le simulateur Streamlit)
# ----------------------------------------------------------------------------
def exporter_modele(fit, variables: list[str], nom: str, garantie: str, type_modele: str,
                    infos: dict | None = None) -> dict:
    rel = relativites(fit, variables)
    return {
        "nom": nom, "garantie": garantie, "type": type_modele,
        "intercept": float(np.exp(fit.params["Intercept"])),
        "variables": {v: {r["Modalité"]: float(r["Relativité"]) for _, r in rel[rel.Variable == v].iterrows()}
                      for v in variables},
        "infos": infos or {},
    }


def sauver_json(obj, chemin: str | Path) -> None:
    Path(chemin).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def appliquer_tarif(modele: dict, profil: dict) -> tuple[float, list[tuple[str, float]]]:
    """Valeur annuelle (fréquence ou coût moyen) et décomposition multiplicative."""
    valeur = modele["intercept"]
    detail = [("Base", modele["intercept"])]
    for var, rels in modele["variables"].items():
        r = rels.get(str(profil[var]), 1.0)
        valeur *= r
        detail.append((var, r))
    return valeur, detail
