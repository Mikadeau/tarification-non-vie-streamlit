"""Indicateurs actuariels : exposition, fréquence, coût moyen, charge (prime pure observée)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def indicateurs_globaux(df: pd.DataFrame, garantie: str = "DDE") -> pd.Series:
    nb = df[f"Sini_{garantie}"].sum()
    cout = df[f"Cout_{garantie}"].sum()
    expo = df["Expo"].sum()
    return pd.Series({
        "Nb contrats": len(df),
        "Exposition": expo,
        "Nb sinistres": nb,
        "Charge": cout,
        "Fréquence": nb / expo,
        "Coût moyen": cout / nb if nb else np.nan,
        "Prime pure": cout / expo,
    })


def table_indicateurs(df: pd.DataFrame, by, garantie: str = "DDE") -> pd.DataFrame:
    """Tableau type du cours : Expo, Nb, Charge, Fréquence, CM, CG par modalité(s)."""
    by = [by] if isinstance(by, str) else list(by)
    t = (df.groupby(by, observed=True)
           .agg(Exposition=("Expo", "sum"),
                Nb_sinistres=(f"Sini_{garantie}", "sum"),
                Charge=(f"Cout_{garantie}", "sum"),
                Nb_contrats=("Expo", "size"))
           .reset_index())
    t["Fréquence"] = t["Nb_sinistres"] / t["Exposition"]
    t["Coût moyen"] = t["Charge"] / t["Nb_sinistres"].replace(0, np.nan)
    t["Prime pure"] = t["Charge"] / t["Exposition"]
    t["Part expo"] = t["Exposition"] / t["Exposition"].sum()
    return t


def table_par_exercice(df: pd.DataFrame, garantie: str = "DDE") -> pd.DataFrame:
    t = table_indicateurs(df, "Exercice", garantie)
    tot = indicateurs_globaux(df, garantie)
    tot_row = {"Exercice": "Total", "Exposition": tot["Exposition"], "Nb_sinistres": tot["Nb sinistres"],
               "Charge": tot["Charge"], "Nb_contrats": tot["Nb contrats"], "Fréquence": tot["Fréquence"],
               "Coût moyen": tot["Coût moyen"], "Prime pure": tot["Prime pure"], "Part expo": 1.0}
    t["Exercice"] = t["Exercice"].astype(str)
    return pd.concat([t, pd.DataFrame([tot_row])], ignore_index=True)


def part_expo_annuelle(df: pd.DataFrame, var: str) -> pd.DataFrame:
    """Stabilité temporelle : part d'exposition de chaque modalité par exercice."""
    t = df.groupby(["Exercice", var], observed=True)["Expo"].sum().reset_index()
    t["Part expo"] = t["Expo"] / t.groupby("Exercice")["Expo"].transform("sum")
    return t


def v_cramer(x: pd.Series, y: pd.Series) -> float:
    tab = pd.crosstab(x, y)
    chi2 = chi2_contingency(tab, correction=False)[0]
    return float(np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1))))


def matrice_cramer(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    m = pd.DataFrame(np.eye(len(variables)), index=variables, columns=variables)
    for i, a in enumerate(variables):
        for b in variables[i + 1:]:
            m.loc[a, b] = m.loc[b, a] = v_cramer(df[a], df[b])
    return m


def couts_unitaires(df: pd.DataFrame, garantie: str = "DDE") -> pd.Series:
    """Coût par sinistre (coût de la ligne / nombre de sinistres de la ligne)."""
    s = df[df[f"Sini_{garantie}"] > 0]
    return (s[f"Cout_{garantie}"] / s[f"Sini_{garantie}"]).rename("Coût unitaire")
