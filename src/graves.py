"""Ex VII : sinistres extrêmes, écrêtement et mutualisation de la charge grave."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import base_cout


def mean_excess(couts: pd.Series, seuils: np.ndarray | None = None, min_obs: int = 15):
    """Fonction d'excès moyen e(u) = E[X - u | X > u]."""
    x = np.sort(np.asarray(couts, dtype=float))
    if seuils is None:
        seuils = np.quantile(x, np.linspace(0.5, 0.999, 150))
    me, s_ok = [], []
    for u in seuils:
        exc = x[x > u] - u
        if len(exc) >= min_obs:
            me.append(exc.mean()); s_ok.append(u)
    return np.array(s_ok), np.array(me)


def ecreter(df: pd.DataFrame, seuil: float, garantie: str = "Vol") -> pd.DataFrame:
    """Écrête le coût de chaque sinistre au seuil (coût moyen de la ligne plafonné à `seuil`).

    Ajoute : CM_<g>_ecr (coût moyen écrêté), Cout_<g>_ecr (charge écrêtée), Exces_<g> (sur-crête).
    """
    d = df.copy()
    n = d[f"Sini_{garantie}"]
    cm = np.where(n > 0, d[f"Cout_{garantie}"] / n.replace(0, np.nan), 0.0)
    cm_ecr = np.minimum(cm, seuil)
    d[f"Cout_{garantie}_ecr"] = cm_ecr * n
    d[f"Exces_{garantie}"] = d[f"Cout_{garantie}"] - d[f"Cout_{garantie}_ecr"]
    return d


def base_cout_ecretee(df: pd.DataFrame, seuil: float, garantie: str = "Vol") -> pd.DataFrame:
    bdc = base_cout(ecreter(df, seuil, garantie), garantie)
    bdc[f"CM_{garantie}_ecr"] = bdc[f"Cout_{garantie}_ecr"] / bdc[f"Sini_{garantie}"]
    return bdc


def taux_surcharge(df: pd.DataFrame, seuil: float, garantie: str = "Vol") -> float:
    """Charge au-delà du seuil rapportée à la charge écrêtée : coefficient de mutualisation."""
    d = ecreter(df, seuil, garantie)
    return float(d[f"Exces_{garantie}"].sum() / d[f"Cout_{garantie}_ecr"].sum())


def synthese_seuils(df: pd.DataFrame, seuils, garantie: str = "Vol") -> pd.DataFrame:
    lignes = []
    cu = df.loc[df[f"Sini_{garantie}"] > 0, f"Cout_{garantie}"] / df.loc[df[f"Sini_{garantie}"] > 0, f"Sini_{garantie}"]
    for s in seuils:
        d = ecreter(df, s, garantie)
        lignes.append({"Seuil": s, "Nb sinistres écrêtés": int((cu > s).sum()),
                       "Part écrêtée de la charge": d[f"Exces_{garantie}"].sum() / d[f"Cout_{garantie}"].sum(),
                       "Taux de surcharge": taux_surcharge(df, s, garantie)})
    return pd.DataFrame(lignes)
