"""Chargement et préparation de la base de tarification habitation (DDE et Vol)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

RACINE = Path(__file__).resolve().parents[1]
CHEMIN_BASE = RACINE / "data" / "base_tarif.rds"

# Ordre « métier » des modalités : la première modalité sert de référence dans les GLM.
ORDRE_MODALITES: dict[str, list[str]] = {
    "Couverture": ["C1", "C2", "C3"],
    "Nbpieces": ["1-2-3", "4-5", "6-8", ">=9"],
    "Nbpers": ["1", "2", "3", "4", ">=5"],
    "Qualite": ["Loc", "Prop"],
    "Kassu": ["12000e", "24000e", "36000e", "48000e", "64000e", "96000e", "128000e", "256000e"],
    "Age2": ["18-25", "26-35", "36-50", "51-65", ">=66"],
    "Usage": ["PRINC", "SECON", "PRO"],
    "Local": ["Maison", "Rdc", "Inter-1-2-3", "Inter-4+", "Dernier"],
    "Direct": ["N", "O"],
    "DMG": ["DMG1", "DMG2"],
    "Kassu_reg": ["12000e", "24000e", "36000e", "48-64000e", "96000e", ">=128000e"],
    "Zonier_DDE": ["Z1", "Z2", "Z3", "Z4"],
    "Ante_DDE": ["0", "1", ">=2"],
    "Zonier_Vol": ["Z1", "Z2", "Z3", "Z4", "Z5"],
    "Ante_Vol": ["0", "1", ">=2"],
}

LIBELLES: dict[str, str] = {
    "Couverture": "Formule de garantie",
    "Nbpieces": "Nombre de pièces",
    "Nbpers": "Nombre de personnes",
    "Qualite": "Qualité de l'occupant",
    "Kassu": "Capital assuré",
    "Kassu_reg": "Capital assuré (regroupé)",
    "Age2": "Classe d'âge",
    "Usage": "Usage du logement",
    "Local": "Type de local / étage",
    "Direct": "Souscription en direct",
    "DMG": "Indicateur DMG",
    "Zonier_DDE": "Zonier DDE",
    "Ante_DDE": "Antécédents DDE",
    "Zonier_Vol": "Zonier Vol",
    "Ante_Vol": "Antécédents Vol",
    "Exercice": "Exercice",
    "Dep": "Département",
}

# Ex III.4 : la relativité Kassu n'est pas monotone (48 000 > 64 000) et 256 000 est très peu exposé.
# Regroupement inspiré du cours (48 000 et 64 000 fusionnés), complété par 128 000 + 256 000.
REGROUPEMENT_KASSU = {"12000e": "12000e", "24000e": "24000e", "36000e": "36000e", "48000e": "48-64000e",
                      "64000e": "48-64000e", "96000e": "96000e", "128000e": ">=128000e", "256000e": ">=128000e"}

# Variables tarifaires utilisées dans le TD pour la garantie DDE (Ex III à V)
VAR_TARIF_DDE = ["Couverture", "Nbpieces", "Nbpers", "Qualite", "Kassu", "Age2",
                 "Usage", "Local", "Direct", "DMG", "Zonier_DDE", "Ante_DDE"]
# Variables imposées par l'énoncé pour le Vol (Ex VII)
VAR_TARIF_VOL = ["Couverture", "Qualite", "Zonier_Vol"]
# Variables autorisées à interagir (Ex IV.2)
VAR_INTERACTIONS = ["Nbpieces", "Nbpers", "Qualite", "Local", "Zonier_DDE", "Ante_DDE"]


def charger_base(chemin: str | Path = CHEMIN_BASE) -> pd.DataFrame:
    """Lit le .rds et type proprement les colonnes (catégories ordonnées, numériques)."""
    bd = pyreadr.read_r(str(chemin))[None]
    bd["Exercice"] = bd["Exercice"].astype(int)
    bd["Age_num"] = pd.to_numeric(bd["Age"], errors="coerce").astype("Int64")
    for var, ordre in ORDRE_MODALITES.items():
        if var in bd.columns:
            bd[var] = pd.Categorical(bd[var], categories=ordre)
    bd["Kassu_reg"] = pd.Categorical(bd["Kassu"].astype(str).map(REGROUPEMENT_KASSU),
                                     categories=ORDRE_MODALITES["Kassu_reg"])
    bd["Dep"] = bd["Dep"].astype(str).str.zfill(2)
    for col in ["Sini_DDE", "Sini_Vol", "Cout_DDE", "Cout_Vol", "Expo"]:
        bd[col] = bd[col].astype(float)
    return bd


def decouper(bd: pd.DataFrame, seed: int = 2016, n: int = 300_000):
    """Découpage apprentissage / test exactement comme l'énoncé du TD.

    np.random.seed(seed); samp = np.random.choice(len(bd), n, replace=False)
    (le notebook du cours utilise bd.sample, qui ne tire PAS les mêmes lignes).
    """
    np.random.seed(seed)
    samp = np.random.choice(len(bd), n, replace=False)
    bda = bd.iloc[samp].copy()
    bdt = bd.drop(bd.index[samp]).copy()
    return bda, bdt, samp


def injecter_graves(bd: pd.DataFrame) -> pd.DataFrame:
    """Ex VII : ajoute deux sinistres Vol extrêmes (2 M€ et 3 M€) aux positions de l'énoncé."""
    bd2 = bd.copy()
    idx = bd2.index[[2344, 100788]]
    bd2.loc[idx, "Sini_Vol"] = 1.0
    bd2.loc[idx, "Cout_Vol"] = [2e6, 3e6]
    return bd2


def base_cout(df: pd.DataFrame, garantie: str = "DDE") -> pd.DataFrame:
    """Lignes sinistrées, avec le coût moyen par sinistre de la ligne."""
    bdc = df.loc[df[f"Cout_{garantie}"] > 0].copy()
    bdc[f"CM_{garantie}"] = bdc[f"Cout_{garantie}"] / bdc[f"Sini_{garantie}"]
    return bdc
