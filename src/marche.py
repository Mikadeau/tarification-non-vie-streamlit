"""Ex VI : marché concurrentiel entre assureurs (chaque assuré choisit le moins cher)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def simuler_marche(primes_pures: dict[str, np.ndarray], marges: dict[str, float], cout: np.ndarray,
                   variables: pd.DataFrame | None = None) -> dict:
    """Simule le marché sur la base test.

    primes_pures : prime pure (sur la période d'exposition) de chaque assureur pour chaque contrat
    marges       : marge commerciale de chaque assureur (prime commerciale = PP × (1 + marge))
    cout         : sinistres réellement payés par contrat
    variables    : colonnes à décrire dans le portefeuille de chaque assureur (ex. Local)
    En cas d'égalité parfaite de prix, l'assuré choisit le premier assureur de la liste.
    """
    noms = list(primes_pures)
    P = np.column_stack([np.asarray(primes_pures[n]) * (1 + marges[n]) for n in noms])
    choix = P.argmin(axis=1)
    cout = np.asarray(cout)
    res = {"noms": noms, "choix": choix, "primes_commerciales": P}
    lignes = []
    for j, n in enumerate(noms):
        m = choix == j
        primes, sinistres = P[m, j].sum(), cout[m].sum()
        lignes.append({
            "Assureur": n, "Marge": marges[n], "Nb contrats": int(m.sum()), "Part de marché": m.mean(),
            "Primes encaissées": primes, "Sinistres payés": sinistres, "Résultat": primes - sinistres,
            "S/P": sinistres / primes if primes else np.nan,
            "Prime moyenne": P[m, j].mean() if m.any() else np.nan,
            "Résultat seul": P[:, j].sum() - cout.sum(),
            "S/P seul": cout.sum() / P[:, j].sum(),
        })
    res["synthese"] = pd.DataFrame(lignes)
    if variables is not None:
        res["repartitions"] = {}
        for var in variables.columns:
            rep = pd.crosstab(pd.Series(np.array(noms)[choix], name="Assureur"),
                              variables[var].astype(str).to_numpy(), normalize="index")
            res["repartitions"][var] = rep.reindex(noms)
    return res


def resultat_assureur(j: int, primes_pures: dict[str, np.ndarray], marges: dict[str, float],
                      cout: np.ndarray) -> tuple[float, float]:
    noms = list(primes_pures)
    P = np.column_stack([np.asarray(primes_pures[n]) * (1 + marges[n]) for n in noms])
    m = P.argmin(axis=1) == j
    return float(P[m, j].sum() - np.asarray(cout)[m].sum()), float(m.mean())


def balayage_marge(j: int, primes_pures: dict, marges: dict, cout: np.ndarray, grille: np.ndarray):
    """Résultat et part de marché de l'assureur j pour chaque marge de la grille."""
    noms = list(primes_pures)
    res, parts = [], []
    for g in grille:
        mg = dict(marges); mg[noms[j]] = g
        r, p = resultat_assureur(j, primes_pures, mg, cout)
        res.append(r); parts.append(p)
    return np.array(res), np.array(parts)


def marge_optimale(j: int, primes_pures: dict, marges: dict, cout: np.ndarray,
                   bornes=(0.0, 0.6), pas_grossier=0.01, pas_fin=0.0005):
    """Recherche en deux temps (grille au % puis affinage) : la fonction résultat est en escalier,
    un optimiseur continu (minimize_scalar du corrigé) peut s'arrêter sur un optimum local."""
    g1 = np.arange(bornes[0], bornes[1] + 1e-12, pas_grossier)
    r1, _ = balayage_marge(j, primes_pures, marges, cout, g1)
    m0 = g1[r1.argmax()]
    g2 = np.arange(max(bornes[0], m0 - pas_grossier), m0 + pas_grossier + 1e-12, pas_fin)
    r2, _ = balayage_marge(j, primes_pures, marges, cout, g2)
    return {"grille": g1, "resultats": r1, "marge_au_pct": float(m0), "resultat_au_pct": float(r1.max()),
            "marge_fine": float(g2[r2.argmax()]), "resultat_fin": float(r2.max())}


# ----------------------------------------------------------------------------
# Version agrégée (dashboard) : les primes de chaque assureur ne dépendent que des variables
# tarifaires et de l'exposition, donc le choix de l'assuré est identique pour tous les contrats
# d'une même cellule (Nbpieces × Nbpers × Qualite × Local). Le résultat est exactement celui de
# la simulation contrat par contrat, pour un calcul instantané.
# ----------------------------------------------------------------------------
def simuler_cellules(cell: pd.DataFrame, marges: dict[str, float]) -> dict:
    noms = list(marges)
    P = np.column_stack([cell[f"PP {n}"].to_numpy() * (1 + marges[n]) for n in noms])  # prime annuelle
    choix = P.argmin(axis=1)
    expo, charge, nb = cell["Exposition"].to_numpy(), cell["Charge"].to_numpy(), cell["Nb_contrats"].to_numpy()
    lignes = []
    for j, n in enumerate(noms):
        m = choix == j
        primes = (P[m, j] * expo[m]).sum()
        primes_seul = (P[:, j] * expo).sum()
        lignes.append({"Assureur": n, "Marge": marges[n], "Nb contrats": int(nb[m].sum()),
                       "Part de marché": nb[m].sum() / nb.sum(), "Primes encaissées": primes,
                       "Sinistres payés": charge[m].sum(), "Résultat": primes - charge[m].sum(),
                       "S/P": charge[m].sum() / primes if primes else np.nan,
                       "Résultat seul": primes_seul - charge.sum(), "S/P seul": charge.sum() / primes_seul})
    return {"synthese": pd.DataFrame(lignes), "choix": np.array(noms)[choix]}


def repartition_cellules(cell: pd.DataFrame, choix: np.ndarray, var: str, noms: list[str]) -> pd.DataFrame:
    d = pd.DataFrame({"Assureur": choix, var: cell[var].to_numpy(), "n": cell["Nb_contrats"].to_numpy()})
    t = d.pivot_table(index="Assureur", columns=var, values="n", aggfunc="sum", fill_value=0)
    return (t.div(t.sum(axis=1), axis=0)).reindex(noms)


def balayage_cellules(cell: pd.DataFrame, marges: dict[str, float], nom: str, grille: np.ndarray):
    res, parts = [], []
    for g in grille:
        mg = dict(marges); mg[nom] = g
        s = simuler_cellules(cell, mg)["synthese"].set_index("Assureur")
        res.append(s.loc[nom, "Résultat"]); parts.append(s.loc[nom, "Part de marché"])
    return np.array(res), np.array(parts)
