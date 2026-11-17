"""Bibliothèque de graphiques Plotly, partagée par les notebooks et le dashboard.

Les types de graphiques reprennent ceux du cours (barres d'exposition + fréquence, courbes par
exercice, bulles Nbpieces × Nbpers, histogrammes de coût, relativités Kassu, V de Cramer,
adéquation des lois...) avec la charte bleu nuit / or du logo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme as T
from .data import LIBELLES, ORDRE_MODALITES

LIB = lambda v: LIBELLES.get(v, v)  # noqa: E731


def _lueur(fig: go.Figure, x, y, couleur=T.OR, secondary_y=None, **kw):
    """Effet « or lumineux » : une large trace translucide sous la courbe principale."""
    args = dict(x=x, y=y, mode="lines", hoverinfo="skip", showlegend=False,
                line=dict(color=couleur, width=12, shape=kw.get("shape", "linear")), opacity=0.18)
    if secondary_y is None:
        fig.add_trace(go.Scatter(**args))
    else:
        fig.add_trace(go.Scatter(**args), secondary_y=secondary_y)


def _degrade_barres(n: int) -> list[str]:
    """Dégradé de bleus (du royal au bleu vif) pour une série de barres."""
    base = np.array([0x0B, 0x3A, 0x78]); fin = np.array([0x3D, 0x7B, 0xFF])
    if n <= 1:
        return [T.ROYAL]
    return ["#%02X%02X%02X" % tuple((base + (fin - base) * i / (n - 1)).astype(int)) for i in range(n)]


# =============================================================================
# Exploration
# =============================================================================
def kpi_exercice(tab: pd.DataFrame, garantie: str = "DDE") -> go.Figure:
    """Exposition (barres) et fréquence / coût moyen (courbes) par exercice."""
    t = tab[tab["Exercice"] != "Total"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=t["Exercice"], y=t["Exposition"], name="Exposition (années)",
                         marker=dict(color=_degrade_barres(len(t))),
                         hovertemplate="%{x}<br>Exposition : %{y:,.0f}<extra></extra>"), secondary_y=False)
    _lueur(fig, t["Exercice"], t["Fréquence"] * 100, secondary_y=True, shape="spline")
    fig.add_trace(go.Scatter(x=t["Exercice"], y=t["Fréquence"] * 100, name=f"Fréquence {garantie} (%)",
                             mode="lines+markers+text", line=dict(color=T.OR, width=3, shape="spline"),
                             marker=dict(size=10, color=T.OR_CLAIR, line=dict(color=T.NUIT, width=2)),
                             text=[T.pct_txt(v) for v in t["Fréquence"] * 100], textposition="top center",
                             textfont=dict(color=T.OR_CLAIR, size=11),
                             hovertemplate="%{x}<br>Fréquence : %{y:.2f} %<extra></extra>"), secondary_y=True)
    fig.update_layout(title=T.titre(f"Portefeuille et sinistralité {garantie} par exercice",
                                    "Barres : exposition en années-polices · Courbe : fréquence annuelle"),
                      height=460)
    fig.update_yaxes(title_text="Exposition", secondary_y=False, tickformat=",.0f")
    fig.update_yaxes(title_text="Fréquence (%)", secondary_y=True, showgrid=False, tickformat=".1f", ticksuffix=" %",
                     range=[0, max(t["Fréquence"]) * 100 * 1.35])
    return fig


def cout_moyen_exercice(tab: pd.DataFrame, garantie: str = "DDE") -> go.Figure:
    t = tab[tab["Exercice"] != "Total"]
    moy = tab.loc[tab["Exercice"] == "Total", "Coût moyen"].iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=t["Exercice"], y=t["Coût moyen"], name="Coût moyen",
                         marker=dict(color=_degrade_barres(len(t)), line=dict(color=T.BLEU_VIF, width=1)),
                         text=[T.fmt_eur(v) for v in t["Coût moyen"]], textposition="outside",
                         textfont=dict(color=T.CHAMPAGNE),
                         hovertemplate="%{x}<br>Coût moyen : %{y:,.0f} €<extra></extra>"))
    fig.add_hline(y=moy, line=dict(color=T.OR, dash="dash", width=2),
                  annotation_text=f"Moyenne {T.fmt_eur(moy)}", annotation_font_color=T.OR_CLAIR,
                  annotation_position="bottom right")
    fig.update_layout(title=T.titre(f"Coût moyen d'un sinistre {garantie} par exercice"),
                      yaxis_title="€", height=420, showlegend=False,
                      yaxis_range=[0, t["Coût moyen"].max() * 1.2])
    return fig


def histogramme_couts(couts: pd.Series, seuils: dict[str, float] | None = None, log: bool = True,
                      garantie: str = "DDE") -> go.Figure:
    """Distribution du coût par sinistre (échelle log) avec les seuils de gravité."""
    x = np.log10(couts[couts > 0]) if log else couts
    fig = go.Figure(go.Histogram(x=x, nbinsx=90, marker=dict(color=T.ROYAL, line=dict(color=T.BLEU_VIF, width=0.5)),
                                 opacity=0.95, hovertemplate="Coût ≈ %{x:.2f}<br>%{y} sinistres<extra></extra>",
                                 name="Sinistres"))
    for i, (nom, s) in enumerate((seuils or {}).items()):
        xs = np.log10(s) if log else s
        fig.add_vline(x=xs, line=dict(color=[T.OR, T.CHAMPAGNE, T.ROUGE][i % 3], width=2, dash="dot"))
        fig.add_annotation(x=xs, y=1, yref="paper", text=f"{nom}<br>{T.fmt_eur(s)}", showarrow=False,
                           font=dict(color=[T.OR, T.CHAMPAGNE, T.ROUGE][i % 3], size=11), xanchor="left",
                           yanchor="top", xshift=4)
    if log:
        ticks = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
        fig.update_xaxes(tickvals=np.log10(ticks), ticktext=[T.fmt_eur(t) for t in ticks])
    fig.update_layout(title=T.titre(f"Distribution du coût par sinistre {garantie}",
                                    "Échelle logarithmique · asymétrie à droite typique des coûts d'assurance"),
                      xaxis_title="Coût du sinistre", yaxis_title="Nombre de sinistres", height=450, showlegend=False)
    return fig


def boite_couts(couts: pd.Series, garantie: str = "DDE") -> go.Figure:
    fig = go.Figure(go.Box(x=couts, boxpoints="suspectedoutliers", name="", orientation="h",
                           marker=dict(color=T.OR, size=4, opacity=0.6), line=dict(color=T.BLEU_VIF),
                           fillcolor="rgba(30,79,191,0.35)", hovertemplate="%{x:,.0f} €<extra></extra>"))
    fig.update_layout(title=T.titre(f"Boîte à moustaches des coûts {garantie}", "Points : valeurs atypiques"),
                      xaxis_type="log", xaxis_title="Coût (€, échelle log)", height=300, showlegend=False)
    return fig


def stabilite_modalites(t: pd.DataFrame, var: str) -> go.Figure:
    """Part d'exposition annuelle de chaque modalité (aires empilées à 100 %)."""
    fig = go.Figure()
    mods = [m for m in ORDRE_MODALITES.get(var, sorted(t[var].unique())) if m in set(t[var].astype(str))]
    for i, m in enumerate(mods):
        s = t[t[var].astype(str) == m]
        fig.add_trace(go.Scatter(x=s["Exercice"], y=s["Part expo"] * 100, name=str(m), stackgroup="un",
                                 mode="lines", line=dict(width=1.5, color=T.SERIES[i % len(T.SERIES)]),
                                 hovertemplate=f"{m}<br>%{{x}} : %{{y:.2f}} %<extra></extra>"))
    fig.update_layout(title=T.titre(f"Stabilité temporelle de « {LIB(var)} »",
                                    "Part de l'exposition annuelle de chaque modalité"),
                      yaxis=dict(title="Part de l'exposition (%)", range=[0, 100], ticksuffix=" %"),
                      xaxis_title="Exercice", height=430)
    return fig


def univarie(tab: pd.DataFrame, var: str, garantie: str = "DDE", indicateur: str = "Fréquence") -> go.Figure:
    """Graphique signature du cours : barres = part d'exposition, points = fréquence (ou CM)."""
    tab = tab.copy()
    tab[var] = tab[var].astype(str)
    est_pct = indicateur == "Fréquence"
    y = tab[indicateur] * (100 if est_pct else 1)
    moy = (tab["Nb_sinistres"].sum() / tab["Exposition"].sum() * 100 if est_pct
           else tab["Charge"].sum() / tab["Nb_sinistres"].sum() if indicateur == "Coût moyen"
           else tab["Charge"].sum() / tab["Exposition"].sum())
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=tab[var], y=tab["Part expo"] * 100, name="Part d'exposition",
                         marker=dict(color=_degrade_barres(len(tab)), line=dict(color=T.BLEU_VIF, width=1)),
                         opacity=0.85, customdata=tab["Exposition"],
                         hovertemplate="%{x}<br>Part d'exposition : %{y:.1f} %<br>Exposition : %{customdata:,.0f}<extra></extra>"),
                  secondary_y=False)
    _lueur(fig, tab[var], y, secondary_y=True)
    suffixe = " %" if est_pct else " €"
    fig.add_trace(go.Scatter(x=tab[var], y=y, name=indicateur, mode="lines+markers+text",
                             line=dict(color=T.OR, width=3),
                             marker=dict(size=14, color=T.OR_CLAIR, line=dict(color=T.NUIT, width=2), symbol="diamond"),
                             text=[(T.pct_txt(v) if est_pct else T.fmt_eur(v)) for v in y],
                             textposition="top center", textfont=dict(color=T.OR_CLAIR, size=12),
                             customdata=np.c_[tab["Nb_sinistres"]],
                             hovertemplate="%{x}<br>" + indicateur + " : %{y:,.2f}" + suffixe +
                                           "<br>Sinistres : %{customdata[0]:,.0f}<extra></extra>"),
                  secondary_y=True)
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y2", y0=moy, y1=moy,
                  line=dict(color=T.CHAMPAGNE, dash="dot", width=1.5))
    fig.add_annotation(xref="paper", x=0.01, yref="y2", y=moy, yanchor="bottom", xanchor="left", showarrow=False,
                       text="Portefeuille : " + (T.pct_txt(moy) if est_pct else T.fmt_eur(moy, 2 if indicateur == "Prime pure" else 0)),
                       font=dict(color=T.CHAMPAGNE, size=11))
    fig.update_layout(title=T.titre(f"{indicateur} {garantie} selon « {LIB(var)} »",
                                    "Barres : part d'exposition · Losanges dorés : " + indicateur.lower()),
                      height=470, bargap=0.35)
    fig.update_yaxes(title_text="Part d'exposition (%)", secondary_y=False, ticksuffix=" %")
    fig.update_yaxes(title_text=indicateur + (" (%)" if est_pct else " (€)"), secondary_y=True, showgrid=False,
                     range=[0, float(np.nanmax(y)) * 1.3], tickformat=".1f" if est_pct else ",.0f",
                     ticksuffix=" %" if est_pct else " €")
    fig.update_xaxes(title_text=LIB(var), type="category")
    return fig


def evolution_par_modalite(t: pd.DataFrame, var: str, indicateur: str = "Fréquence",
                           garantie: str = "DDE") -> go.Figure:
    """Stabilité annuelle de la fréquence ou du CM, une courbe par modalité."""
    fig = go.Figure()
    mods = [m for m in ORDRE_MODALITES.get(var, []) if m in set(t[var].astype(str))]
    est_pct = indicateur == "Fréquence"
    for i, m in enumerate(mods):
        s = t[t[var].astype(str) == m]
        y = s[indicateur] * (100 if est_pct else 1)
        c = T.SERIES[i % len(T.SERIES)]
        fig.add_trace(go.Scatter(x=s["Exercice"], y=y, name=str(m), mode="lines+markers",
                                 line=dict(color=c, width=3, shape="spline"),
                                 marker=dict(size=9, color=c, line=dict(color=T.NUIT, width=2)),
                                 customdata=s["Nb_sinistres"],
                                 hovertemplate=f"{m} · %{{x}}<br>{indicateur} : %{{y:,.2f}}"
                                               + (" %" if est_pct else " €") +
                                               "<br>Sinistres : %{customdata:,.0f}<extra></extra>"))
    fig.update_layout(title=T.titre(f"Stabilité annuelle : {indicateur.lower()} {garantie} par « {LIB(var)} »",
                                    "Une modalité peu exposée produit une courbe instable (peu de sinistres)"),
                      xaxis_title="Exercice", yaxis_title=indicateur + (" (%)" if est_pct else " (€)"), height=450)
    return fig


def bulles_croisement(tab: pd.DataFrame, var_x: str = "Nbpieces", var_y: str = "Nbpers",
                      garantie: str = "DDE") -> go.Figure:
    """Graphique bivarié du cours : taille = fréquence, couleur = coût moyen."""
    t = tab.copy()
    t[var_x] = t[var_x].astype(str); t[var_y] = t[var_y].astype(str)
    taille = t["Fréquence"] / t["Fréquence"].max()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[var_x], y=t[var_y], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(size=taille * 95 + 18, color="rgba(245,197,66,0.10)", line=dict(width=0))))
    fig.add_trace(go.Scatter(
        x=t[var_x], y=t[var_y], mode="markers+text", text=[T.pct_txt(v, 1) for v in t["Fréquence"] * 100],
        textfont=dict(color=T.BLANC, size=11), showlegend=False,
        marker=dict(size=taille * 75 + 12, color=t["Coût moyen"], coloraxis="coloraxis",
                    line=dict(color=T.OR_CLAIR, width=1.5), opacity=0.95),
        customdata=np.c_[t["Fréquence"] * 100, t["Coût moyen"], t["Exposition"], t["Nb_sinistres"]],
        hovertemplate=f"{LIB(var_x)} : %{{x}}<br>{LIB(var_y)} : %{{y}}<br>Fréquence : %{{customdata[0]:.2f}} %"
                      "<br>Coût moyen : %{customdata[1]:,.0f} €<br>Exposition : %{customdata[2]:,.0f}"
                      "<br>Sinistres : %{customdata[3]:,.0f}<extra></extra>"))
    fig.update_layout(title=T.titre(f"Croisement « {LIB(var_x)} » × « {LIB(var_y)} » ({garantie})",
                                    "Taille de la bulle : fréquence · Couleur : coût moyen"),
                      coloraxis=dict(colorscale=T.ECHELLE_NUIT_OR, colorbar=dict(title="CM (€)")),
                      xaxis=dict(title=LIB(var_x), categoryorder="array", categoryarray=ORDRE_MODALITES.get(var_x)),
                      yaxis=dict(title=LIB(var_y), categoryorder="array", categoryarray=ORDRE_MODALITES.get(var_y),
                                 showgrid=False),
                      height=520)
    return fig


def heatmap_croisement(tab: pd.DataFrame, var_x: str, var_y: str, indicateur: str = "Fréquence") -> go.Figure:
    t = tab.pivot(index=var_y, columns=var_x, values=indicateur)
    t = t.reindex(index=[m for m in ORDRE_MODALITES.get(var_y, t.index) if m in t.index],
                  columns=[m for m in ORDRE_MODALITES.get(var_x, t.columns) if m in t.columns])
    est_pct = indicateur == "Fréquence"
    z = t.values * (100 if est_pct else 1)
    fig = go.Figure(go.Heatmap(z=z, x=[str(c) for c in t.columns], y=[str(i) for i in t.index],
                               colorscale=T.ECHELLE_NUIT_OR, xgap=3, ygap=3,
                               text=[[(T.pct_txt(v) if est_pct else T.fmt_eur(v)) if pd.notna(v) else "" for v in r] for r in z],
                               texttemplate="%{text}", textfont=dict(size=12),
                               colorbar=dict(title="%" if est_pct else "€"),
                               hovertemplate=f"{var_x} : %{{x}}<br>{var_y} : %{{y}}<br>{indicateur} : %{{z:,.2f}}<extra></extra>"))
    fig.update_layout(title=T.titre(f"{indicateur} par croisement « {LIB(var_x)} » × « {LIB(var_y)} »"),
                      xaxis_title=LIB(var_x), yaxis_title=LIB(var_y), height=460)
    return fig


def sinistralite_age(tab: pd.DataFrame, garantie: str = "DDE", expo_min: float = 500) -> go.Figure:
    """Fréquence et coût moyen par âge détaillé, avec lissage (moyenne mobile pondérée)."""
    t = tab[tab["Exposition"] >= expo_min].sort_values("Age_num").copy()
    w = 5
    lisse = lambda col: (t[col] * t["Exposition"]).rolling(w, center=True, min_periods=1).sum() / \
        t["Exposition"].rolling(w, center=True, min_periods=1).sum()  # noqa: E731
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Fréquence (%)", "Coût moyen (€)"), horizontal_spacing=0.09)
    fig.add_trace(go.Bar(x=t["Age_num"], y=t["Fréquence"] * 100, marker_color="rgba(30,79,191,0.55)",
                         name="Fréquence observée", hovertemplate="%{x} ans : %{y:.2f} %<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=t["Age_num"], y=lisse("Fréquence") * 100, name="Tendance lissée",
                             line=dict(color=T.OR, width=4, shape="spline"),
                             hovertemplate="%{x} ans · lissé : %{y:.2f} %<extra></extra>"), 1, 1)
    fig.add_trace(go.Bar(x=t["Age_num"], y=t["Coût moyen"], marker_color="rgba(138,105,67,0.6)",
                         name="CM observé", hovertemplate="%{x} ans : %{y:,.0f} €<extra></extra>"), 1, 2)
    fig.add_trace(go.Scatter(x=t["Age_num"], y=lisse("Coût moyen"), name="CM lissé",
                             line=dict(color=T.BLEU_VIF, width=4, shape="spline"),
                             hovertemplate="%{x} ans · lissé : %{y:,.0f} €<extra></extra>"), 1, 2)
    for x0, x1 in [(25.5, 35.5), (50.5, 65.5)]:
        for c in (1, 2):
            fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(231,201,128,0.05)", line_width=0, row=1, col=c)
    fig.update_layout(title=T.titre(f"Sinistralité {garantie} par âge de l'assuré",
                                    f"Âges avec au moins {expo_min:.0f} années d'exposition · bandes : classes d'âge Age2"),
                      height=460, bargap=0.1)
    fig.update_annotations(font=dict(color=T.CHAMPAGNE, size=13))
    fig.update_xaxes(title_text="Âge")
    return fig


def _couleur_echelle(v: float, echelle=T.ECHELLE_NUIT_OR) -> str:
    from plotly.colors import sample_colorscale
    return sample_colorscale(echelle, [min(max(v, 0.0), 1.0)])[0]


def carte_departements(tab: pd.DataFrame, geojson: dict, indicateur: str = "Fréquence",
                       garantie: str = "DDE") -> go.Figure:
    """Carte choroplèthe dessinée en polygones Plotly (aucun fond de carte externe requis)."""
    est_pct = indicateur == "Fréquence"
    t = tab.set_index("Dep")
    vals = t[indicateur] * (100 if est_pct else 1)
    vmin, vmax = np.nanquantile(vals, 0.02), np.nanquantile(vals, 0.98)
    k = np.cos(np.radians(46.5))
    fig = go.Figure()
    for f in geojson["features"]:
        code, nom = f["properties"]["code"], f["properties"]["nom"]
        if code in ("2A", "2B") and code not in vals.index and "20" in vals.index:
            code = "20"  # la base code la Corse en « 20 »
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        xs, ys = [], []
        for poly in polys:
            ring = np.asarray(poly[0])
            xs += list(ring[:, 0] * k) + [None]; ys += list(ring[:, 1]) + [None]
        if code in vals.index and pd.notna(vals[code]):
            v = vals[code]; coul = _couleur_echelle((v - vmin) / (vmax - vmin))
            txt = (f"<b>{code} · {nom}</b><br>{indicateur} : " + (f"{v:.2f} %" if est_pct else T.fmt_eur(v, 2)) +
                   f"<br>Exposition : {T.fmt_nb(t.loc[code, 'Exposition'])}<br>Sinistres : {T.fmt_nb(t.loc[code, 'Nb_sinistres'])}")
        else:
            coul, txt = "#24365E", f"{code} · {nom}<br>pas de données"
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=coul, hoveron="fills",
                                 line=dict(color=T.NUIT, width=0.7), text=txt, hoverinfo="text", showlegend=False))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(colorscale=T.ECHELLE_NUIT_OR, cmin=vmin, cmax=vmax, color=[vmin],
                                         colorbar=dict(title="%" if est_pct else "€", outlinewidth=0, thickness=12,
                                                       tickfont=dict(color=T.CHAMPAGNE)))))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(title=T.titre(f"{indicateur} {garantie} par département",
                                    "Les zones urbaines et littorales concentrent la sinistralité"),
                      height=640, margin=dict(l=10, r=10, t=90, b=10), hovermode="closest")
    return fig


def classement_departements(tab: pd.DataFrame, indicateur: str = "Fréquence", n: int = 15) -> go.Figure:
    est_pct = indicateur == "Fréquence"
    t = tab.sort_values(indicateur, ascending=False)
    t = pd.concat([t.head(n), t.tail(n)])
    v = t[indicateur] * (100 if est_pct else 1)
    couleurs = [T.OR if i < n else T.BLEU_VIF for i in range(len(t))]
    fig = go.Figure(go.Bar(x=t["Dep"], y=v, marker_color=couleurs,
                           hovertemplate="Dép. %{x}<br>" + indicateur + " : %{y:,.2f}<extra></extra>"))
    fig.update_layout(title=T.titre(f"Les {n} départements les plus et les moins sinistrés",
                                    "Or : plus sinistrés · Bleu : moins sinistrés"),
                      xaxis=dict(type="category", title="Département"),
                      yaxis_title=indicateur + (" (%)" if est_pct else " (€)"), height=420)
    return fig


def heatmap_cramer(m: pd.DataFrame) -> go.Figure:
    masque = np.triu(np.ones_like(m.values, dtype=bool), 1)
    z = np.where(masque, np.nan, m.values)
    fig = go.Figure(go.Heatmap(z=z, x=m.columns, y=m.index, colorscale=T.ECHELLE_NUIT_OR, zmin=0, zmax=1,
                               xgap=2, ygap=2, text=np.where(np.isnan(z), "", np.round(z, 2).astype(str)),
                               texttemplate="%{text}", textfont=dict(size=10),
                               colorbar=dict(title="V de Cramer"),
                               hovertemplate="%{y} × %{x}<br>V de Cramer : %{z:.3f}<extra></extra>"))
    fig.update_layout(title=T.titre("Liaisons entre variables tarifaires : V de Cramer",
                                    "0 = indépendance · 1 = liaison parfaite"),
                      yaxis=dict(autorange="reversed", showgrid=False), height=640,
                      xaxis=dict(tickangle=-40))
    return fig


def adequation_poisson(k, p_obs, p_th, garantie: str = "DDE") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=k, y=p_obs, name="Observé", marker_color=T.ROYAL,
                         hovertemplate="k = %{x}<br>Observé : %{y:.5%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=k, y=p_th, name="Poisson ajustée", mode="lines+markers",
                             line=dict(color=T.OR, width=3), marker=dict(size=11, symbol="diamond", color=T.OR_CLAIR),
                             hovertemplate="k = %{x}<br>Poisson : %{y:.5%}<extra></extra>"))
    fig.update_layout(title=T.titre(f"Sinistres {garantie} : observé vs Poisson",
                                    "Échelle logarithmique pour lire les queues (k ≥ 2)"),
                      yaxis=dict(type="log", title="Proportion des lignes", tickformat=".0e"),
                      xaxis=dict(title="Nombre de sinistres par ligne", dtick=1), height=440)
    return fig


def adequation_couts(couts: np.ndarray, lois: dict, garantie: str = "DDE") -> go.Figure:
    """Histogramme des coûts + densités ajustées (gauche) et QQ-plots (droite)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Densités ajustées", "QQ-plot (quantiles théoriques vs observés)"),
                        horizontal_spacing=0.1)
    x_max = np.quantile(couts, 0.99)
    fig.add_trace(go.Histogram(x=couts[couts <= x_max], histnorm="probability density", nbinsx=120,
                               marker_color="rgba(30,79,191,0.55)", name="Observé"), 1, 1)
    grille = np.linspace(np.quantile(couts, 0.01), x_max, 400)
    dens_max = np.histogram(couts[couts <= x_max], bins=120, density=True)[0].max()
    q = np.linspace(0.005, 0.995, 200)
    q_obs = np.quantile(couts, q)
    couleurs = [T.OR, T.BLEU_PALE, T.CHAMPAGNE]
    for i, (nom, loi) in enumerate(lois.items()):
        fig.add_trace(go.Scatter(x=grille, y=loi.pdf(grille), name=nom, line=dict(color=couleurs[i], width=3)), 1, 1)
        fig.add_trace(go.Scatter(x=loi.ppf(q), y=q_obs, mode="markers", name=f"QQ {nom}", showlegend=False,
                                 marker=dict(color=couleurs[i], size=6, opacity=0.85)), 1, 2)
    lim = [0, np.quantile(couts, 0.995)]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=T.GRIS, dash="dash"), showlegend=False), 1, 2)
    fig.update_xaxes(title_text="Coût (€)", row=1, col=1)
    fig.update_xaxes(title_text="Quantile théorique (€)", row=1, col=2, range=lim)
    fig.update_yaxes(title_text="Quantile observé (€)", row=1, col=2, range=lim)
    fig.update_yaxes(range=[0, dens_max * 1.25], row=1, col=1)
    fig.update_layout(title=T.titre(f"Coût des sinistres {garantie} : Gamma vs log-normale"), height=490,
                      legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5))
    fig.update_annotations(font=dict(color=T.CHAMPAGNE))
    return fig


# =============================================================================
# Modélisation
# =============================================================================
def comparaison_modeles(crit: pd.DataFrame, colonnes=("AIC", "BIC", "Déviance")) -> go.Figure:
    """Écart de chaque modèle au meilleur sur chaque critère (plus bas = meilleur)."""
    colonnes = [c for c in colonnes if c in crit.columns]
    fig = make_subplots(rows=1, cols=len(colonnes), subplot_titles=[f"{c} (écart au meilleur)" for c in colonnes],
                        horizontal_spacing=0.06)
    for j, c in enumerate(colonnes, start=1):
        ecart = crit[c] - crit[c].min()
        best = ecart == 0
        fig.add_trace(go.Bar(y=crit["Modèle"], x=ecart, orientation="h", showlegend=False,
                             marker_color=[T.OR if b else T.ROYAL for b in best],
                             text=[("★ meilleur" if b else "+" + T.fmt_nb(v, 5 if ecart.max() < 1 else 1))
                                   for v, b in zip(ecart, best)],
                             textposition="outside", textfont=dict(color=T.CHAMPAGNE, size=11), cliponaxis=False,
                             customdata=crit[c], hovertemplate="%{y}<br>" + c + " : %{customdata:,.1f}<extra></extra>"),
                      1, j)
        fig.update_yaxes(showticklabels=(j == 1), autorange="reversed", row=1, col=j)
        fig.update_xaxes(showgrid=False, showticklabels=False, range=[0, ecart.max() * 1.35 + 1], row=1, col=j)
    fig.update_layout(title=T.titre("Comparaison des modèles GLM", "Barre dorée : modèle retenu par le critère"),
                      height=170 + 55 * len(crit), margin=dict(l=170))
    fig.update_annotations(font=dict(color=T.CHAMPAGNE))
    return fig


def relativites_variable(rel: pd.DataFrame, var: str, titre_modele: str = "") -> go.Figure:
    r = rel[rel["Variable"] == var]
    fig = go.Figure()
    fig.add_hrect(y0=0.95, y1=1.05, fillcolor="rgba(231,201,128,0.06)", line_width=0)
    fig.add_hline(y=1, line=dict(color=T.CHAMPAGNE, dash="dot", width=1))
    fig.add_trace(go.Bar(x=r["Modalité"], y=r["Relativité"] - 1, base=1, name="Relativité",
                         marker=dict(color=[T.GRIS if ref else (T.OR if v >= 1 else T.BLEU_VIF)
                                            for v, ref in zip(r["Relativité"], r["Référence"])],
                                     line=dict(width=0)),
                         error_y=dict(type="data", symmetric=False, array=r["IC haut"] - r["Relativité"],
                                      arrayminus=r["Relativité"] - r["IC bas"], color=T.BLANC, thickness=1.2, width=5),
                         text=[("réf." if ref else f"×{v:.2f}") for v, ref in zip(r["Relativité"], r["Référence"])],
                         textposition="outside", textfont=dict(color=T.OR_CLAIR, size=12), cliponaxis=False,
                         customdata=np.c_[r["IC bas"], r["IC haut"], r["p-value"]],
                         hovertemplate="%{x}<br>Relativité : %{y:.3f}<br>IC 95 % : [%{customdata[0]:.3f} ; "
                                       "%{customdata[1]:.3f}]<br>p-value : %{customdata[2]:.2e}<extra></extra>"))
    fig.update_layout(title=T.titre(f"Relativités « {LIB(var)} »" + (f" · {titre_modele}" if titre_modele else ""),
                                    "Barres depuis 1 (référence) · Moustaches : IC à 95 %"),
                      yaxis_title="Coefficient multiplicatif", xaxis_title=LIB(var), height=430, showlegend=False,
                      xaxis_type="category")
    return fig


def relativites_grille(rel: pd.DataFrame, variables: list[str], titre_txt: str, cols: int = 3) -> go.Figure:
    """Toutes les relativités d'un modèle en petits multiples."""
    rows = int(np.ceil(len(variables) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[LIB(v) for v in variables],
                        vertical_spacing=0.12 if rows > 2 else 0.18, horizontal_spacing=0.06)
    for i, var in enumerate(variables):
        r = rel[rel["Variable"] == var]
        ri, ci = i // cols + 1, i % cols + 1
        fig.add_trace(go.Bar(x=r["Modalité"], y=r["Relativité"] - 1, base=1, showlegend=False,
                             marker_color=[T.GRIS if ref else (T.OR if v >= 1 else T.BLEU_VIF)
                                           for v, ref in zip(r["Relativité"], r["Référence"])],
                             error_y=dict(type="data", symmetric=False, array=r["IC haut"] - r["Relativité"],
                                          arrayminus=r["Relativité"] - r["IC bas"], color="rgba(255,255,255,0.7)",
                                          thickness=1, width=3),
                             customdata=np.c_[r["IC bas"], r["IC haut"]],
                             hovertemplate=f"{var} = %{{x}}<br>×%{{y:.3f}}<br>IC : [%{{customdata[0]:.2f}} ; "
                                           "%{customdata[1]:.2f}]<extra></extra>"), ri, ci)
        fig.add_hline(y=1, line=dict(color=T.CHAMPAGNE, dash="dot", width=1), row=ri, col=ci)
    fig.update_layout(title=T.titre(titre_txt, "Or : majoration · Bleu : minoration · Gris : modalité de référence"),
                      height=290 * rows + 80)
    fig.update_annotations(font=dict(color=T.OR_CLAIR, size=13))
    fig.update_xaxes(tickfont=dict(size=10), type="category")
    return fig


def courbe_kassu(rel: pd.DataFrame, expo: pd.Series, titre_txt: str = "Relativités du capital assuré (Kassu)") -> go.Figure:
    """Graphique du cours : relativités Kassu, taille des points = exposition."""
    r = rel[rel["Variable"] == "Kassu"].copy()
    r["expo"] = r["Modalité"].map(expo).fillna(0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r["Modalité"], y=r["IC haut"], mode="lines", line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=r["Modalité"], y=r["IC bas"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(61,123,255,0.18)", name="IC 95 %", hoverinfo="skip"))
    _lueur(fig, r["Modalité"], r["Relativité"])
    fig.add_trace(go.Scatter(x=r["Modalité"], y=r["Relativité"], mode="lines+markers", name="Relativité",
                             line=dict(color=T.OR, width=3),
                             marker=dict(size=np.sqrt(r["expo"]) / np.sqrt(r["expo"].max()) * 40 + 8,
                                         color=T.OR_CLAIR, line=dict(color=T.NUIT, width=2)),
                             customdata=r["expo"],
                             hovertemplate="%{x}<br>Relativité : %{y:.3f}<br>Exposition : %{customdata:,.0f}<extra></extra>"))
    fig.update_layout(title=T.titre(titre_txt, "Taille du point : exposition · une courbe non monotone signale une anomalie"),
                      yaxis_title="Relativité", xaxis_title="Capital assuré", height=450)
    return fig


def importance_barres(t: pd.DataFrame, col_nom: str, col_val: str, titre_txt: str, sous_titre: str = "",
                      seuil: float | None = None) -> go.Figure:
    t = t.sort_values(col_val)
    if seuil is None and "p-value" in t.columns:  # or = significatif à 5 %
        couleurs = [T.OR if p < 0.05 else T.GRIS for p in t["p-value"]]
    else:
        couleurs = [T.OR if (seuil is None or v > seuil) else T.GRIS for v in t[col_val]]
    fig = go.Figure(go.Bar(x=t[col_val], y=t[col_nom], orientation="h", marker_color=couleurs,
                           text=[T.fmt_nb(v, 1).replace(".", ",") for v in t[col_val]], textposition="outside",
                           textfont=dict(color=T.CHAMPAGNE), cliponaxis=False,
                           hovertemplate="%{y}<br>" + col_val + " : %{x:,.1f}<extra></extra>"))
    if seuil is not None:
        fig.add_vline(x=seuil, line=dict(color=T.ROUGE, dash="dash"))
    fig.update_layout(title=T.titre(titre_txt, sous_titre), height=140 + 34 * len(t), xaxis_title=col_val,
                      yaxis=dict(showgrid=False), margin=dict(l=170))
    return fig


def trajectoire_selection(hist_b: pd.DataFrame, hist_f: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for h, nom, c in [(hist_b, "Backward", T.OR), (hist_f, "Forward", T.BLEU_VIF)]:
        fig.add_trace(go.Scatter(x=h["Nb variables"], y=h["Critère"], name=nom, mode="lines+markers+text",
                                 line=dict(color=c, width=3), marker=dict(size=10, line=dict(color=T.NUIT, width=2)),
                                 text=h["Variable"].where(h["Variable"] != "-", ""), textposition="top center",
                                 textfont=dict(color=c, size=10),
                                 hovertemplate=nom + "<br>%{x} variables<br>AIC : %{y:,.1f}<br>%{text}<extra></extra>"))
    fig.update_layout(title=T.titre("Sélection pas à pas : backward vs forward",
                                    "AIC selon le nombre de variables · les deux chemins convergent-ils ?"),
                      xaxis=dict(title="Nombre de variables", dtick=1), yaxis_title="AIC", height=480)
    return fig


def lift(t: pd.DataFrame, titre_txt: str = "Courbe de lift de la prime pure (base test)") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=t["Décile"], y=t["PP observée"], name="Prime pure observée", marker_color=T.ROYAL,
                         hovertemplate="Décile %{x}<br>Observée : %{y:,.2f} €<extra></extra>"))
    _lueur(fig, t["Décile"], t["PP prédite"])
    fig.add_trace(go.Scatter(x=t["Décile"], y=t["PP prédite"], name="Prime pure prédite", mode="lines+markers",
                             line=dict(color=T.OR, width=3), marker=dict(size=10, color=T.OR_CLAIR, symbol="diamond"),
                             hovertemplate="Décile %{x}<br>Prédite : %{y:,.2f} €<extra></extra>"))
    ratio = t["PP observée"].iloc[-1] / t["PP observée"].iloc[0]
    fig.update_layout(title=T.titre(titre_txt, f"Déciles d'exposition triés par prime prédite · écart décile 10 / décile 1 : ×{ratio:.1f}"),
                      xaxis=dict(title="Décile de risque prédit", dtick=1), yaxis_title="Prime pure annuelle (€)",
                      height=450)
    return fig


def lorenz(courbes: dict[str, pd.DataFrame], ginis: dict[str, float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=T.GRIS, dash="dash"),
                             name="Aléatoire"))
    for i, (nom, c) in enumerate(courbes.items()):
        fig.add_trace(go.Scatter(x=c["Part exposition"], y=c["Part charge"], mode="lines",
                                 name=f"{nom} (Gini {ginis[nom]:.3f})",
                                 line=dict(color=[T.OR, T.BLEU_VIF, T.CHAMPAGNE, T.BRONZE][i % 4], width=3)))
    fig.update_layout(title=T.titre("Pouvoir discriminant : courbes de Lorenz", "Plus la courbe est creusée, meilleure est la segmentation"),
                      xaxis=dict(title="Part cumulée de l'exposition", tickformat=".0%"),
                      yaxis=dict(title="Part cumulée de la charge", tickformat=".0%"), height=480)
    return fig


def observe_predit(t: pd.DataFrame, var: str) -> go.Figure:
    t = t.copy(); t[var] = t[var].astype(str)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=t[var], y=t["Exposition"], name="Exposition", marker_color="rgba(30,79,191,0.45)"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=t[var], y=t["PP observée"], name="Observé", mode="lines+markers",
                             line=dict(color=T.BLEU_PALE, width=2, dash="dot"), marker=dict(size=9)), secondary_y=True)
    fig.add_trace(go.Scatter(x=t[var], y=t["PP prédite"], name="Prédit", mode="lines+markers",
                             line=dict(color=T.OR, width=3), marker=dict(size=11, symbol="diamond")), secondary_y=True)
    fig.update_layout(title=T.titre(f"Prime pure observée vs prédite selon « {LIB(var)} »", "Base test"), height=430,
                      xaxis_type="category")
    fig.update_yaxes(title_text="Exposition", secondary_y=False)
    fig.update_yaxes(title_text="Prime pure annuelle (€)", secondary_y=True, showgrid=False)
    return fig


def waterfall_prime(detail: list[tuple[str, float]], titre_txt: str, unite: str = "€") -> go.Figure:
    """Décomposition multiplicative du tarif en cascade (effet de chaque variable en €)."""
    noms, effets, valeur = [], [], None
    for nom, r in detail:
        if valeur is None:
            valeur = r; noms.append(nom); effets.append(r)
        else:
            nouv = valeur * r; noms.append(LIB(nom)); effets.append(nouv - valeur); valeur = nouv
    fig = go.Figure(go.Waterfall(
        x=noms + ["Total"], y=effets + [valeur], measure=["absolute"] + ["relative"] * (len(effets) - 1) + ["total"],
        connector=dict(line=dict(color="rgba(231,201,128,0.35)", dash="dot")),
        increasing=dict(marker=dict(color=T.OR)), decreasing=dict(marker=dict(color=T.BLEU_VIF)),
        totals=dict(marker=dict(color=T.OR_CLAIR, line=dict(color=T.BLANC, width=1))),
        text=[(("+" if e >= 0 else "−") + T.fmt_nb(abs(e), 2).replace(".", ",")) if i else T.fmt_nb(e, 2).replace(".", ",")
              for i, e in enumerate(effets)] + [T.fmt_nb(valeur, 2).replace(".", ",") + f" {unite}"],
        textposition="outside", textfont=dict(color=T.CHAMPAGNE),
        hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>"))
    fig.update_layout(title=T.titre(titre_txt, "Or : majoration · Bleu : minoration"), height=460,
                      yaxis_title=unite, showlegend=False)
    return fig


# =============================================================================
# Marché concurrentiel
# =============================================================================
def parts_de_marche(parts: dict[str, float], titre_txt: str = "Parts de marché") -> go.Figure:
    noms = list(parts)
    fig = go.Figure(go.Pie(labels=noms, values=[parts[n] for n in noms], hole=0.62, sort=False,
                           marker=dict(colors=[T.ASSUREURS.get(n, T.ROYAL) for n in noms],
                                       line=dict(color=T.NUIT, width=4)),
                           textinfo="label+percent", textfont=dict(color=T.BLANC, size=13),
                           hovertemplate="%{label}<br>Part : %{percent}<extra></extra>"))
    fig.add_annotation(text="<b>Marché</b><br>simulé", showarrow=False, font=dict(color=T.OR_CLAIR, size=16))
    fig.update_layout(title=T.titre(titre_txt), height=420, showlegend=False)
    return fig


def resultats_assureurs(seul: dict[str, float], marche: dict[str, float]) -> go.Figure:
    noms = list(seul)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=noms, y=[seul[n] for n in noms], name="Seul sur le marché",
                         marker=dict(color="rgba(159,184,255,0.25)", line=dict(color=T.BLEU_PALE, width=1.5)),
                         text=[T.fmt_eur(seul[n]) for n in noms], textposition="outside",
                         textfont=dict(color=T.BLEU_PALE)))
    fig.add_trace(go.Bar(x=noms, y=[marche[n] for n in noms], name="En concurrence",
                         marker_color=[T.VERT if marche[n] >= 0 else T.ROUGE for n in noms],
                         text=[T.fmt_eur(marche[n]) for n in noms], textposition="outside",
                         textfont=dict(color=T.BLANC)))
    fig.add_hline(y=0, line=dict(color=T.CHAMPAGNE, width=1))
    fig.update_layout(title=T.titre("Résultat technique : primes encaissées − sinistres payés",
                                    "Base test · seul sur le marché vs marché concurrentiel"),
                      barmode="group", yaxis_title="€", height=460)
    return fig


def repartition_variable(rep: pd.DataFrame, var: str) -> go.Figure:
    """Répartition (en %) de la variable dans le portefeuille de chaque assureur."""
    fig = go.Figure()
    mods = [m for m in ORDRE_MODALITES.get(var, rep.columns) if m in rep.columns]
    for i, m in enumerate(mods):
        fig.add_trace(go.Bar(y=rep.index, x=rep[m] * 100, name=str(m), orientation="h",
                             marker=dict(color=T.SERIES[i % len(T.SERIES)], line=dict(color=T.NUIT, width=1)),
                             text=[f"{v:.0f}%" if v >= 5 else "" for v in rep[m] * 100], textposition="inside",
                             insidetextanchor="middle", textfont=dict(color=T.NUIT, size=12),
                             hovertemplate=f"{m}<br>%{{y}} : %{{x:.1f}} %<extra></extra>"))
    fig.update_layout(barmode="stack", title=T.titre(f"Composition des portefeuilles selon « {LIB(var)} »",
                                                     "Qui attire quels risques ?"),
                      xaxis=dict(title="Part du portefeuille (%)", range=[0, 100], ticksuffix=" %"),
                      yaxis=dict(autorange="reversed", showgrid=False), height=380)
    return fig


def courbe_marge(marges: np.ndarray, resultats: np.ndarray, parts: np.ndarray | None = None,
                 optimum: tuple[float, float] | None = None) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if parts is not None:
        fig.add_trace(go.Scatter(x=marges * 100, y=parts * 100, name="Part de marché", fill="tozeroy",
                                 line=dict(color="rgba(61,123,255,0.8)", width=1.5), fillcolor="rgba(30,79,191,0.18)",
                                 hovertemplate="Marge %{x:.1f} %<br>Part : %{y:.1f} %<extra></extra>"), secondary_y=True)
    _lueur(fig, marges * 100, resultats, secondary_y=False)
    fig.add_trace(go.Scatter(x=marges * 100, y=resultats, name="Résultat assureur 3", mode="lines",
                             line=dict(color=T.OR, width=3.5),
                             hovertemplate="Marge %{x:.1f} %<br>Résultat : %{y:,.0f} €<extra></extra>"), secondary_y=False)
    if optimum is not None:
        mo, ro = optimum
        fig.add_trace(go.Scatter(x=[mo * 100], y=[ro], mode="markers", name="Optimum",
                                 marker=dict(size=22, color=T.OR_CLAIR, symbol="star", line=dict(color=T.BLANC, width=2))),
                      secondary_y=False)
        fig.add_annotation(x=mo * 100, y=ro, text=f"<b>Marge optimale {T.fmt_pct(mo, 0)}</b><br>{T.fmt_eur(ro)}",
                           showarrow=True, arrowcolor=T.OR_CLAIR, ax=60, ay=-60,
                           font=dict(color=T.OR_CLAIR, size=13), bgcolor="rgba(0,20,49,0.85)", bordercolor=T.OR)
    fig.add_hline(y=0, line=dict(color=T.CHAMPAGNE, width=1, dash="dot"), secondary_y=False)
    fig.update_layout(title=T.titre("Assureur 3 : résultat en fonction de sa marge",
                                    "Les assureurs 1 et 2 gardent leurs tarifs (30 % et 15 %)"),
                      height=480, xaxis=dict(title="Marge de l'assureur 3 (%)", ticksuffix=" %"))
    fig.update_yaxes(title_text="Résultat (€)", secondary_y=False)
    fig.update_yaxes(title_text="Part de marché (%)", secondary_y=True, showgrid=False, ticksuffix=" %")
    return fig


def ratio_sp(sp: dict[str, float]) -> go.Figure:
    noms = list(sp)
    fig = go.Figure(go.Bar(x=noms, y=[sp[n] * 100 for n in noms],
                           marker_color=[T.ROUGE if sp[n] > 1 else T.ASSUREURS.get(n, T.OR) for n in noms],
                           text=[f"{sp[n]:.0%}" for n in noms], textposition="outside", textfont=dict(color=T.BLANC)))
    fig.add_hline(y=100, line=dict(color=T.ROUGE, dash="dash"), annotation_text="S/P = 100 %",
                  annotation_font_color=T.ROUGE)
    fig.update_layout(title=T.titre("Ratio sinistres / primes en concurrence"), yaxis=dict(title="S/P (%)", ticksuffix=" %"),
                      height=400, showlegend=False)
    return fig


# =============================================================================
# Sinistres graves
# =============================================================================
def mean_excess(seuils: np.ndarray, me: np.ndarray, choix: float | None = None, garantie: str = "Vol") -> go.Figure:
    fig = go.Figure()
    _lueur(fig, seuils, me)
    fig.add_trace(go.Scatter(x=seuils, y=me, mode="lines", line=dict(color=T.OR, width=3), name="e(u)",
                             hovertemplate="Seuil %{x:,.0f} €<br>Excès moyen : %{y:,.0f} €<extra></extra>"))
    if choix:
        fig.add_vline(x=choix, line=dict(color=T.BLEU_VIF, dash="dash", width=2),
                      annotation_text=f"Seuil retenu {T.fmt_eur(choix)}", annotation_font_color=T.BLEU_PALE)
    fig.update_layout(title=T.titre(f"Mean excess plot · coûts {garantie}",
                                    "e(u) = E[X − u | X > u] · une rupture de pente signale le début de la queue"),
                      xaxis_title="Seuil u (€)", yaxis_title="Excès moyen (€)", height=440)
    return fig


def top_sinistres(couts: pd.Series, n: int = 20, seuil: float | None = None, garantie: str = "Vol") -> go.Figure:
    top = couts.sort_values(ascending=False).head(n).reset_index(drop=True)
    fig = go.Figure(go.Bar(x=[f"#{i + 1}" for i in range(len(top))], y=top,
                           marker_color=[T.ROUGE if (seuil and v > seuil) else T.ROYAL for v in top],
                           text=[(f"{v / 1e6:.0f} M€" if v >= 1e6 else f"{v / 1e3:.0f} k€") for v in top],
                           textposition="outside", textfont=dict(color=T.CHAMPAGNE, size=11),
                           hovertemplate="%{x}<br>%{y:,.0f} €<extra></extra>"))
    if seuil:
        fig.add_hline(y=seuil, line=dict(color=T.OR, dash="dash"), annotation_text=f"Seuil {T.fmt_eur(seuil)}",
                      annotation_font_color=T.OR)
    ticks = [t for t in [5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6] if top.min() / 2 <= t <= top.max() * 2]
    fig.update_layout(title=T.titre(f"Les {n} plus gros sinistres {garantie}", "Échelle logarithmique · rouge : au-delà du seuil"),
                      yaxis=dict(type="log", title="Coût (€)", tickvals=ticks,
                                 ticktext=[(f"{t / 1e6:g} M€" if t >= 1e6 else f"{t / 1e3:g} k€") for t in ticks]),
                      xaxis=dict(type="category"), height=440, showlegend=False)
    return fig


def comparaison_coefficients(df: pd.DataFrame, titre_txt: str) -> go.Figure:
    """Relativités sans graves / avec graves / avec graves retraités, par modalité."""
    fig = go.Figure()
    couleurs = [T.BLEU_PALE, T.ROUGE, T.OR]
    for i, col in enumerate([c for c in df.columns if c != "Terme"]):
        fig.add_trace(go.Bar(x=df["Terme"], y=df[col], name=col, marker_color=couleurs[i % 3],
                             hovertemplate="%{x}<br>" + col + " : %{y:.3f}<extra></extra>"))
    fig.add_hline(y=1, line=dict(color=T.CHAMPAGNE, dash="dot"))
    fig.update_layout(title=T.titre(titre_txt, "exp(coefficients) · référence = 1"), barmode="group",
                      yaxis_title="Relativité", height=470, xaxis=dict(tickangle=-30))
    return fig
