"""Charte graphique « TEAM » : bleu nuit et or métallique, extraite du logo."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

NUIT = "#001431"        # fond principal
NUIT_2 = "#062A58"      # cartes / fond secondaire
NUIT_3 = "#0B3A78"      # bordures, survol
ROYAL = "#1E4FBF"       # bleu royal
BLEU_VIF = "#3D7BFF"    # bleu lumineux
BLEU_PALE = "#9FB8FF"
OR = "#D4A017"          # or principal
OR_CLAIR = "#F5C542"    # or clair (titres, KPI)
CHAMPAGNE = "#E7C980"   # texte secondaire
BRONZE = "#8A6943"
BLANC = "#FFFFFF"
GRIS = "#8FA3C7"
ROUGE = "#E8566C"       # pertes / alertes (rouge doux, lisible sur bleu nuit)
VERT = "#3CCB9A"        # gains

SERIES = [OR, BLEU_VIF, CHAMPAGNE, ROYAL, BRONZE, BLEU_PALE, OR_CLAIR, "#5E7FB8"]
ASSUREURS = {"Assureur 1": BRONZE, "Assureur 2": BLEU_VIF, "Assureur 3": OR}

# Échelles continues : du bleu nuit vers l'or en passant par le bleu royal
ECHELLE_NUIT_OR = [[0.0, "#0A2A66"], [0.35, ROYAL], [0.6, "#7A86B8"], [0.8, CHAMPAGNE], [1.0, OR_CLAIR]]
ECHELLE_OR = [[0.0, "#2A2410"], [0.5, BRONZE], [1.0, OR_CLAIR]]

POLICE = "Montserrat, 'Segoe UI', Helvetica, Arial, sans-serif"
POLICE_TITRE = "'Playfair Display', Georgia, serif"


def _template() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=POLICE, color=BLANC, size=13),
        separators=", ",  # virgule décimale, espace pour les milliers (format français)
        title=dict(font=dict(family=POLICE_TITRE, size=21, color=OR_CLAIR), x=0.01, xanchor="left",
                   yref="container", y=1, yanchor="top", pad=dict(t=16)),
        paper_bgcolor=NUIT, plot_bgcolor=NUIT,
        colorway=SERIES,
        margin=dict(l=70, r=40, t=110, b=60),
        hoverlabel=dict(bgcolor=NUIT_2, bordercolor=OR, font=dict(family=POLICE, color=BLANC, size=13)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=CHAMPAGNE), orientation="h",
                    yanchor="bottom", y=1.0, xanchor="right", x=1),
        xaxis=dict(showgrid=False, linecolor="rgba(231,201,128,0.35)", tickcolor="rgba(231,201,128,0.35)",
                   ticks="outside", tickfont=dict(color=CHAMPAGNE), title=dict(font=dict(color=GRIS)),
                   zeroline=False),
        yaxis=dict(gridcolor="rgba(159,184,255,0.10)", griddash="dot", linecolor="rgba(0,0,0,0)",
                   tickfont=dict(color=CHAMPAGNE), title=dict(font=dict(color=GRIS)), zeroline=False),
        coloraxis=dict(colorscale=ECHELLE_NUIT_OR,
                       colorbar=dict(outlinewidth=0, tickfont=dict(color=CHAMPAGNE), thickness=12)),
        bargap=0.25,
    )
    t.data.bar = [go.Bar(marker=dict(line=dict(width=0)))]
    t.data.scatter = [go.Scatter(line=dict(width=3))]
    return t


pio.templates["team"] = _template()
pio.templates.default = "team"


def titre(texte: str, sous_titre: str | None = None) -> dict:
    """Titre doré + sous-titre champagne (utilisable dans update_layout(title=...))."""
    if sous_titre:
        texte = (f"{texte}<br><span style='font-size:13px;color:{CHAMPAGNE};"
                 f"font-family:{POLICE}'>{sous_titre}</span>")
    return dict(text=texte)


def fmt_eur(x: float, dec: int = 0) -> str:
    return f"{x:,.{dec}f} €".replace(",", " ").replace(".", ",")


def fmt_pct(x: float, dec: int = 2) -> str:
    return f"{100 * x:.{dec}f} %".replace(".", ",")


def pct_txt(v: float, dec: int = 2) -> str:
    """Valeur déjà en % (ex. 1.76) -> « 1,76 % »."""
    return f"{v:.{dec}f} %".replace(".", ",")


def fmt_nb(x: float, dec: int = 0) -> str:
    return f"{x:,.{dec}f}".replace(",", " ").replace(".", ",")


def configurer_notebook() -> str:
    """Rendu Plotly dans Jupyter : interactif + image PNG de secours (lisible sur GitHub) si kaleido est disponible."""
    try:
        pio.to_image(go.Figure(), format="png", width=10, height=10)
        pio.renderers.default = "plotly_mimetype+notebook_connected+png"
        pio.renderers["png"].width, pio.renderers["png"].height, pio.renderers["png"].scale = 1100, None, 1.5
    except Exception:  # pas de Chrome / kaleido : rendu interactif seul
        pio.renderers.default = "plotly_mimetype+notebook_connected"
    return pio.renderers.default
