"""Éléments communs à toutes les pages du dashboard (chargement, mise en forme, cache)."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

RACINE = Path(__file__).resolve().parents[1]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from src import theme as T  # noqa: E402
from src.data import charger_base, decouper  # noqa: E402

MODELES = RACINE / "models"
ASSETS = RACINE / "assets"


def configurer(titre: str = ""):
    """Style et barre latérale (la configuration de page est faite une fois dans Accueil.py)."""
    st.markdown(f"<style>{(ASSETS / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    with st.sidebar:
        st.image(str(ASSETS / "logo_team.png"), width="stretch")
        st.markdown("<div style='text-align:center;color:#E7C980;font-size:0.85rem;margin-top:-6px'>"
                    "Tarification non vie · Habitation<br><span style='color:#8FA3C7'>Master II EURIA</span></div>",
                    unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def logo_b64() -> str:
    return base64.b64encode((ASSETS / "logo_team.png").read_bytes()).decode()


def bandeau(titre: str, sous_titre: str):
    st.markdown(f"<div class='hero'><img src='data:image/png;base64,{logo_b64()}'/>"
                f"<div><div class='t'>{titre}</div><div class='s'>{sous_titre}</div></div></div>",
                unsafe_allow_html=True)


def kpis(cartes: list[tuple[str, str, str]]):
    html = "".join(f"<div class='kpi'><div class='lbl'>{l}</div><div class='val'>{v}</div>"
                   f"<div class='sub'>{s}</div></div>" for l, v, s in cartes)
    st.markdown(f"<div class='kpi-grid'>{html}</div>", unsafe_allow_html=True)


def encadre(texte: str, tag: str = "Lecture actuarielle"):
    st.markdown(f"<div class='insight'><span class='tag'>{tag}</span>{texte}</div>", unsafe_allow_html=True)


def pied():
    st.markdown("<div class='footer'>Think SMART, Solve SMARTer · Tarification non vie avec Python · "
                "GLM fréquence × coût · Master II EURIA</div>", unsafe_allow_html=True)


def graphe(fig, **kw):
    st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False}, **kw)


@st.cache_resource(show_spinner="Chargement de la base (1,1 million de lignes)…")
def base() -> pd.DataFrame:
    return charger_base()


@st.cache_resource(show_spinner=False)
def bases_apprentissage_test():
    bda, bdt, _ = decouper(base(), seed=2016)
    return bda, bdt


@st.cache_data(show_spinner=False)
def lire(nom: str) -> pd.DataFrame:
    return pd.read_parquet(MODELES / f"{nom}.parquet")


@st.cache_data(show_spinner=False)
def lire_json(nom: str) -> dict:
    return json.loads((MODELES / f"{nom}.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def geojson() -> dict:
    return json.loads((ASSETS / "departements.geojson").read_text(encoding="utf-8"))
