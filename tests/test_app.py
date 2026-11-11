"""Tests de fumée de la démo Streamlit (streamlit.testing.AppTest).

Chaque page est exécutée de bout en bout ; le test vérifie qu'aucune exception
n'est levée et que les éléments clés sont affichés. Les tests sont ignorés si
les modèles (``python -m src.train``) ou les jeux découpés sont absents.
"""

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from streamlit.testing.v1 import AppTest  # noqa: E402

import commun  # noqa: E402

needs_models = pytest.mark.skipif(not commun.models_available(), reason="modèles non entraînés")
needs_data = pytest.mark.skipif(not commun.splits_available(), reason="jeux découpés absents")


def run_page(name: str) -> AppTest:
    at = AppTest.from_file(str(APP_DIR / "vues" / f"{name}.py"), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    return at


def test_accueil_via_navigation():
    """L'accueil (page par défaut) est lancé via le point d'entrée multipage."""
    at = AppTest.from_file(str(APP_DIR / "streamlit_app.py"), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    assert "Détection de fraude" in at.title[0].value


def metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


@needs_models
def test_analyser_presets_give_expected_decisions():
    at = run_page("analyser")
    # Exemple par défaut : fraude typique (compte vidé de nuit) -> alerte des deux modèles
    assert len(at.error) == 2 and not at.success
    # Retrait légitime en journée -> aucune alerte
    at.selectbox[0].select("Retrait légitime en journée").run()
    assert not at.exception
    assert len(at.success) == 2 and not at.error


@needs_models
@needs_data
def test_simuler_reacts_to_alert_cost():
    at = run_page("simuler")
    alerts_default = int(metric(at, "Alertes levées").replace(" ", ""))
    at.select_slider[0].set_value(50_000).run()
    assert not at.exception
    alerts_costly = int(metric(at, "Alertes levées").replace(" ", ""))
    assert alerts_costly < alerts_default  # fausse alerte plus chère -> moins d'alertes


@needs_models
def test_a_propos():
    at = run_page("a_propos")
    assert any("Limites" in h.value for h in at.header)
