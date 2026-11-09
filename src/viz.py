"""Style graphique commun à tous les notebooks.

Palette : bleu / orange, paire validée pour la lisibilité par les personnes
daltoniennes. Les couleurs encodent toujours la même chose dans tout le projet :
bleu = transaction légitime, orange = fraude.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.config import FIGURES_DIR

COLOR_LEGIT = "#2a78d6"
COLOR_FRAUD = "#eb6834"
CLASS_COLORS = {0: COLOR_LEGIT, 1: COLOR_FRAUD}
CLASS_LABELS = {0: "Légitime", 1: "Fraude"}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SEQUENTIAL_CMAP = "Blues"


def set_style() -> None:
    """Applique le style graphique du projet (axes et grille discrets)."""
    plt.rcParams.update(
        {
            "figure.figsize": (9, 4.5),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": GRID_COLOR,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "text.color": TEXT_PRIMARY,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def save_fig(fig: plt.Figure, name: str, folder: Path = FIGURES_DIR) -> Path:
    """Enregistre une figure en PNG dans reports/figures/."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    return path
