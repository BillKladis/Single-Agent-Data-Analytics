"""
Centralised visualization theme for consistent, publication-quality charts.

All tools and evaluation plots draw from the same palette and helpers so the
project has a single, professional visual identity.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# A calm, professional palette (color-blind friendly base).
PALETTE = {
    "primary": "#2F6DB5",     # blue
    "accent": "#E1812C",      # orange
    "negative": "#C44E52",    # red
    "positive": "#55A868",    # green
    "neutral": "#8C8C8C",     # grey
    "muted": "#B9C7D9",       # light blue
}

SEQUENCE = ["#2F6DB5", "#E1812C", "#55A868", "#C44E52", "#8172B3", "#937860"]


def apply_theme() -> None:
    sns.set_theme(style="whitegrid", font_scale=0.95)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelcolor": "#222222",
            "axes.prop_cycle": plt.cycler(color=SEQUENCE),
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.7,
            "font.family": "DejaVu Sans",
        }
    )


def thousands(ax, axis: str = "x") -> None:
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    if axis == "x":
        ax.xaxis.set_major_formatter(fmt)
    else:
        ax.yaxis.set_major_formatter(fmt)


def annotate_barh(ax, values, fmt: str = "{:,.0f}", pad: float = 0.0) -> None:
    """Place value labels at the end of horizontal bars."""
    span = max(abs(min(values)), abs(max(values))) or 1.0
    for i, v in enumerate(values):
        offset = span * 0.01 + pad
        ax.text(
            v + (offset if v >= 0 else -offset),
            i,
            fmt.format(v),
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8.5,
            color="#333333",
        )


def bar_colors(values, positive=None, negative=None):
    positive = positive or PALETTE["primary"]
    negative = negative or PALETTE["negative"]
    return [negative if v < 0 else positive for v in values]


apply_theme()
