"""Static figures embedded in the HTML report.

Colour handling follows one rule: fuels are an identity (categorical) scale,
demand magnitude is a sequential scale. The categorical slots are used in a
fixed order and capped at eight - a ninth fuel folds into "Other" rather than
inventing a new hue, which is what keeps the palette colour-blind safe.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .logging_utils import get_logger  # noqa: E402

log = get_logger("figures")

# Validated categorical slots, used strictly in this order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#8a8981"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#76756e"
GRID = "#e6e5e0"

MAX_SERIES = len(SERIES)


def fuel_colours(fuels: List[str]) -> Dict[str, str]:
    """Fixed fuel -> colour assignment; anything past slot 8 becomes Other."""
    mapping: Dict[str, str] = {}
    for i, fuel in enumerate(fuels):
        mapping[fuel] = SERIES[i] if i < MAX_SERIES else OTHER
    mapping.setdefault("Other", OTHER)
    return mapping


def top_fuels(generation: pd.DataFrame, value_col: str = "capacity_mw",
              limit: int = MAX_SERIES) -> List[str]:
    ranked = (generation.groupby("fueltype")[value_col].sum()
              .sort_values(ascending=False).index.tolist())
    if len(ranked) <= limit:
        return ranked
    return ranked[:limit - 1] + ["Other"]


def _fold(generation: pd.DataFrame, keep: List[str]) -> pd.DataFrame:
    frame = generation.copy()
    frame["fueltype"] = frame["fueltype"].where(frame["fueltype"].isin(keep), "Other")
    return frame


def _style(ax) -> None:
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1.0, alpha=0.9)
    ax.set_axisbelow(True)


def _save(fig, path: Optional[Path]) -> str:
    """Write the figure (optional) and return it as a base64 data URI."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight",
                facecolor="none", transparent=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format="png", dpi=160, bbox_inches="tight",
                    facecolor="none", transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------


def capacity_by_country(generation: pd.DataFrame, path: Optional[Path] = None) -> str:
    """Stacked installed capacity per country, stacked by fuel."""
    keep = top_fuels(generation)
    frame = _fold(generation, keep)
    pivot = (frame.pivot_table(index="country", columns="fueltype",
                               values="capacity_mw", aggfunc="sum")
             .fillna(0.0))
    pivot = pivot.reindex(columns=[f for f in keep if f in pivot.columns])
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    colours = fuel_colours(keep)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))

    for fuel in pivot.columns:
        values = pivot[fuel].values
        # 2px surface-coloured edge keeps neighbouring segments legible.
        ax.bar(x, values, bottom=bottom, width=0.62, label=fuel,
               color=colours[fuel], edgecolor=SURFACE, linewidth=1.4)
        bottom += values

    totals = pivot.sum(axis=1).values
    for xi, total in zip(x, totals):
        ax.text(xi, total + totals.max() * 0.02, f"{total:,.0f}",
                ha="center", va="bottom", fontsize=9.5, color=TEXT_PRIMARY, fontweight="600")

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=10, color=TEXT_PRIMARY)
    ax.set_ylabel("Installed capacity (MW)", fontsize=9.5, color=TEXT_SECONDARY)
    ax.set_ylim(0, totals.max() * 1.12)
    _style(ax)

    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY,
              loc="upper right", ncol=2, handlelength=1.1, handleheight=1.1)
    return _save(fig, path)


def capacity_trajectory(generation: pd.DataFrame, years: List[int],
                        path: Optional[Path] = None) -> Optional[str]:
    """Capacity per fuel across the study years."""
    columns = [str(y) for y in years if str(y) in generation.columns]
    if len(columns) < 2:
        return None

    keep = top_fuels(generation)
    frame = _fold(generation, keep)
    series = frame.groupby("fueltype")[columns].sum()
    series = series.reindex([f for f in keep if f in series.index])
    colours = fuel_colours(keep)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = [int(c) for c in columns]

    for fuel, row in series.iterrows():
        ax.plot(x, row.values, color=colours[fuel], linewidth=2.0,
                marker="o", markersize=5.5, markeredgecolor=SURFACE,
                markeredgewidth=1.4, label=fuel, zorder=3)

    ceiling = float(series[columns].values.max()) * 1.12
    ax.set_ylim(0, ceiling)

    # Direct labels for the four largest fuels, nudged apart where their end
    # values sit too close to read; the legend below covers the rest.
    ranked = series[columns[-1]].sort_values(ascending=False).head(4)
    placed: List[float] = []
    gap = ceiling * 0.05
    for fuel, value in ranked.items():
        y = float(value)
        for taken in sorted(placed):
            if abs(y - taken) < gap:
                y = taken + gap
        placed.append(y)
        ax.annotate(f" {fuel}", xy=(x[-1], value), xytext=(x[-1], y),
                    color=TEXT_SECONDARY, fontsize=9, va="center", ha="left")

    ax.set_xticks(x)
    ax.set_xticklabels(columns, fontsize=9.5, color=TEXT_PRIMARY)
    ax.set_ylabel("Capacity online (MW)", fontsize=9.5, color=TEXT_SECONDARY)
    ax.set_xlim(x[0] - 0.4, x[-1] + (x[-1] - x[0]) * 0.24)
    _style(ax)
    # Legend below the plot so it can never sit on top of the series.
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY,
              loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=min(4, len(series)), handlelength=1.6, columnspacing=1.6)
    return _save(fig, path)


def demand_by_region(demand: pd.DataFrame, path: Optional[Path] = None,
                     limit: int = 15) -> str:
    """Largest regions by share of national demand (single sequential hue)."""
    frame = demand.sort_values("load_share", ascending=False).head(limit)
    frame = frame.iloc[::-1]

    fig, ax = plt.subplots(figsize=(7.6, max(3.0, 0.30 * len(frame) + 1.0)))
    y = np.arange(len(frame))
    values = frame["load_share"].values * 100

    # Sequential ramp: one hue, light to dark, keyed to magnitude.
    ramp = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "share", ["#bcd8f7", "#2a78d6", "#123a68"]
    )
    norm = plt.matplotlib.colors.Normalize(vmin=0, vmax=values.max() or 1)

    bars = ax.barh(y, values, height=0.62, color=[ramp(norm(v)) for v in values])
    for bar in bars:
        bar.set_capstyle("round")

    for yi, value in zip(y, values):
        ax.text(value + values.max() * 0.015, yi, f"{value:.1f}%",
                va="center", fontsize=9, color=TEXT_SECONDARY)

    ax.set_yticks(y)
    ax.set_yticklabels(frame["region"], fontsize=9, color=TEXT_PRIMARY)
    ax.set_xlabel("Share of national electricity demand (%)",
                  fontsize=9.5, color=TEXT_SECONDARY)
    ax.set_xlim(0, values.max() * 1.14)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=1.0)
    return _save(fig, path)


def capacity_by_region(generation: pd.DataFrame, path: Optional[Path] = None,
                       limit: int = 15) -> str:
    """Largest regions by installed capacity, stacked by fuel."""
    keep = top_fuels(generation)
    frame = _fold(generation, keep)
    pivot = (frame.pivot_table(index="region", columns="fueltype",
                               values="capacity_mw", aggfunc="sum").fillna(0.0))
    pivot = pivot.reindex(columns=[f for f in keep if f in pivot.columns])
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).head(limit).index]
    pivot = pivot.iloc[::-1]

    colours = fuel_colours(keep)
    fig, ax = plt.subplots(figsize=(7.6, max(3.0, 0.30 * len(pivot) + 1.0)))
    y = np.arange(len(pivot))
    left = np.zeros(len(pivot))

    for fuel in pivot.columns:
        values = pivot[fuel].values
        ax.barh(y, values, left=left, height=0.62, label=fuel,
                color=colours[fuel], edgecolor=SURFACE, linewidth=1.4)
        left += values

    for yi, total in zip(y, left):
        ax.text(total + left.max() * 0.015, yi, f"{total:,.0f}",
                va="center", fontsize=9, color=TEXT_SECONDARY)

    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index, fontsize=9, color=TEXT_PRIMARY)
    ax.set_xlabel("Installed capacity (MW)", fontsize=9.5, color=TEXT_SECONDARY)
    ax.set_xlim(0, left.max() * 1.16)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=1.0)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY,
              loc="lower right", ncol=2, handlelength=1.1, handleheight=1.1)
    return _save(fig, path)


def build_all(generation: pd.DataFrame, demand: pd.DataFrame, years: List[int],
              figure_dir: Optional[Path]) -> Dict[str, str]:
    """Render every report figure; returns {name: data URI}."""
    figures: Dict[str, str] = {}

    def _path(name: str) -> Optional[Path]:
        return figure_dir / f"{name}.png" if figure_dir is not None else None

    try:
        figures["capacity_by_country"] = capacity_by_country(generation, _path("capacity_by_country"))
        figures["capacity_by_region"] = capacity_by_region(generation, _path("capacity_by_region"))
        figures["demand_by_region"] = demand_by_region(demand, _path("demand_by_region"))
        trajectory = capacity_trajectory(generation, years, _path("capacity_trajectory"))
        if trajectory:
            figures["capacity_trajectory"] = trajectory
    except Exception as exc:  # pragma: no cover - a bad figure must not kill a run
        log.warning("Figure generation failed: %s", exc)

    log.info("Rendered %d figures", len(figures))
    return figures
