"""Bar-chart summaries for static monitor comparisons.

These plots are intended for small grids where line charts overlap heavily.
They make the two main Phase 1 messages easier to read:

* absolute report-time error per monitor class;
* relative gap from M1 per monitor class.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_CLASSES = ("M1", "M1-hard", "M2", "M3")
CLASS_COLORS = {
    "M1": "#1f77b4",
    "M1-hard": "#9ecae1",
    "M2": "#ff7f0e",
    "M3": "#d62728",
}


def _load_static_rows(csv_path: str | Path, classes: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["class"].isin(list(classes))].copy()
    for col in ("n", "k", "f_report", "baseline_f"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if df.empty:
        raise ValueError(f"No static monitor rows in {csv_path}")
    return df.sort_values(["n", "k", "class"])


def _group_labels(df: pd.DataFrame) -> list[str]:
    groups = df[["n", "k"]].drop_duplicates().sort_values(["n", "k"])
    return [f"n={int(row.n)}\nk={int(row.k)}" for row in groups.itertuples(index=False)]


def plot_monitor_error_bars(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    classes: Iterable[str] = DEFAULT_CLASSES,
    include_baseline: bool = True,
    title: str | None = None,
) -> None:
    """Grouped bars of absolute report-time error for each monitor class."""
    df = _load_static_rows(csv_path, classes)
    pivot = df.pivot_table(index=["n", "k"], columns="class", values="f_report", aggfunc="min")
    pivot = pivot.sort_index()

    labels = _group_labels(df)
    x = np.arange(len(pivot))
    classes = tuple(cls for cls in classes if cls in pivot.columns)
    width = min(0.18, 0.72 / max(1, len(classes)))

    fig, ax = plt.subplots(1, 1, figsize=(max(6.5, 1.4 * len(labels) + 2), 4.4))
    offsets = np.linspace(-width * (len(classes) - 1) / 2, width * (len(classes) - 1) / 2, len(classes))
    for offset, cls in zip(offsets, classes):
        vals = pivot[cls].to_numpy()
        ax.bar(x + offset, vals, width, label=cls, color=CLASS_COLORS.get(cls))

    if include_baseline and "baseline_f" in df.columns:
        baseline = (
            df[df["class"] == "M1"]
            .drop_duplicates(["n", "k"])
            .sort_values(["n", "k"])["baseline_f"]
            .to_numpy()
        )
        if len(baseline) == len(x) and np.isfinite(baseline).any():
            ax.scatter(x, baseline, marker="_", s=220, linewidths=2.0, color="#444444",
                       label="baseline")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$f_{\mathrm{report}}$")
    ax.set_title(title or f"Monitor error bars ({Path(csv_path).stem})")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_monitor_gap_bars(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    classes: Iterable[str] = ("M1-hard", "M2", "M3"),
    title: str | None = None,
) -> None:
    """Grouped bars of relative gap `(f_class - f_M1) / f_M1`.

    This is the most readable Phase 1 plot when M1, M1-hard, and M2 overlap in the
    absolute-error chart.
    """
    df = _load_static_rows(csv_path, ("M1", *tuple(classes)))
    pivot = df.pivot_table(index=["n", "k"], columns="class", values="f_report", aggfunc="min")
    pivot = pivot.sort_index()
    if "M1" not in pivot.columns:
        raise ValueError(f"No M1 rows in {csv_path}; cannot compute relative gaps")

    labels = _group_labels(df)
    x = np.arange(len(pivot))
    classes = tuple(cls for cls in classes if cls in pivot.columns)
    width = min(0.22, 0.72 / max(1, len(classes)))

    fig, ax = plt.subplots(1, 1, figsize=(max(6.5, 1.4 * len(labels) + 2), 4.4))
    offsets = np.linspace(-width * (len(classes) - 1) / 2, width * (len(classes) - 1) / 2, len(classes))
    m1 = pivot["M1"].replace(0.0, np.nan)
    for offset, cls in zip(offsets, classes):
        vals = ((pivot[cls] - pivot["M1"]) / m1).to_numpy()
        bars = ax.bar(x + offset, vals, width, label=f"{cls} - M1",
                      color=CLASS_COLORS.get(cls))
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + max(0.002, abs(val) * 0.03),
                    f"{val:.2%}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"relative gap from M1")
    ax.set_title(title or f"Monitor gaps from M1 ({Path(csv_path).stem})")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
