"""
Grouped bar chart of hard/count/minimal gaps across (n, k).

Delta^count is the key stochastic-counter gap. Plotting the three gaps side by
side makes the hierarchy story readable.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GAP_COLS = ("delta_hard", "delta_count", "delta_minimal")
GAP_COLORS = {
    "delta_hard": "#1f77b4",
    "delta_count": "#ff7f0e",
    "delta_minimal": "#d62728",
}
GAP_LABELS = {
    "delta_hard": r"$\Delta^{\mathrm{hard}}$",
    "delta_count": r"$\Delta^{\mathrm{count}}$",
    "delta_minimal": r"$\Delta^{\mathrm{minimal}}$",
}


def plot_class_gap_bars(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    normalise: bool = False,
    title: str | None = None,
) -> None:
    """
    One grouped bar per (n, k) configuration, with three bars per group.

    If `normalise=True`, gaps are divided by f_M1 (taken from the M1 row) to
    express them as a fraction of the best-static error.  This is usually the
    more readable version for the H1 story.
    """
    df = pd.read_csv(csv_path)
    if "delta_hard" not in df.columns and "delta_acc" in df.columns:
        df["delta_hard"] = df["delta_acc"]
    # one row per (n, k): take gaps from the M1 row (they are repeated across classes)
    piv = (
        df[df["class"] == "M1"]
          .sort_values(["n", "k"])[["n", "k", *GAP_COLS, "f_report"]]
          .reset_index(drop=True)
    )
    if piv.empty:
        raise ValueError(f"No M1 rows in {csv_path}; cannot extract class gaps.")

    if normalise:
        for col in GAP_COLS:
            piv[col] = piv[col] / piv["f_report"]

    groups = [f"n={int(n)}\nk={int(k)}" for n, k in zip(piv["n"], piv["k"])]
    x = np.arange(len(groups))
    width = 0.26

    fig, ax = plt.subplots(1, 1, figsize=(max(6, 1.2 * len(groups) + 2), 4))
    for i, col in enumerate(GAP_COLS):
        vals = piv[col].to_numpy()
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=GAP_LABELS[col], color=GAP_COLORS[col])
        # annotate small values so Delta^count ~ 0 is visible
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2,
                        v + (0.002 if not normalise else 0.002),
                        f"{v:.3g}",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("gap" + (r" / $f_{M1}$" if normalise else ""))
    ax.set_title(title or f"Class gaps ({Path(csv_path).stem})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
