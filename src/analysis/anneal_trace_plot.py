"""
Annealing-trace plot for Hypothesis H2.

Reads the five `anneal_loss_tau_*` columns written by `experiments.common`
and draws one line per (n, k) showing loss as tau shrinks.  A flat trace is
evidence for a deterministic optimum; an upward trace as tau -> 0 is evidence
that randomised transitions are genuinely required.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# tau column suffix -> numeric value
_SUFFIX_RE = re.compile(r"anneal_loss_tau_(\d+)_(\d+)")


def _tau_columns(df: pd.DataFrame) -> list[tuple[float, str]]:
    out = []
    for col in df.columns:
        m = _SUFFIX_RE.fullmatch(col)
        if m:
            whole, frac = m.groups()
            tau = float(f"{whole}.{frac}")
            out.append((tau, col))
    return sorted(out, key=lambda x: -x[0])  # largest tau first


def plot_anneal_traces(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    class_name: str = "M1-hard",
    title: str | None = None,
) -> None:
    """One line per (n, k), x = tau (log scale), y = annealed loss."""
    df = pd.read_csv(csv_path)
    df = df[df["class"] == class_name]
    taus_cols = _tau_columns(df)
    if not taus_cols:
        raise ValueError(f"No anneal_loss_tau_* columns in {csv_path}")

    taus = np.array([t for t, _ in taus_cols])
    cols = [c for _, c in taus_cols]

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    cmap = plt.get_cmap("viridis")
    rows = df.sort_values(["n", "k"]).dropna(subset=cols, how="all")
    if rows.empty:
        raise ValueError(f"{csv_path} has no anneal_loss_tau_* values for class {class_name}")
    n_rows = len(rows)
    for i, (_, row) in enumerate(rows.iterrows()):
        vals = row[cols].to_numpy(dtype=float)
        ax.plot(
            taus, vals,
            marker="o", linewidth=1.4,
            color=cmap(i / max(1, n_rows - 1)),
            label=f"n={int(row['n'])}, k={int(row['k'])}",
        )

    ax.set_xscale("log")
    ax.invert_xaxis()  # anneal direction: 1.0 -> 0.001
    ax.set_xlabel(r"softmax temperature $\tau$")
    ax.set_ylabel(f"{class_name} loss under annealed softmax")
    ax.set_title(title or f"Annealing trace (H2) -- {class_name}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
