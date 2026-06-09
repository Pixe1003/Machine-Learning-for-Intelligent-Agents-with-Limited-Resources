"""
Restart-loss distribution plot.

Reads the JSON list in `restart_losses_json` and draws a box / strip plot
per (n, k, class).  This is the clearest visual evidence for convergence
reliability and, for the Phase 1 warm-start chain, for the zero-variance
observation.
"""
from __future__ import annotations

import json
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_CLASSES = ("M1", "M1-hard", "M2", "M3")


def _parse_losses(cell: object) -> list[float]:
    if isinstance(cell, list):
        return [float(x) for x in cell]
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        data = json.loads(cell)
    except json.JSONDecodeError:
        return []
    return [float(x) for x in data] if isinstance(data, list) else []


def plot_restart_distribution(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
    title: str | None = None,
) -> None:
    df = pd.read_csv(csv_path)
    df = df[df["class"].isin(classes)].copy()
    df["losses"] = df["restart_losses_json"].apply(_parse_losses)

    rows = df[df["losses"].map(len) > 0].sort_values(["n", "k", "class"])
    if rows.empty:
        raise ValueError(f"No restart_losses_json data in {csv_path}")

    labels = [f"{r['class']}\nn={int(r['n'])}\nk={int(r['k'])}"
              for _, r in rows.iterrows()]
    data = [r["losses"] for _, r in rows.iterrows()]

    fig, ax = plt.subplots(1, 1, figsize=(max(6, 0.6 * len(labels) + 2), 4.5))
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False,
                    patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor="#9ecae1", alpha=0.6)

    # overlay individual restarts as thin strip
    for i, d in enumerate(data, start=1):
        jitter = (np.random.default_rng(i).random(len(d)) - 0.5) * 0.15
        ax.scatter(np.full(len(d), i) + jitter, d,
                   s=8, alpha=0.7, color="#08306b")

    ax.set_ylabel(r"restart $f_{\mathrm{report}}$")
    ax.set_title(title or f"Restart distribution ({Path(csv_path).stem})")
    ax.grid(alpha=0.25, axis="y")
    plt.setp(ax.get_xticklabels(), fontsize=8)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_stability_summary(csv_path: str | Path, out_path: str | Path) -> None:
    """Write a compact restart-stability table for reporting.

    The table keeps one row per `(n, k, class)` and records best/mean/std plus
    the spread of individual restart report losses.
    """
    df = pd.read_csv(csv_path)
    rows: list[dict[str, object]] = []
    for _, row in df.sort_values(["n", "k", "class"]).iterrows():
        losses = _parse_losses(row.get("restart_losses_json"))
        rows.append(
            {
                "n": int(row["n"]),
                "k": int(row["k"]),
                "class": row["class"],
                "best": float(row["f_report"]),
                "mean": float(row["f_mean"]) if "f_mean" in row and pd.notna(row["f_mean"]) else None,
                "std": float(row["f_std"]) if "f_std" in row and pd.notna(row["f_std"]) else None,
                "restart_min": min(losses) if losses else None,
                "restart_max": max(losses) if losses else None,
                "restart_spread": (max(losses) - min(losses)) if losses else None,
                "restart_count": len(losses),
            }
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "n",
                "k",
                "class",
                "best",
                "mean",
                "std",
                "restart_min",
                "restart_max",
                "restart_spread",
                "restart_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
