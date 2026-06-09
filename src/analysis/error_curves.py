"""
Summary error-curve plots generated from phase*.csv.

Two entry points:
    plot_error_curves_vs_k   -- main performance figure: f_report vs k,
                                 one line per monitor class, with CI95 error bars.
                                 One subplot per n.
    plot_static_vs_dynamic   -- static-best vs dynamic with Delta fill, for
                                 phase2/phase3 CSVs that include a Dynamic class.

Both functions accept a CSV path (produced by experiments/common.py) and an
output image path.  They are intentionally self-contained so they can also be
run standalone as a post-hoc analysis step.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STATIC_CLASSES = ("M1", "M1-hard", "M2", "M3")
DYNAMIC_CLASSES = ("Dynamic", "DynamicShared")

# One fixed palette / marker per class so colours stay consistent across phases.
CLASS_STYLE = {
    "M1": dict(color="#1f77b4", marker="o"),
    "M1-hard": dict(color="#9ecae1", marker="s"),
    "M2": dict(color="#ff7f0e", marker="^"),
    "M3": dict(color="#d62728", marker="D"),
    "Dynamic": dict(color="#2ca02c", marker="x"),
    "DynamicShared": dict(color="#17becf", marker="P"),
}


def _load(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # coerce numerics we care about
    for col in ("n", "k", "f_report", "ci95_low", "ci95_high", "baseline_f"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def plot_error_curves_vs_k(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    classes: Iterable[str] = STATIC_CLASSES,
    include_baseline: bool = True,
    title: str | None = None,
) -> None:
    """f_report vs k for each class; one subplot per n (horizontal layout)."""
    df = _load(csv_path)
    ns = sorted(df["n"].dropna().unique().astype(int).tolist())
    if not ns:
        raise ValueError(f"No usable rows in {csv_path}")

    fig, axes = plt.subplots(
        1, len(ns), figsize=(4.2 * len(ns), 3.8), squeeze=False, sharey=True
    )
    for idx, n in enumerate(ns):
        ax = axes[0, idx]
        sub = df[df["n"] == n]
        for cls in classes:
            cls_rows = sub[sub["class"] == cls].sort_values("k")
            if cls_rows.empty:
                continue
            style = CLASS_STYLE.get(cls, {})
            ks = cls_rows["k"].to_numpy()
            fs = cls_rows["f_report"].to_numpy()
            lo = cls_rows.get("ci95_low")
            hi = cls_rows.get("ci95_high")
            yerr = None
            if lo is not None and hi is not None and not lo.isna().all():
                yerr = [
                    (fs - lo.to_numpy()).clip(min=0.0),
                    (hi.to_numpy() - fs).clip(min=0.0),
                ]
            ax.errorbar(
                ks, fs,
                yerr=yerr,
                label=cls,
                capsize=3, linewidth=1.5, markersize=5,
                **style,
            )
        if include_baseline:
            base = sub[sub["class"] == "M1"].sort_values("k")
            if "baseline_f" in base.columns and not base["baseline_f"].isna().all():
                ax.plot(base["k"], base["baseline_f"],
                        linestyle="--", color="gray", label="Kontorovich M(k)",
                        linewidth=1)

        ax.set_xlabel("k")
        if idx == 0:
            ax.set_ylabel(r"$f(n, k)$ (report-time)")
        ax.set_title(f"n = {n}", fontsize=10)
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.02),
               ncol=min(len(labels), 6), frameon=False, fontsize=9)
    fig.suptitle(title or f"Error curves ({Path(csv_path).stem})",
                 fontsize=11, y=1.08)
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_static_vs_dynamic(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    dynamic_classes: Iterable[str] = DYNAMIC_CLASSES,
    title: str | None = None,
) -> None:
    """Static-best vs dynamic line plot with fill_between marking Delta(n,k)."""
    df = _load(csv_path)
    static = df[df["class"].isin(STATIC_CLASSES)]
    dyn = df[df["class"].isin(list(dynamic_classes))]
    if dyn.empty:
        raise ValueError(f"{csv_path} contains no dynamic monitor rows")

    ns = sorted(df["n"].dropna().unique().astype(int).tolist())
    fig, axes = plt.subplots(
        1, len(ns), figsize=(4.2 * len(ns), 3.8), squeeze=False, sharey=True
    )
    for idx, n in enumerate(ns):
        ax = axes[0, idx]
        sub_s = static[static["n"] == n]
        sub_d = dyn[dyn["n"] == n]
        # best static per k
        best_static = (
            sub_s.groupby("k", as_index=False)["f_report"].min()
                 .sort_values("k")
        )
        ax.plot(best_static["k"], best_static["f_report"],
                color="#1f77b4", marker="o", linewidth=1.8, label="best static")

        dyn_curve = sub_d.groupby("k", as_index=False)["f_report"].min().sort_values("k")
        ax.plot(dyn_curve["k"], dyn_curve["f_report"],
                color="#2ca02c", marker="x", linewidth=1.8, label="dynamic")

        # Delta fill (only on overlapping ks)
        merged = best_static.merge(
            dyn_curve, on="k", suffixes=("_static", "_dyn")
        )
        if not merged.empty:
            ax.fill_between(
                merged["k"],
                merged["f_report_dyn"],
                merged["f_report_static"],
                color="#2ca02c", alpha=0.15,
                label=r"$\Delta(n, k)$",
            )
        ax.set_xlabel("k")
        if idx == 0:
            ax.set_ylabel(r"$f(n, k)$")
        ax.set_title(f"n = {n}", fontsize=10)
        ax.grid(alpha=0.25)

    axes[0, 0].legend(loc="best", fontsize=9)
    fig.suptitle(title or f"Static vs dynamic ({Path(csv_path).stem})",
                 fontsize=11)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
