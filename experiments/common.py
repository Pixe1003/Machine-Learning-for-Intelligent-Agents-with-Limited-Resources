"""Shared helpers for experiment runners.

This module centralises three concerns:

1. CSV / diagnostic schema (RESULT_COLUMNS, DIAGNOSTIC_COLUMNS).
2. Per-monitor diagnostics (tridiagonality, entropy, posterior monotonicity).
3. Figure generation.  We no longer emit one heatmap + scatter per class:
   instead a single consolidated figure is produced per (n, k) that shows
   all four (or five) monitor classes side by side.  The summary figures
   (error curves, class-gap bars, annealing traces, restart distributions)
   are emitted once at the end of each phase's main() by calling
   `write_summary_figures(csv_path, figure_dir)`.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

from src.analysis.anneal_trace_plot import plot_anneal_traces
from src.analysis.class_gap_bars import plot_class_gap_bars
from src.analysis.diagnostics import monitor_diagnostics
from src.analysis.error_curves import plot_error_curves_vs_k, plot_static_vs_dynamic
from src.analysis.heatmap import plot_class_comparison_heatmap
from src.analysis.monitor_bar_charts import plot_monitor_error_bars, plot_monitor_gap_bars
from src.analysis.posterior_scatter import plot_class_comparison_scatter
from src.analysis.restart_distribution import plot_restart_distribution, write_stability_summary
from src.baselines.mk_random_walk import MkBaseline
from src.core.loss import evaluate_hard, evaluate_soft, report_evaluate
from src.optim.anneal import anneal_trace
from src.optim.gradient import TrainResult

ANNEAL_TAUS = (1.0, 0.5, 0.1, 0.01, 0.001)
RESULT_COLUMNS = [
    "phase",
    "n",
    "k",
    "class",
    "f_train_best",
    "f_report",
    "f_report_soft",
    "f_report_hard",
    "f_mean",
    "f_std",
    "ci95_low",
    "ci95_high",
    "restarts",
    "restart_losses_json",
    "wall_clock_s",
    "delta_hard",
    "delta_acc",
    "delta_acc_soft",
    "delta_acc_hard",
    "delta_count",
    "delta_count_hard",
    "delta_minimal",
    "delta_minimal_hard",
    "delta_dynamic",
    "baseline_f",
    "baseline_gap",
    "ga_f",
    "ga_gap",
    "exhaustive_f",
    "exhaustive_gap",
    "anneal_loss_tau_1_0",
    "anneal_loss_tau_0_5",
    "anneal_loss_tau_0_1",
    "anneal_loss_tau_0_01",
    "anneal_loss_tau_0_001",
    "tridiagonality",
    "transition_entropy_mean",
    "transition_entropy_min",
    "transition_entropy_max",
    "posterior_monotonicity_mean",
    "posterior_monotonicity_min",
    "heatmap_path",
    "posterior_scatter_path",
]
DIAGNOSTIC_COLUMNS = [
    "phase",
    "n",
    "k",
    "class",
    "tridiagonality",
    "transition_entropy_mean",
    "transition_entropy_min",
    "transition_entropy_max",
    "posterior_monotonicity_mean",
    "posterior_monotonicity_min",
]


def empty_result_row() -> dict[str, object]:
    return {key: None for key in RESULT_COLUMNS}


def summarise_train_result(result: TrainResult, *, n: int | None = None) -> dict[str, object]:
    """Summarise a TrainResult into CSV columns.

    When ``n`` is supplied, the best monitor is additionally evaluated under
    both soft (continuous sigmoid) and hard (binary thresholded) acceptance
    semantics and written into ``f_report_soft`` / ``f_report_hard``.  This
    makes Δ^acc and other class-gap quantities comparable across the four
    monitor classes regardless of which eval mode was used as the training
    selection criterion.  Callers that do not need dual eval may omit ``n``
    (the two new columns will simply be ``None``).
    """
    ci_low, ci_high = result.ci95
    row: dict[str, object] = {
        "f_train_best": result.best_train_loss,
        "f_report": result.best_loss,
        "f_report_soft": None,
        "f_report_hard": None,
        "f_mean": result.mean_loss,
        "f_std": result.std_loss,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "restarts": len(result.restarts),
        "restart_losses_json": json.dumps(result.report_losses),
    }
    if n is not None:
        row["f_report_soft"] = evaluate_soft(result.best_monitor, n)
        row["f_report_hard"] = evaluate_hard(result.best_monitor, n)
    return row


def baseline_value(n: int, k: int) -> float:
    return report_evaluate(MkBaseline(k=k), n)


def annealing_columns(monitor, n: int, enabled: bool) -> dict[str, object]:
    cols = {f"anneal_loss_tau_{str(tau).replace('.', '_')}": None for tau in ANNEAL_TAUS}
    if not enabled:
        return cols
    trace = anneal_trace(monitor, n, taus=ANNEAL_TAUS)
    for tau, loss in zip(trace.taus, trace.losses):
        cols[f"anneal_loss_tau_{str(tau).replace('.', '_')}"] = loss
    return cols


# ---------------------------------------------------------------------------
# diagnostics + figure generation
# ---------------------------------------------------------------------------

def compute_monitor_metrics(
    *,
    monitor,
    phase: str,
    n: int,
    k: int,
    class_name: str,
    shared_figure: Path | None = None,
    samples: int = 2048,
) -> dict[str, object]:
    """Return the per-monitor metrics dict (no per-class figures).

    `shared_figure` is stamped into the heatmap_path / posterior_scatter_path
    columns so the CSV still points to a valid file.
    """
    diagnostics = monitor_diagnostics(monitor, n=n, samples=samples)
    path_str = str(shared_figure.resolve()) if shared_figure else ""
    return {
        **diagnostics,
        "heatmap_path": path_str,
        "posterior_scatter_path": path_str,
    }


def write_consolidated_figures(
    *,
    suite: Mapping[str, object],
    phase: str,
    n: int,
    k: int,
    output_dir: Path,
    samples: int = 2048,
    diagnostics: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Path]:
    """Write the two consolidated figures (heatmap + posterior scatter) that
    cover every monitor class for a single (n, k) setting."""
    figure_dir = output_dir / "figures" / phase
    figure_dir.mkdir(parents=True, exist_ok=True)

    # resolve each class's monitor to its best-loss instance if it's a TrainResult
    monitors: dict[str, object] = {}
    for name, obj in suite.items():
        monitors[name] = obj.best_monitor if hasattr(obj, "best_monitor") else obj

    heatmap_path = figure_dir / f"n{n}_k{k}_classes_heatmap.png"
    scatter_path = figure_dir / f"n{n}_k{k}_classes_posterior.png"

    plot_class_comparison_heatmap(
        monitors=monitors,
        n=n,
        k=k,
        out_path=heatmap_path,
        title=f"{phase}  n={n}, k={k}  -- H / T by class",
        diagnostics=diagnostics,
    )
    plot_class_comparison_scatter(
        monitors=monitors,
        n=n,
        out_path=scatter_path,
        M=min(samples * 2, 12_000),
        title=f"{phase}  n={n}  -- state-posterior by class",
    )
    return {"heatmap": heatmap_path, "posterior_scatter": scatter_path}


def write_summary_figures(
    csv_path: str | Path,
    figure_dir: str | Path,
    *,
    phase: str,
    include_dynamic: bool = False,
) -> dict[str, Path]:
    """After a phase finishes, emit the four cross-config summary figures.

    Any plot that fails (e.g. empty data) is logged and skipped -- we do not
    want a single plotting glitch to break the whole pipeline.
    """
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    targets = [
        (
            "error_curves",
            figure_dir / f"{phase}_error_curves.png",
            lambda p: plot_error_curves_vs_k(csv_path, p, title=f"{phase}: error curves"),
        ),
        (
            "class_gap_bars",
            figure_dir / f"{phase}_class_gap_bars.png",
            lambda p: plot_class_gap_bars(csv_path, p, normalise=True,
                                          title=f"{phase}: class gaps / f_M1"),
        ),
        (
            "monitor_error_bars",
            figure_dir / f"{phase}_monitor_error_bars.png",
            lambda p: plot_monitor_error_bars(csv_path, p,
                                              title=f"{phase}: monitor errors"),
        ),
        (
            "monitor_gap_bars",
            figure_dir / f"{phase}_monitor_gap_bars.png",
            lambda p: plot_monitor_gap_bars(csv_path, p,
                                            title=f"{phase}: relative gaps from M1"),
        ),
        (
            "anneal_traces",
            figure_dir / f"{phase}_anneal_traces.png",
            lambda p: plot_anneal_traces(csv_path, p,
                                         title=f"{phase}: annealing trace (M1-hard)"),
        ),
        (
            "restart_distribution",
            figure_dir / f"{phase}_restart_distribution.png",
            lambda p: plot_restart_distribution(csv_path, p,
                                                title=f"{phase}: restart spread"),
        ),
        (
            "stability_summary",
            figure_dir / f"{phase}_stability_summary.csv",
            lambda p: write_stability_summary(csv_path, p),
        ),
    ]
    if include_dynamic:
        targets.append(
            (
                "static_vs_dynamic",
                figure_dir / f"{phase}_static_vs_dynamic.png",
                lambda p: plot_static_vs_dynamic(csv_path, p,
                                                 title=f"{phase}: static vs dynamic"),
            )
        )

    for name, out_path, fn in targets:
        try:
            fn(out_path)
            outputs[name] = out_path
        except Exception as exc:  # noqa: BLE001 -- keep pipeline robust
            print(f"  [summary:{name}] skipped ({exc})")
    return outputs


def diagnostic_row(phase: str, n: int, k: int, class_name: str,
                   metrics: dict[str, object]) -> dict[str, object]:
    row = {key: metrics.get(key) for key in DIAGNOSTIC_COLUMNS}
    row.update({"phase": phase, "n": n, "k": k, "class": class_name})
    return row


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Back-compat shim so the old call-sites still work during incremental edits.
# ---------------------------------------------------------------------------

def write_monitor_artifacts(
    *,
    monitor,
    phase: str,
    n: int,
    k: int,
    class_name: str,
    output_dir: Path,
    samples: int = 2048,
    generate: bool = False,
    shared_figure: Path | None = None,
) -> dict[str, object]:
    """Deprecated: kept only so older experiment scripts still import.

    New experiment runners should call `compute_monitor_metrics` for the row
    and `write_consolidated_figures` once per (n, k).  When `generate=True`
    this shim does nothing extra -- figures are handled at the (n, k) level.
    """
    return compute_monitor_metrics(
        monitor=monitor,
        phase=phase,
        n=n,
        k=k,
        class_name=class_name,
        shared_figure=shared_figure,
        samples=samples,
    )
