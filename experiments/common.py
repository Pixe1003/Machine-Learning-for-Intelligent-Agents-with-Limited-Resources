"""Shared helpers for experiment runners."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from src.analysis.diagnostics import monitor_diagnostics
from src.analysis.heatmap import plot_dynamic_heatmap, plot_heatmap
from src.analysis.posterior_scatter import plot_posterior_scatter
from src.baselines.mk_random_walk import MkBaseline
from src.core.loss import report_evaluate
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
    "f_mean",
    "f_std",
    "ci95_low",
    "ci95_high",
    "restarts",
    "restart_losses_json",
    "wall_clock_s",
    "delta_acc",
    "delta_count",
    "delta_minimal",
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


def summarise_train_result(result: TrainResult) -> dict[str, object]:
    ci_low, ci_high = result.ci95
    return {
        "f_train_best": result.best_train_loss,
        "f_report": result.best_loss,
        "f_mean": result.mean_loss,
        "f_std": result.std_loss,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "restarts": len(result.restarts),
        "restart_losses_json": json.dumps(result.report_losses),
    }


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


def write_monitor_artifacts(
    *,
    monitor,
    phase: str,
    n: int,
    k: int,
    class_name: str,
    output_dir: Path,
    samples: int = 2048,
    generate: bool = True,
) -> dict[str, object]:
    diagnostics = monitor_diagnostics(monitor, n=n, samples=samples)
    heatmap_path = ""
    scatter_path = ""
    if generate:
        figure_dir = output_dir / "figures" / phase
        figure_dir.mkdir(parents=True, exist_ok=True)
        stem = f"n{n}_k{k}_{class_name}"
        heatmap_path = (figure_dir / f"{stem}_heatmap.png").resolve()
        scatter_path = (figure_dir / f"{stem}_posterior.png").resolve()
        H, T = monitor.transitions(n)
        if H.dim() == 3:
            plot_dynamic_heatmap(H, T, title=f"{class_name} heatmap (n={n}, k={k})", out_path=heatmap_path)
        else:
            plot_heatmap(H, T, title=f"{class_name} heatmap (n={n}, k={k})", out_path=heatmap_path)
        plot_posterior_scatter(monitor, n=n, out_path=scatter_path, M=samples)
    return {
        **diagnostics,
        "heatmap_path": str(heatmap_path),
        "posterior_scatter_path": str(scatter_path),
    }


def diagnostic_row(phase: str, n: int, k: int, class_name: str, metrics: dict[str, object]) -> dict[str, object]:
    row = {key: metrics.get(key) for key in DIAGNOSTIC_COLUMNS}
    row.update({"phase": phase, "n": n, "k": k, "class": class_name})
    return row


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
