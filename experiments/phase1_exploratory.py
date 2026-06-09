"""
Phase 1 exploratory runs.

Defaults follow the proposal's exploratory grid, but the entrypoint is now
parameterised so tests and smoke runs can execute a tiny slice of the pipeline.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiments.common import (
    RESULT_COLUMNS,
    annealing_columns,
    baseline_value,
    compute_monitor_metrics,
    diagnostic_row,
    empty_result_row,
    summarise_train_result,
    write_consolidated_figures,
    write_csv,
    write_summary_figures,
)
from src.analysis.class_gaps import ClassGaps
from src.core.loss import evaluate_hard, evaluate_soft, report_evaluate
from src.monitors import M1HardMonitor, M1Monitor, M2Monitor, M3Monitor
from src.optim.exhaustive import exhaustive_search
from src.optim.genetic import run_ga as run_ga_search
from src.optim.gradient import train_with_restarts

DEFAULT_OUTPUT_DIR = Path("results")
N_VALUES = [11, 21]
# k=5 added per supervisor's request to expose class-gap differences that
# k=2,3 can mask.  Boundary check: f(n,k)=0 only when k ≥ ⌈n/2⌉+1, so
# (11,5) needs k ≥ 7 and (21,5) needs k ≥ 12 to be trivial — both safe.
K_VALUES = [2, 3, 5]
RESTARTS = 20
STEPS = 3000
# Mixed cold/warm restart strategy: every COLD_INTERVAL-th restart is cold
# (random init only, no warm-start cascade). The remaining restarts apply the
# warm-start *and* a Gaussian perturbation so they explore around the seed
# basin instead of all collapsing onto the warm-start fixed point.
COLD_INTERVAL = 4
PERTURB_SCALE = 0.1
BASE_SEED = 12345
EXHAUSTIVE_CASES = [(11, 2)]
GROUND_TRUTH_CASES = [(5, 2)]
GA_CASES = ((21, 3),)
GROUND_TRUTH_COLUMNS = [
    "n",
    "k",
    "class",
    "gradient_f",
    "gradient_mean",
    "gradient_std",
    "restarts",
    "steps",
    "exhaustive_f",
    "absolute_gap",
    "relative_gap",
    "gap_threshold",
    "passed",
    "action",
]


def _perturb_in_place(monitor, scale: float, seed: int) -> None:
    """Add a deterministic Gaussian perturbation to every parameter in-place.

    Used to break the degeneracy where a warm-start would otherwise overwrite
    the seed-specific random initialisation, collapsing all restarts onto a
    single point.
    """
    if scale <= 0:
        return
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) & 0x7FFFFFFF)
    with torch.no_grad():
        for p in monitor.parameters():
            noise = torch.empty_like(p)
            noise.normal_(0.0, scale, generator=gen)
            p.add_(noise)


def _train_suite(
    n: int,
    k: int,
    restarts: int,
    steps: int,
    cold_interval: int = COLD_INTERVAL,
    perturb_scale: float = PERTURB_SCALE,
    base_seed: int = BASE_SEED,
):
    """Train the full M3 → M2 → M1 → M1-hard cascade with mixed cold/warm restarts.

    For the warm-started classes (M2/M1/M1-hard), every ``cold_interval``-th
    restart skips the warm-start entirely (cold), while the rest apply the
    warm-start followed by a Gaussian perturbation of magnitude
    ``perturb_scale``. With ``cold_interval=4`` and ``restarts=20`` this gives
    5 cold + 15 warm-and-perturbed restarts per class.
    """

    def _is_cold(seed: int) -> bool:
        if cold_interval <= 0:
            return False
        return (seed - base_seed) % cold_interval == 0

    r3 = train_with_restarts(
        monitor_factory=lambda seed: M3Monitor(k=k, seed=seed),
        n=n,
        restarts=restarts,
        lr=1e-2,
        steps=steps,
    )

    def m2_factory(seed: int):
        monitor = M2Monitor(k=k, seed=seed)
        if not _is_cold(seed):
            r3.best_monitor.to_m2_warm_start(monitor)
            _perturb_in_place(monitor, perturb_scale, seed=seed * 7919 + 17)
        return monitor

    r2 = train_with_restarts(monitor_factory=m2_factory, n=n, restarts=restarts, lr=1e-2, steps=steps)

    def m1_factory(seed: int):
        monitor = M1Monitor(k=k, seed=seed)
        if not _is_cold(seed):
            r2.best_monitor.to_unconstrained_warm_start(monitor)
            _perturb_in_place(monitor, perturb_scale, seed=seed * 7919 + 31)
        return monitor

    r1 = train_with_restarts(monitor_factory=m1_factory, n=n, restarts=restarts, lr=1e-2, steps=steps)

    def m1_hard_factory(seed: int):
        monitor = M1HardMonitor(k=k, seed=seed)
        if not _is_cold(seed):
            r2.best_monitor.to_unconstrained_warm_start(monitor)
            _perturb_in_place(monitor, perturb_scale, seed=seed * 7919 + 53)
        return monitor

    r1_hard = train_with_restarts(
        monitor_factory=m1_hard_factory,
        n=n,
        restarts=restarts,
        lr=1e-2,
        steps=steps,
        report_eval_fn=report_evaluate,
    )

    return {"M1": r1, "M1-hard": r1_hard, "M2": r2, "M3": r3}


def _ground_truth_gap(gradient_f: float, exhaustive_f: float) -> tuple[float, float]:
    absolute_gap = abs(gradient_f - exhaustive_f)
    relative_gap = absolute_gap / max(abs(exhaustive_f), 1e-12)
    return absolute_gap, relative_gap


def _ground_truth_row(
    *,
    n: int,
    k: int,
    restarts: int,
    steps: int,
    lr: float,
    exhaustive_res: float,
    gap_threshold: float,
) -> dict[str, object]:
    result = train_with_restarts(
        monitor_factory=lambda seed: M1HardMonitor(k=k, seed=seed),
        n=n,
        restarts=restarts,
        lr=lr,
        steps=steps,
        report_eval_fn=report_evaluate,
        verbose=False,
    )
    exhaustive_f = exhaustive_search(n=n, k=k, res=exhaustive_res, verbose=False).f_star
    absolute_gap, relative_gap = _ground_truth_gap(result.best_loss, exhaustive_f)
    action = "passed"

    if relative_gap > gap_threshold:
        result = train_with_restarts(
            monitor_factory=lambda seed: M1HardMonitor(k=k, seed=seed),
            n=n,
            restarts=30,
            lr=5e-3,
            steps=steps,
            report_eval_fn=report_evaluate,
            verbose=False,
        )
        absolute_gap, relative_gap = _ground_truth_gap(result.best_loss, exhaustive_f)
        action = "rerun_30_restarts_lr_5e-3"

    passed = relative_gap <= gap_threshold
    if not passed:
        run_ga_search(
            k=k,
            n=n,
            population_size=60,
            generations=120,
            gradient_refine_steps=50,
            verbose=False,
        )
        action = "ga_required"

    return {
        "n": n,
        "k": k,
        "class": "M1-hard",
        "gradient_f": result.best_loss,
        "gradient_mean": result.mean_loss,
        "gradient_std": result.std_loss,
        "restarts": len(result.restarts),
        "steps": steps,
        "exhaustive_f": exhaustive_f,
        "absolute_gap": absolute_gap,
        "relative_gap": relative_gap,
        "gap_threshold": gap_threshold,
        "passed": passed,
        "action": action,
    }


def main(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    n_values: list[int] | tuple[int, ...] = N_VALUES,
    k_values: list[int] | tuple[int, ...] = K_VALUES,
    restarts: int = RESTARTS,
    steps: int = STEPS,
    cold_interval: int = COLD_INTERVAL,
    perturb_scale: float = PERTURB_SCALE,
    run_ga: bool = True,
    ga_cases: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = GA_CASES,
    ga_population: int = 60,
    ga_generations: int = 120,
    ga_gradient_refine_steps: int = 50,
    run_exhaustive: bool = True,
    exhaustive_cases: list[tuple[int, int]] | tuple[tuple[int, int], ...] = EXHAUSTIVE_CASES,
    exhaustive_res: float = 0.1,
    run_ground_truth: bool = True,
    ground_truth_cases: list[tuple[int, int]] | tuple[tuple[int, int], ...] = GROUND_TRUTH_CASES,
    ground_truth_restarts: int = RESTARTS,
    ground_truth_steps: int = STEPS,
    ground_truth_lr: float = 1e-2,
    ground_truth_exhaustive_res: float = 0.1,
    ground_truth_gap_threshold: float = 0.01,
    run_anneal: bool = True,
    generate_artifacts: bool = True,
    analysis_samples: int = 2048,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    results_csv = output_dir / "csv" / "phase1.csv"
    diagnostics_csv = output_dir / "csv" / "phase1_diagnostics.csv"
    ground_truth_csv = output_dir / "csv" / "phase1_ground_truth.csv"
    logs_path = output_dir / "logs" / "phase1.jsonl"

    rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    logs_path.parent.mkdir(parents=True, exist_ok=True)

    with logs_path.open("w", encoding="utf-8") as log_handle:
        for n in n_values:
            for k in k_values:
                print(f"\n=== Phase 1: n={n}, k={k} ===")
                t0 = time.perf_counter()
                suite = _train_suite(
                    n=n,
                    k=k,
                    restarts=restarts,
                    steps=steps,
                    cold_interval=cold_interval,
                    perturb_scale=perturb_scale,
                )
                wall = time.perf_counter() - t0

                gaps = ClassGaps(
                    n=n,
                    k=k,
                    f_M1=suite["M1"].best_loss,
                    f_M1_hard=suite["M1-hard"].best_loss,
                    f_M2=suite["M2"].best_loss,
                    f_M3=suite["M3"].best_loss,
                    # Dual-eval cross-values for consistent-eval gap quantities.
                    f_M1_alt=evaluate_hard(suite["M1"].best_monitor, n),
                    f_M1_hard_alt=evaluate_soft(suite["M1-hard"].best_monitor, n),
                    f_M2_alt=evaluate_hard(suite["M2"].best_monitor, n),
                    f_M3_alt=evaluate_hard(suite["M3"].best_monitor, n),
                )
                print(gaps.summary())
                violations = gaps.nesting_violations()
                if violations:
                    print("  !! nesting violations:", violations)

                baseline_f = baseline_value(n, k)
                ga_soft_f = None
                ga_hard_f = None
                exhaustive_f = None

                should_run_ga = run_ga and (ga_cases is None or (n, k) in set(ga_cases))
                if should_run_ga:
                    ga_monitor, _ = run_ga_search(
                        k=k,
                        n=n,
                        population_size=ga_population,
                        generations=ga_generations,
                        gradient_refine_steps=ga_gradient_refine_steps,
                        verbose=False,
                    )
                    ga_soft_f = evaluate_soft(ga_monitor, n)
                    ga_hard_f = evaluate_hard(ga_monitor, n)

                if run_exhaustive and (n, k) in exhaustive_cases:
                    exhaustive_f = exhaustive_search(n=n, k=k, res=exhaustive_res, verbose=False).f_star

                # First pass: compute diagnostics for every class (no figures yet).
                class_metrics: dict[str, dict[str, object]] = {}
                for class_name, result in suite.items():
                    class_metrics[class_name] = compute_monitor_metrics(
                        monitor=result.best_monitor,
                        phase="phase1",
                        n=n,
                        k=k,
                        class_name=class_name,
                        samples=analysis_samples,
                    )

                # Second pass: a single consolidated figure covering M1-M4.
                shared_paths = {}
                if generate_artifacts:
                    shared_paths = write_consolidated_figures(
                        suite=suite,
                        phase="phase1",
                        n=n,
                        k=k,
                        output_dir=output_dir,
                        samples=analysis_samples,
                        diagnostics=class_metrics,
                    )

                for class_name, result in suite.items():
                    metrics = dict(class_metrics[class_name])
                    if shared_paths:
                        metrics["heatmap_path"] = str(shared_paths["heatmap"].resolve())
                        metrics["posterior_scatter_path"] = str(
                            shared_paths["posterior_scatter"].resolve()
                        )
                    row = empty_result_row()
                    row.update(
                        {
                            "phase": "phase1",
                            "n": n,
                            "k": k,
                            "class": class_name,
                            "wall_clock_s": wall,
                            "delta_hard": gaps.delta_hard,
                            "delta_acc": gaps.delta_hard,
                            "delta_acc_soft": gaps.delta_acc_soft,
                            "delta_acc_hard": gaps.delta_acc_hard,
                            "delta_count": gaps.delta_count,
                            "delta_count_hard": gaps.delta_count_hard,
                            "delta_minimal": gaps.delta_minimal,
                            "delta_minimal_hard": gaps.delta_minimal_hard,
                            "baseline_f": baseline_f,
                        }
                    )
                    row.update(summarise_train_result(result, n=n))
                    row["baseline_gap"] = row["f_report"] - baseline_f
                    if class_name == "M1":
                        row["ga_f"] = ga_soft_f
                        row["ga_gap"] = None if ga_soft_f is None else ga_soft_f - row["f_report"]
                    if class_name == "M1-hard":
                        row["ga_f"] = ga_hard_f
                        row["ga_gap"] = None if ga_hard_f is None else ga_hard_f - row["f_report"]
                        row["exhaustive_f"] = exhaustive_f
                        row["exhaustive_gap"] = None if exhaustive_f is None else row["f_report"] - exhaustive_f
                        row.update(annealing_columns(result.best_monitor, n=n, enabled=run_anneal))
                    row.update(metrics)
                    rows.append(row)
                    diagnostics_rows.append(diagnostic_row("phase1", n, k, class_name, metrics))

                    log_handle.write(
                        json.dumps(
                            {
                                "phase": "phase1",
                                "n": n,
                                "k": k,
                                "class": class_name,
                                "train_best": result.best_train_loss,
                                "report_best": result.best_loss,
                                "report_losses": result.report_losses,
                                "curves": [rr.best_loss_curve for rr in result.restarts],
                            }
                        )
                        + "\n"
                    )

    write_csv(results_csv, rows, RESULT_COLUMNS)
    write_csv(diagnostics_csv, diagnostics_rows, diagnostics_rows[0].keys() if diagnostics_rows else [])
    if run_ground_truth:
        ground_truth_rows = [
            _ground_truth_row(
                n=n,
                k=k,
                restarts=ground_truth_restarts,
                steps=ground_truth_steps,
                lr=ground_truth_lr,
                exhaustive_res=ground_truth_exhaustive_res,
                gap_threshold=ground_truth_gap_threshold,
            )
            for n, k in ground_truth_cases
        ]
        write_csv(ground_truth_csv, ground_truth_rows, GROUND_TRUTH_COLUMNS)

    summary_paths: dict[str, object] = {}
    if generate_artifacts and rows:
        summary_paths = write_summary_figures(
            csv_path=results_csv,
            figure_dir=output_dir / "figures" / "phase1" / "summary",
            phase="phase1",
            include_dynamic=False,
        )

    print(f"\nPhase 1 complete. Results at {results_csv}")
    return {
        "results_csv": str(results_csv.resolve()),
        "diagnostics_csv": str(diagnostics_csv.resolve()),
        "ground_truth_csv": str(ground_truth_csv.resolve()),
        "logs_jsonl": str(logs_path.resolve()),
        **{k: str(Path(v).resolve()) for k, v in summary_paths.items()},
    }


if __name__ == "__main__":
    main()
