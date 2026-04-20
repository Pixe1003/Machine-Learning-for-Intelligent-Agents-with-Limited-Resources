"""
Phase 1 exploratory runs.

Defaults follow the proposal's exploratory grid, but the entrypoint is now
parameterised so tests and smoke runs can execute a tiny slice of the pipeline.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from experiments.common import (
    RESULT_COLUMNS,
    annealing_columns,
    baseline_value,
    diagnostic_row,
    empty_result_row,
    summarise_train_result,
    write_csv,
    write_monitor_artifacts,
)
from src.analysis.class_gaps import ClassGaps
from src.core.loss import report_evaluate
from src.monitors import M1Monitor, M2Monitor, M3Monitor, M4Monitor
from src.optim.exhaustive import exhaustive_search
from src.optim.genetic import run_ga as run_ga_search
from src.optim.gradient import train_with_restarts

DEFAULT_OUTPUT_DIR = Path("results")
N_VALUES = [11, 21]
K_VALUES = [2, 3]
RESTARTS = 20
STEPS = 3000
EXHAUSTIVE_CASES = [(11, 2)]


def _train_suite(n: int, k: int, restarts: int, steps: int):
    r4 = train_with_restarts(
        monitor_factory=lambda seed: M4Monitor(k=k, seed=seed),
        n=n,
        restarts=restarts,
        lr=1e-2,
        steps=steps,
    )

    def m3_factory(seed: int):
        monitor = M3Monitor(k=k, seed=seed)
        r4.best_monitor.to_m3_warm_start(monitor)
        return monitor

    r3 = train_with_restarts(monitor_factory=m3_factory, n=n, restarts=restarts, lr=1e-2, steps=steps)

    def m2_factory(seed: int):
        monitor = M2Monitor(k=k, seed=seed)
        r3.best_monitor.to_m1_warm_start(monitor)
        return monitor

    r2 = train_with_restarts(monitor_factory=m2_factory, n=n, restarts=restarts, lr=1e-2, steps=steps)

    def m1_factory(seed: int):
        monitor = M1Monitor(k=k, seed=seed)
        r3.best_monitor.to_m1_warm_start(monitor)
        return monitor

    r1 = train_with_restarts(
        monitor_factory=m1_factory,
        n=n,
        restarts=restarts,
        lr=1e-2,
        steps=steps,
        report_eval_fn=report_evaluate,
    )

    return {"M1": r1, "M2": r2, "M3": r3, "M4": r4}


def main(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    n_values: list[int] | tuple[int, ...] = N_VALUES,
    k_values: list[int] | tuple[int, ...] = K_VALUES,
    restarts: int = RESTARTS,
    steps: int = STEPS,
    run_ga: bool = True,
    ga_population: int = 60,
    ga_generations: int = 120,
    ga_gradient_refine_steps: int = 50,
    run_exhaustive: bool = True,
    exhaustive_cases: list[tuple[int, int]] | tuple[tuple[int, int], ...] = EXHAUSTIVE_CASES,
    exhaustive_res: float = 0.1,
    run_anneal: bool = True,
    generate_artifacts: bool = True,
    analysis_samples: int = 2048,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    results_csv = output_dir / "csv" / "phase1.csv"
    diagnostics_csv = output_dir / "csv" / "phase1_diagnostics.csv"
    logs_path = output_dir / "logs" / "phase1.jsonl"

    rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    logs_path.parent.mkdir(parents=True, exist_ok=True)

    with logs_path.open("w", encoding="utf-8") as log_handle:
        for n in n_values:
            for k in k_values:
                print(f"\n=== Phase 1: n={n}, k={k} ===")
                t0 = time.perf_counter()
                suite = _train_suite(n=n, k=k, restarts=restarts, steps=steps)
                wall = time.perf_counter() - t0

                gaps = ClassGaps(
                    n=n,
                    k=k,
                    f_M1=suite["M1"].best_loss,
                    f_M2=suite["M2"].best_loss,
                    f_M3=suite["M3"].best_loss,
                    f_M4=suite["M4"].best_loss,
                )
                print(gaps.summary())
                violations = gaps.nesting_violations()
                if violations:
                    print("  !! nesting violations:", violations)

                baseline_f = baseline_value(n, k)
                ga_f = None
                exhaustive_f = None

                if run_ga:
                    ga_monitor, _ = run_ga_search(
                        k=k,
                        n=n,
                        population_size=ga_population,
                        generations=ga_generations,
                        gradient_refine_steps=ga_gradient_refine_steps,
                        verbose=False,
                    )
                    ga_f = report_evaluate(ga_monitor, n)

                if run_exhaustive and (n, k) in exhaustive_cases:
                    exhaustive_f = exhaustive_search(n=n, k=k, res=exhaustive_res, verbose=False).f_star

                for class_name, result in suite.items():
                    metrics = write_monitor_artifacts(
                        monitor=result.best_monitor,
                        phase="phase1",
                        n=n,
                        k=k,
                        class_name=class_name,
                        output_dir=output_dir,
                        samples=analysis_samples,
                        generate=generate_artifacts,
                    )
                    row = empty_result_row()
                    row.update(
                        {
                            "phase": "phase1",
                            "n": n,
                            "k": k,
                            "class": class_name,
                            "wall_clock_s": wall,
                            "delta_acc": gaps.delta_acc,
                            "delta_count": gaps.delta_count,
                            "delta_minimal": gaps.delta_minimal,
                            "baseline_f": baseline_f,
                        }
                    )
                    row.update(summarise_train_result(result))
                    row["baseline_gap"] = row["f_report"] - baseline_f
                    if class_name == "M1":
                        row["ga_f"] = ga_f
                        row["ga_gap"] = None if ga_f is None else ga_f - row["f_report"]
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
    print(f"\nPhase 1 complete. Results at {results_csv}")
    return {
        "results_csv": str(results_csv.resolve()),
        "diagnostics_csv": str(diagnostics_csv.resolve()),
        "logs_jsonl": str(logs_path.resolve()),
    }


if __name__ == "__main__":
    main()
