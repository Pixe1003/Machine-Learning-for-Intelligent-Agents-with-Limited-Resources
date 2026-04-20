"""Phase 2 core results: static + dynamic."""
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
from src.monitors import DynamicMonitor, M1Monitor, M2Monitor, M3Monitor, M4Monitor
from src.optim.genetic import run_ga as run_ga_search
from src.optim.gradient import train_with_restarts

DEFAULT_OUTPUT_DIR = Path("results")
N_VALUES = [51, 101]
K_VALUES = [2, 3, 5]
RESTARTS_STATIC = 20
RESTARTS_DYNAMIC = 5
STEPS = 3000


def main(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    n_values: list[int] | tuple[int, ...] = N_VALUES,
    k_values: list[int] | tuple[int, ...] = K_VALUES,
    restarts_static: int = RESTARTS_STATIC,
    restarts_dynamic: int = RESTARTS_DYNAMIC,
    steps: int = STEPS,
    run_ga: bool = False,
    ga_population: int = 60,
    ga_generations: int = 120,
    ga_gradient_refine_steps: int = 50,
    run_anneal: bool = True,
    generate_artifacts: bool = True,
    analysis_samples: int = 2048,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    results_csv = output_dir / "csv" / "phase2.csv"
    diagnostics_csv = output_dir / "csv" / "phase2_diagnostics.csv"
    logs_path = output_dir / "logs" / "phase2.jsonl"
    rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    logs_path.parent.mkdir(parents=True, exist_ok=True)

    with logs_path.open("w", encoding="utf-8") as log_handle:
        for n in n_values:
            for k in k_values:
                print(f"\n=== Phase 2: n={n}, k={k} ===")
                t0 = time.perf_counter()
                r4 = train_with_restarts(lambda seed: M4Monitor(k=k, seed=seed), n=n, restarts=restarts_static, lr=1e-2, steps=steps)
                r3 = train_with_restarts(lambda seed: M3Monitor(k=k, seed=seed), n=n, restarts=restarts_static, lr=1e-2, steps=steps)
                r2 = train_with_restarts(lambda seed: M2Monitor(k=k, seed=seed), n=n, restarts=restarts_static, lr=1e-2, steps=steps)
                r1 = train_with_restarts(
                    lambda seed: M1Monitor(k=k, seed=seed),
                    n=n,
                    restarts=restarts_static,
                    lr=1e-2,
                    steps=steps,
                    report_eval_fn=report_evaluate,
                )
                rd = train_with_restarts(
                    lambda seed: DynamicMonitor(k=k, n=n, seed=seed),
                    n=n,
                    restarts=restarts_dynamic,
                    lr=1e-2,
                    steps=steps,
                )
                wall = time.perf_counter() - t0
                delta_dyn = r1.best_loss - rd.best_loss
                gaps = ClassGaps(n=n, k=k, f_M1=r1.best_loss, f_M2=r2.best_loss, f_M3=r3.best_loss, f_M4=r4.best_loss)
                baseline_f = baseline_value(n, k)

                ga_f = None
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

                suite = {"M1": r1, "M2": r2, "M3": r3, "M4": r4, "Dynamic": rd}
                for class_name, result in suite.items():
                    metrics = write_monitor_artifacts(
                        monitor=result.best_monitor,
                        phase="phase2",
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
                            "phase": "phase2",
                            "n": n,
                            "k": k,
                            "class": class_name,
                            "wall_clock_s": wall,
                            "delta_acc": gaps.delta_acc,
                            "delta_count": gaps.delta_count,
                            "delta_minimal": gaps.delta_minimal,
                            "delta_dynamic": delta_dyn,
                            "baseline_f": baseline_f,
                        }
                    )
                    row.update(summarise_train_result(result))
                    row["baseline_gap"] = row["f_report"] - baseline_f
                    if class_name == "M1":
                        row["ga_f"] = ga_f
                        row["ga_gap"] = None if ga_f is None else ga_f - row["f_report"]
                        row.update(annealing_columns(result.best_monitor, n=n, enabled=run_anneal))
                    row.update(metrics)
                    rows.append(row)
                    diagnostics_rows.append(diagnostic_row("phase2", n, k, class_name, metrics))
                    log_handle.write(
                        json.dumps(
                            {
                                "phase": "phase2",
                                "n": n,
                                "k": k,
                                "class": class_name,
                                "train_best": result.best_train_loss,
                                "report_best": result.best_loss,
                                "report_losses": result.report_losses,
                            }
                        )
                        + "\n"
                    )

                print(gaps.summary())
                print(f"  dynamic f = {rd.best_loss:.5f}  Δ(n,k) = {delta_dyn:+.5f}   total wall {wall:.1f}s")

    write_csv(results_csv, rows, RESULT_COLUMNS)
    write_csv(diagnostics_csv, diagnostics_rows, diagnostics_rows[0].keys() if diagnostics_rows else [])
    print(f"\nPhase 2 complete. Results at {results_csv}")
    return {
        "results_csv": str(results_csv.resolve()),
        "diagnostics_csv": str(diagnostics_csv.resolve()),
        "logs_jsonl": str(logs_path.resolve()),
    }


if __name__ == "__main__":
    main()
