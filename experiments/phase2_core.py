"""Phase 2 core results: static + dynamic."""
from __future__ import annotations

import json
import time
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from src.monitors import DynamicMonitor, M1HardMonitor, M1Monitor, M2Monitor, M3Monitor
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
                r3 = train_with_restarts(lambda seed: M3Monitor(k=k, seed=seed), n=n, restarts=restarts_static, lr=1e-2, steps=steps)
                r2 = train_with_restarts(lambda seed: M2Monitor(k=k, seed=seed), n=n, restarts=restarts_static, lr=1e-2, steps=steps)
                r1 = train_with_restarts(lambda seed: M1Monitor(k=k, seed=seed), n=n, restarts=restarts_static, lr=1e-2, steps=steps)
                r1_hard = train_with_restarts(
                    lambda seed: M1HardMonitor(k=k, seed=seed),
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
                gaps = ClassGaps(
                    n=n,
                    k=k,
                    f_M1=r1.best_loss,
                    f_M1_hard=r1_hard.best_loss,
                    f_M2=r2.best_loss,
                    f_M3=r3.best_loss,
                    f_M1_alt=evaluate_hard(r1.best_monitor, n),
                    f_M1_hard_alt=evaluate_soft(r1_hard.best_monitor, n),
                    f_M2_alt=evaluate_hard(r2.best_monitor, n),
                    f_M3_alt=evaluate_hard(r3.best_monitor, n),
                )
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

                suite = {"M1": r1, "M1-hard": r1_hard, "M2": r2, "M3": r3, "Dynamic": rd}

                class_metrics: dict[str, dict[str, object]] = {}
                for class_name, result in suite.items():
                    class_metrics[class_name] = compute_monitor_metrics(
                        monitor=result.best_monitor,
                        phase="phase2",
                        n=n,
                        k=k,
                        class_name=class_name,
                        samples=analysis_samples,
                    )

                shared_paths = {}
                if generate_artifacts:
                    shared_paths = write_consolidated_figures(
                        suite=suite,
                        phase="phase2",
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
                            "phase": "phase2",
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
                            "delta_dynamic": delta_dyn,
                            "baseline_f": baseline_f,
                        }
                    )
                    row.update(summarise_train_result(result, n=n))
                    row["baseline_gap"] = row["f_report"] - baseline_f
                    if class_name == "M1-hard":
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

    summary_paths: dict[str, object] = {}
    if generate_artifacts and rows:
        summary_paths = write_summary_figures(
            csv_path=results_csv,
            figure_dir=output_dir / "figures" / "phase2" / "summary",
            phase="phase2",
            include_dynamic=True,
        )

    print(f"\nPhase 2 complete. Results at {results_csv}")
    return {
        "results_csv": str(results_csv.resolve()),
        "diagnostics_csv": str(diagnostics_csv.resolve()),
        "logs_jsonl": str(logs_path.resolve()),
        **{k: str(Path(v).resolve()) for k, v in summary_paths.items()},
    }


if __name__ == "__main__":
    main()
