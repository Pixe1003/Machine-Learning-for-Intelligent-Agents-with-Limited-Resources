"""Phase 3 stretch runs using the shared-parameter dynamic monitor."""
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
from src.monitors import M1HardMonitor, M1Monitor, M2Monitor, M3Monitor
from src.monitors.dynamic import SharedDynamicMonitor
from src.optim.gradient import train_with_restarts

DEFAULT_OUTPUT_DIR = Path("results")
N = 201
K_VALUES = [10, 20]
RESTARTS_STATIC = 10
RESTARTS_DYNAMIC = 3
STEPS = 5000


def main(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    n: int = N,
    k_values: list[int] | tuple[int, ...] = K_VALUES,
    restarts_static: int = RESTARTS_STATIC,
    restarts_dynamic: int = RESTARTS_DYNAMIC,
    steps: int = STEPS,
    run_anneal: bool = True,
    generate_artifacts: bool = True,
    analysis_samples: int = 2048,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    results_csv = output_dir / "csv" / "phase3.csv"
    diagnostics_csv = output_dir / "csv" / "phase3_diagnostics.csv"
    logs_path = output_dir / "logs" / "phase3.jsonl"
    rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    logs_path.parent.mkdir(parents=True, exist_ok=True)

    with logs_path.open("w", encoding="utf-8") as log_handle:
        for k in k_values:
            print(f"\n=== Phase 3: n={n}, k={k} ===")
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
                lambda seed: SharedDynamicMonitor(k=k, n=n, hidden=64, seed=seed),
                n=n,
                restarts=restarts_dynamic,
                lr=5e-3,
                steps=steps,
            )
            wall = time.perf_counter() - t0
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
            delta_dyn = r1.best_loss - rd.best_loss
            baseline_f = baseline_value(n, k)
            suite = {"M1": r1, "M1-hard": r1_hard, "M2": r2, "M3": r3, "DynamicShared": rd}

            class_metrics: dict[str, dict[str, object]] = {}
            for class_name, result in suite.items():
                class_metrics[class_name] = compute_monitor_metrics(
                    monitor=result.best_monitor,
                    phase="phase3",
                    n=n,
                    k=k,
                    class_name=class_name,
                    samples=analysis_samples,
                )

            shared_paths = {}
            if generate_artifacts:
                shared_paths = write_consolidated_figures(
                    suite=suite,
                    phase="phase3",
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
                        "phase": "phase3",
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
                    row.update(annealing_columns(result.best_monitor, n=n, enabled=run_anneal))
                row.update(metrics)
                rows.append(row)
                diagnostics_rows.append(diagnostic_row("phase3", n, k, class_name, metrics))
                log_handle.write(
                    json.dumps(
                        {
                            "phase": "phase3",
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

            print(f"  wall {wall:.1f}s   static M1={r1.best_loss:.5f}   dyn={rd.best_loss:.5f}")

    write_csv(results_csv, rows, RESULT_COLUMNS)
    write_csv(diagnostics_csv, diagnostics_rows, diagnostics_rows[0].keys() if diagnostics_rows else [])

    summary_paths: dict[str, object] = {}
    if generate_artifacts and rows:
        summary_paths = write_summary_figures(
            csv_path=results_csv,
            figure_dir=output_dir / "figures" / "phase3" / "summary",
            phase="phase3",
            include_dynamic=True,
        )

    print(f"\nPhase 3 complete. Results at {results_csv}")
    return {
        "results_csv": str(results_csv.resolve()),
        "diagnostics_csv": str(diagnostics_csv.resolve()),
        "logs_jsonl": str(logs_path.resolve()),
        **{k: str(Path(v).resolve()) for k, v in summary_paths.items()},
    }


if __name__ == "__main__":
    main()
