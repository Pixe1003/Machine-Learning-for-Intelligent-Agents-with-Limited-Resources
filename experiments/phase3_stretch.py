"""Phase 3 stretch runs using the shared-parameter dynamic monitor."""
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
                lambda seed: SharedDynamicMonitor(k=k, n=n, hidden=64, seed=seed),
                n=n,
                restarts=restarts_dynamic,
                lr=5e-3,
                steps=steps,
            )
            wall = time.perf_counter() - t0
            gaps = ClassGaps(n=n, k=k, f_M1=r1.best_loss, f_M2=r2.best_loss, f_M3=r3.best_loss, f_M4=r4.best_loss)
            delta_dyn = r1.best_loss - rd.best_loss
            baseline_f = baseline_value(n, k)
            suite = {"M1": r1, "M2": r2, "M3": r3, "M4": r4, "DynamicShared": rd}

            for class_name, result in suite.items():
                metrics = write_monitor_artifacts(
                    monitor=result.best_monitor,
                    phase="phase3",
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
                        "phase": "phase3",
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
    print(f"\nPhase 3 complete. Results at {results_csv}")
    return {
        "results_csv": str(results_csv.resolve()),
        "diagnostics_csv": str(diagnostics_csv.resolve()),
        "logs_jsonl": str(logs_path.resolve()),
    }


if __name__ == "__main__":
    main()
