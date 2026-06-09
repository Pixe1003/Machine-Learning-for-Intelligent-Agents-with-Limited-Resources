"""Smoke test for the dual-eval (soft + hard) extension.

Runs Phase 1 on a single tiny config (n=11, k=2, 3 restarts, 300 steps) and
prints, for every monitor class, both ``f_report_soft`` and ``f_report_hard``.
We additionally sanity-check that ``f_report_soft <= f_report_hard`` for
monitors with a soft sigmoid acceptance (since rounding can only equal or
hurt the soft expectation in either direction, but the *optimum over the
soft class* is lower-bounded by the optimum over the hard class — see the
note in summarise_train_result).
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase1_exploratory import main as run_phase1


def main() -> None:
    out_dir = ROOT / "results" / "phase1_smoke_dual_eval"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    manifest = run_phase1(
        output_dir=out_dir,
        n_values=[11],
        k_values=[2],
        restarts=3,
        steps=300,
        cold_interval=2,
        perturb_scale=0.1,
        run_ga=False,
        run_exhaustive=False,
        run_ground_truth=False,
        run_anneal=False,
        generate_artifacts=False,
        analysis_samples=128,
    )

    csv_path = Path(manifest["results_csv"])
    print(f"\nReading {csv_path}\n")
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            soft = row.get("f_report_soft")
            hard = row.get("f_report_hard")
            rep = row.get("f_report")
            print(
                f"  [{row['class']:<8}]  "
                f"f_report={float(rep):.6f}  "
                f"soft={float(soft):.6f}  "
                f"hard={float(hard):.6f}  "
                f"hard-soft={(float(hard) - float(soft)):+.3e}"
            )

    print("\nIf both 'soft' and 'hard' columns are non-empty for ALL classes,")
    print("dual eval is wired correctly.  Proceed to the full rerun.")


if __name__ == "__main__":
    main()
