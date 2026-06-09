"""Smoke test: confirm the cold/warm restart fix yields non-zero std on M1/M2/M1-hard.

Runs Phase 1 with a tiny grid (n=11, k=2 only), 5 restarts, 500 steps, and
prints the restart losses for each class so we can eyeball whether they're
actually different across restarts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase1_exploratory import main as run_phase1


def _summarise(losses: list[float]) -> str:
    if not losses:
        return "(empty)"
    mean = sum(losses) / len(losses)
    var = sum((x - mean) ** 2 for x in losses) / max(1, len(losses) - 1)
    std = var ** 0.5
    rng = max(losses) - min(losses)
    return f"mean={mean:.6f} std={std:.3e} range={rng:.3e} min={min(losses):.6f} max={max(losses):.6f}"


def main() -> None:
    out_dir = ROOT / "results" / "phase1_smoke_fix"
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    manifest = run_phase1(
        output_dir=out_dir,
        n_values=[11],
        k_values=[2],
        restarts=5,
        steps=500,
        cold_interval=4,
        perturb_scale=0.1,
        run_ga=False,
        run_exhaustive=False,
        run_ground_truth=False,
        run_anneal=False,
        generate_artifacts=False,
        analysis_samples=256,
    )

    csv_path = Path(manifest["results_csv"])
    print(f"\nReading {csv_path}")
    import csv

    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cls = row["class"]
            try:
                losses = json.loads(row["restart_losses_json"])
            except Exception:
                losses = []
            print(f"\n[{cls}]  f_report={float(row['f_report']):.6f}  f_std={float(row['f_std']):.3e}")
            print(f"  restart losses: {[f'{x:.6f}' for x in losses]}")
            print(f"  {_summarise(losses)}")


if __name__ == "__main__":
    main()
