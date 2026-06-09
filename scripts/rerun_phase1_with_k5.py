"""Back up the current Phase 1 artefacts and rerun the full grid.

Adds:
* Dual-eval columns ``f_report_soft`` / ``f_report_hard`` on every row.
* k=5 to the default grid (n in {11, 21}, k in {2, 3, 5}).

The pre-existing CSV and figure outputs are first copied into a sibling
directory ``results_pre_dual_eval/`` so the previous run is not lost.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase1_exploratory import main as run_phase1


def _backup_existing(results_dir: Path, backup_dir: Path) -> None:
    """Copy CSV + figure outputs of the previous run into ``backup_dir``."""
    if not results_dir.exists():
        print(f"  (no previous {results_dir} to back up)")
        return
    if backup_dir.exists():
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = backup_dir.with_name(backup_dir.name + f"_{ts}")
    print(f"  backing up {results_dir} -> {backup_dir}")
    shutil.copytree(results_dir, backup_dir, dirs_exist_ok=False)


def main() -> None:
    results_dir = ROOT / "results"
    backup_dir = ROOT / "results_pre_dual_eval"

    print("Step 1/2  Backing up the existing Phase 1 outputs ...")
    # Copy only the CSV + figures subdirs we care about; venv-style ignores.
    for sub in ["csv", "logs", "figures"]:
        src = results_dir / sub
        dst = backup_dir / sub
        if src.exists():
            if dst.exists():
                ts = time.strftime("%Y%m%d_%H%M%S")
                dst = dst.with_name(dst.name + f"_{ts}")
            print(f"    copy  {src}  ->  {dst}")
            shutil.copytree(src, dst)
        else:
            print(f"    skip  {src} (does not exist)")

    print("\nStep 2/2  Rerunning Phase 1 with dual eval + k in {2, 3, 5} ...")
    t0 = time.perf_counter()
    manifest = run_phase1()  # uses module defaults: n in {11,21}, k in {2,3,5}
    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed/60:.1f} minutes.")
    print("Outputs:")
    for k, v in manifest.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
