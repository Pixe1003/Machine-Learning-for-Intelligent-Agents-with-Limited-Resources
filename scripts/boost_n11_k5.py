"""Boost run for (n=11, k=5).

The default Phase 1 run hit a stability warning at (11,5): std/mean ≈ 13%
on M1 / M1-hard.  Inspection of restart_losses_json showed that only ONE of
five cold restarts found the global-min basin (loss ≈ 0.118) while the M2
warm-start chain locked the other 15 restarts into a 0.148 plateau.

This script reruns just (n=11, k=5) with a denser cold-restart pool and
more restarts, then surgically replaces the four (11,5) rows in
results/csv/phase1.csv with the new ones (keeping the rest of the grid
intact).  The corresponding (n=11, k=5) figures are also overwritten and
the summary figures regenerated from the merged CSV.
"""
from __future__ import annotations

import csv
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import RESULT_COLUMNS, write_summary_figures
from experiments.phase1_exploratory import main as run_phase1

TARGET_N = 11
TARGET_K = 5
BOOST_RESTARTS = 30
BOOST_COLD_INTERVAL = 2  # 15 cold + 15 warm+perturb at 30 restarts
BOOST_STEPS = 3000


def _class_order_key(class_name: str) -> int:
    return {"M1": 0, "M1-hard": 1, "M2": 2, "M3": 3}.get(class_name, 99)


def _format_cell(value: object, fmt: str = ".6f") -> str:
    if value in (None, "", "None"):
        return "(empty)"
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)


def _as_float(value: object) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _backfill_consistent_deltas(rows: list[dict]) -> None:
    """Compute delta_acc_soft / delta_acc_hard / delta_count_hard /
    delta_minimal_hard for every (n, k) group from the existing dual-eval
    columns (f_report_soft / f_report_hard).  Rows are modified in-place.

    Lets us backfill the consistent-eval gap columns for legacy rows from a
    previous run that had dual-eval values but not yet the new delta wiring.
    """
    from collections import defaultdict

    groups: dict[tuple[int, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        try:
            key = (int(row["n"]), int(row["k"]))
        except (TypeError, ValueError):
            continue
        groups[key][row["class"]] = row

    for (n, k), cls_rows in groups.items():
        m1 = cls_rows.get("M1")
        m1h = cls_rows.get("M1-hard")
        m2 = cls_rows.get("M2")
        m3 = cls_rows.get("M3")
        if not (m1 and m1h and m2 and m3):
            continue

        m1_soft = _as_float(m1.get("f_report_soft"))
        m1_hard = _as_float(m1.get("f_report_hard"))
        m1h_soft = _as_float(m1h.get("f_report_soft"))
        m1h_hard = _as_float(m1h.get("f_report_hard"))
        m2_hard = _as_float(m2.get("f_report_hard"))
        m3_hard = _as_float(m3.get("f_report_hard"))

        delta_acc_soft = (
            m1h_soft - m1_soft
            if (m1h_soft is not None and m1_soft is not None)
            else None
        )
        delta_acc_hard = (
            m1h_hard - m1_hard
            if (m1h_hard is not None and m1_hard is not None)
            else None
        )
        delta_count_hard = (
            m2_hard - m1_hard
            if (m2_hard is not None and m1_hard is not None)
            else None
        )
        delta_minimal_hard = (
            m3_hard - m2_hard
            if (m3_hard is not None and m2_hard is not None)
            else None
        )

        for r in cls_rows.values():
            if delta_acc_soft is not None:
                r["delta_acc_soft"] = delta_acc_soft
            if delta_acc_hard is not None:
                r["delta_acc_hard"] = delta_acc_hard
            if delta_count_hard is not None:
                r["delta_count_hard"] = delta_count_hard
            if delta_minimal_hard is not None:
                r["delta_minimal_hard"] = delta_minimal_hard


def main() -> None:
    results_dir = ROOT / "results"
    main_csv = results_dir / "csv" / "phase1.csv"
    main_figures = results_dir / "figures" / "phase1"
    boost_dir = results_dir / "phase1_n11_k5_boost"

    # ------------------------------------------------------------------ backup
    if main_csv.exists():
        ts = time.strftime("%Y%m%d_%H%M%S")
        bkp = main_csv.with_name(f"phase1_pre_boost_{ts}.csv")
        shutil.copy2(main_csv, bkp)
        print(f"  Backup of pre-boost main CSV  ->  {bkp}")
    else:
        print(f"  !! {main_csv} not found; the boost run will still produce its own CSV.")

    if boost_dir.exists():
        shutil.rmtree(boost_dir)

    # ----------------------------------------------------------------- compute
    print(
        f"\nRunning boosted (n={TARGET_N}, k={TARGET_K}) with "
        f"{BOOST_RESTARTS} restarts and cold_interval={BOOST_COLD_INTERVAL} ...\n"
    )
    t0 = time.perf_counter()
    manifest = run_phase1(
        output_dir=boost_dir,
        n_values=[TARGET_N],
        k_values=[TARGET_K],
        restarts=BOOST_RESTARTS,
        steps=BOOST_STEPS,
        cold_interval=BOOST_COLD_INTERVAL,
        perturb_scale=0.1,
        run_ga=True,
        run_exhaustive=False,
        run_ground_truth=False,
        run_anneal=True,
        generate_artifacts=True,
        analysis_samples=2048,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nBoost finished in {elapsed/60:.1f} minutes.")

    boost_csv = Path(manifest["results_csv"])
    with boost_csv.open(encoding="utf-8") as fh:
        boost_rows = list(csv.DictReader(fh))

    if not boost_rows:
        print("!!! Boost run produced no rows. Aborting merge.")
        return

    # ----------------------------------------------------------------- merge
    if main_csv.exists():
        with main_csv.open(encoding="utf-8") as fh:
            main_rows = list(csv.DictReader(fh))
        keep = [
            r
            for r in main_rows
            if not (int(r["n"]) == TARGET_N and int(r["k"]) == TARGET_K)
        ]
        merged = keep + boost_rows
    else:
        merged = boost_rows

    merged.sort(
        key=lambda r: (int(r["n"]), int(r["k"]), _class_order_key(r["class"]))
    )

    # Backfill consistent-eval delta columns from f_report_soft / f_report_hard
    # for every (n, k) group, including the legacy rows that did not yet have
    # the new columns wired in at training time.
    _backfill_consistent_deltas(merged)

    main_csv.parent.mkdir(parents=True, exist_ok=True)
    with main_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in merged:
            writer.writerow({key: row.get(key) for key in RESULT_COLUMNS})
    print(f"\nUpdated main CSV  ->  {main_csv}")

    # ----------------------------------------------------------------- figures
    # Copy the boost (n=11, k=5) figure files over the main ones so the
    # dissertation has the regenerated heatmap / scatter.
    boost_fig_dir = boost_dir / "figures" / "phase1"
    if boost_fig_dir.exists():
        main_figures.mkdir(parents=True, exist_ok=True)
        for fig in boost_fig_dir.glob(f"n{TARGET_N}_k{TARGET_K}_*.png"):
            dst = main_figures / fig.name
            shutil.copy2(fig, dst)
            print(f"  Figure copied  ->  {dst.name}")

    # Regenerate the summary figures from the merged CSV.
    summary_dir = main_figures / "summary"
    print("\nRegenerating Phase 1 summary figures from merged CSV ...")
    summary_paths = write_summary_figures(
        csv_path=main_csv,
        figure_dir=summary_dir,
        phase="phase1",
        include_dynamic=False,
    )
    for name, path in summary_paths.items():
        print(f"  {name}  ->  {path}")

    # ----------------------------------------------------------------- report
    print("\nNew (n=11, k=5) rows after boost:")
    for row in sorted(boost_rows, key=lambda r: _class_order_key(r["class"])):
        f_std = _format_cell(row.get("f_std"), ".3e")
        n_restarts = row.get("restarts", "?")
        print(
            f"  [{row['class']:<8}]  "
            f"f_report={_format_cell(row.get('f_report'))}  "
            f"soft={_format_cell(row.get('f_report_soft'))}  "
            f"hard={_format_cell(row.get('f_report_hard'))}  "
            f"f_std={f_std}  restarts={n_restarts}"
        )


if __name__ == "__main__":
    main()
