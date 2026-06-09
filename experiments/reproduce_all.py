"""Run the full reproducibility pipeline from Python."""
from __future__ import annotations

import csv
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.phase1_exploratory import main as run_phase1
from experiments.phase2_core import main as run_phase2
from experiments.phase3_stretch import main as run_phase3_main
from src.analysis.fitting import fit_all


def _collect_m1_results(csv_paths: list[Path]) -> dict[tuple[int, int], float]:
    data: dict[tuple[int, int], float] = {}
    for path in csv_paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("class") != "M1":
                    continue
                data[(int(row["n"]), int(row["k"]))] = float(row["f_report"])
    return data


def _write_fit_csv(path: Path, data: dict[tuple[int, int], float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = fit_all(data)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "params", "r2", "aic", "bic"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row.name,
                    "params": row.params,
                    "r2": row.r2,
                    "aic": row.aic,
                    "bic": row.bic,
                }
            )


def run_all(
    *,
    output_dir: str | Path = "results",
    run_phase3: bool = True,
    phase1_kwargs: dict | None = None,
    phase2_kwargs: dict | None = None,
    phase3_kwargs: dict | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    phase1_manifest = run_phase1(output_dir=output_dir, **(phase1_kwargs or {}))
    phase2_manifest = run_phase2(output_dir=output_dir, **(phase2_kwargs or {}))
    phase3_manifest = None
    csv_paths = [Path(phase1_manifest["results_csv"]), Path(phase2_manifest["results_csv"])]

    if run_phase3:
        phase3_manifest = run_phase3_main(output_dir=output_dir, **(phase3_kwargs or {}))
        csv_paths.append(Path(phase3_manifest["results_csv"]))

    fit_csv = output_dir / "csv" / "fits.csv"
    _write_fit_csv(fit_csv, _collect_m1_results(csv_paths))

    manifest = {
        "phase1_csv": str(Path(phase1_manifest["results_csv"]).resolve()),
        "phase2_csv": str(Path(phase2_manifest["results_csv"]).resolve()),
        "fit_csv": str(fit_csv.resolve()),
    }
    if phase3_manifest is not None:
        manifest["phase3_csv"] = str(Path(phase3_manifest["results_csv"]).resolve())
    return manifest


if __name__ == "__main__":
    run_all()
