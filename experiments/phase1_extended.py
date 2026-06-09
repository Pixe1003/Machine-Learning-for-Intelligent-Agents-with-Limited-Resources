"""Lightweight extended Phase 1 grid.

This entrypoint expands the static grid without replacing the formal Phase 1
run.  Defaults are intentionally cheap enough for exploratory reporting:

* n in {11, 21, 31}
* k in {2, 3, 4}
* 5 restarts
* no GA / exhaustive / ground-truth checks by default
* no per-configuration heatmap/scatter generation by default
"""
from __future__ import annotations

from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import write_summary_figures
from experiments.phase1_exploratory import main as run_phase1

DEFAULT_OUTPUT_DIR = Path("results") / "phase1_extended"
N_VALUES = [11, 21, 31]
K_VALUES = [2, 3, 4]
RESTARTS = 5
STEPS = 1500


def main(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    n_values: list[int] | tuple[int, ...] = N_VALUES,
    k_values: list[int] | tuple[int, ...] = K_VALUES,
    restarts: int = RESTARTS,
    steps: int = STEPS,
    run_ga: bool = False,
    run_exhaustive: bool = False,
    run_ground_truth: bool = False,
    run_anneal: bool = False,
    generate_artifacts: bool = False,
    analysis_samples: int = 1024,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    manifest = run_phase1(
        output_dir=output_dir,
        n_values=n_values,
        k_values=k_values,
        restarts=restarts,
        steps=steps,
        run_ga=run_ga,
        run_exhaustive=run_exhaustive,
        run_ground_truth=run_ground_truth,
        run_anneal=run_anneal,
        generate_artifacts=generate_artifacts,
        analysis_samples=analysis_samples,
    )
    summary_paths = write_summary_figures(
        csv_path=manifest["results_csv"],
        figure_dir=output_dir / "figures" / "phase1_extended" / "summary",
        phase="phase1_extended",
        include_dynamic=False,
    )
    manifest.update({key: str(Path(path).resolve()) for key, path in summary_paths.items()})
    return manifest


if __name__ == "__main__":
    main()
