from pathlib import Path

from experiments.reproduce_all import run_all


def test_reproduce_pipeline_generates_csvs_and_figures(tmp_path: Path) -> None:
    manifest = run_all(
        output_dir=tmp_path,
        run_phase3=False,
        phase1_kwargs={
            "n_values": [11],
            "k_values": [2],
            "restarts": 1,
            "steps": 1,
            "run_ga": False,
            "run_exhaustive": False,
            "run_ground_truth": False,
            "run_anneal": True,
            "generate_artifacts": True,
            "analysis_samples": 64,
        },
        phase2_kwargs={
            "n_values": [11],
            "k_values": [2],
            "restarts_static": 1,
            "restarts_dynamic": 1,
            "steps": 1,
            "run_ga": False,
            "run_anneal": True,
            "generate_artifacts": True,
            "analysis_samples": 64,
        },
    )

    for key in ("phase1_csv", "phase2_csv", "fit_csv"):
        assert Path(manifest[key]).exists()

    figure_dir = tmp_path / "figures"
    assert list(figure_dir.rglob("*.png"))
