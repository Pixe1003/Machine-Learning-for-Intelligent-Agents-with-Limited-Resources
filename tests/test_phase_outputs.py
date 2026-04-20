from pathlib import Path

import pandas as pd

from experiments.phase1_exploratory import main as run_phase1


def test_phase1_smoke_writes_extended_schema_and_artifacts(tmp_path: Path) -> None:
    manifest = run_phase1(
        output_dir=tmp_path,
        n_values=[11],
        k_values=[2],
        restarts=1,
        steps=1,
        run_ga=True,
        ga_population=4,
        ga_generations=2,
        ga_gradient_refine_steps=1,
        run_exhaustive=True,
        exhaustive_cases=[(11, 2)],
        exhaustive_res=1.0,
        run_anneal=True,
        generate_artifacts=True,
        analysis_samples=64,
    )

    phase_csv = pd.read_csv(manifest["results_csv"])
    expected_columns = {
        "phase",
        "n",
        "k",
        "class",
        "f_train_best",
        "f_report",
        "f_mean",
        "f_std",
        "ci95_low",
        "ci95_high",
        "restart_losses_json",
        "baseline_f",
        "baseline_gap",
        "ga_f",
        "ga_gap",
        "exhaustive_f",
        "exhaustive_gap",
        "anneal_loss_tau_0_1",
        "tridiagonality",
        "posterior_monotonicity_mean",
        "transition_entropy_mean",
        "heatmap_path",
        "posterior_scatter_path",
    }
    assert expected_columns.issubset(set(phase_csv.columns))

    diagnostics_csv = pd.read_csv(manifest["diagnostics_csv"])
    assert not diagnostics_csv.empty

    heatmap = Path(phase_csv.loc[phase_csv["class"] == "M1", "heatmap_path"].iloc[0])
    scatter = Path(phase_csv.loc[phase_csv["class"] == "M1", "posterior_scatter_path"].iloc[0])
    assert heatmap.exists()
    assert scatter.exists()
