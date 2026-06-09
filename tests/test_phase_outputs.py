from pathlib import Path

import pandas as pd

from experiments.phase1_exploratory import main as run_phase1
from experiments.common import write_summary_figures
from src.analysis.error_curves import plot_error_curves_vs_k


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
        run_ground_truth=False,
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


def test_phase1_smoke_writes_ground_truth_csv(tmp_path: Path) -> None:
    manifest = run_phase1(
        output_dir=tmp_path,
        n_values=[11],
        k_values=[2],
        restarts=1,
        steps=1,
        run_ga=False,
        run_exhaustive=False,
        run_anneal=False,
        generate_artifacts=False,
        run_ground_truth=True,
        ground_truth_cases=[(5, 2)],
        ground_truth_restarts=1,
        ground_truth_steps=1,
        ground_truth_exhaustive_res=1.0,
        ground_truth_gap_threshold=999.0,
    )

    ground_truth_path = Path(manifest["ground_truth_csv"])
    assert ground_truth_path.exists()

    ground_truth = pd.read_csv(ground_truth_path)
    expected_columns = {
        "n",
        "k",
        "class",
        "gradient_f",
        "gradient_mean",
        "gradient_std",
        "restarts",
        "steps",
        "exhaustive_f",
        "absolute_gap",
        "relative_gap",
        "gap_threshold",
        "passed",
        "action",
    }
    assert expected_columns.issubset(set(ground_truth.columns))
    assert len(ground_truth) == 1
    row = ground_truth.iloc[0]
    assert row["n"] == 5
    assert row["k"] == 2
    assert row["class"] == "M1-hard"
    assert ground_truth["passed"].dtype == bool


def test_phase1_ga_cross_check_is_recorded_on_m1_row(tmp_path: Path) -> None:
    manifest = run_phase1(
        output_dir=tmp_path,
        n_values=[21],
        k_values=[3],
        restarts=1,
        steps=1,
        run_ga=True,
        ga_cases=[(21, 3)],
        ga_population=4,
        ga_generations=2,
        ga_gradient_refine_steps=1,
        run_exhaustive=False,
        run_ground_truth=False,
        run_anneal=False,
        generate_artifacts=False,
        analysis_samples=64,
    )

    phase_csv = pd.read_csv(manifest["results_csv"])
    m1_row = phase_csv.loc[
        (phase_csv["n"] == 21) & (phase_csv["k"] == 3) & (phase_csv["class"] == "M1")
    ].iloc[0]
    m1_hard_row = phase_csv.loc[
        (phase_csv["n"] == 21) & (phase_csv["k"] == 3) & (phase_csv["class"] == "M1-hard")
    ].iloc[0]

    assert pd.notna(m1_row["ga_f"])
    assert pd.notna(m1_row["ga_gap"])
    assert pd.notna(m1_hard_row["ga_f"])


def test_error_curve_plot_handles_best_loss_below_ci_band(tmp_path: Path) -> None:
    csv_path = tmp_path / "phase1.csv"
    pd.DataFrame(
        [
            {
                "phase": "phase1",
                "n": 11,
                "k": 2,
                "class": "M1",
                "f_report": 0.30,
                "ci95_low": 0.31,
                "ci95_high": 0.32,
                "baseline_f": 0.35,
            }
        ]
    ).to_csv(csv_path, index=False)

    out_path = tmp_path / "error_curves.png"
    plot_error_curves_vs_k(csv_path, out_path)

    assert out_path.exists()


def test_monitor_bar_plots_make_overlapping_results_readable(tmp_path: Path) -> None:
    from src.analysis.monitor_bar_charts import plot_monitor_error_bars, plot_monitor_gap_bars

    csv_path = tmp_path / "phase1.csv"
    rows = []
    for n, k, baseline in [(11, 2, 0.38), (11, 3, 0.34)]:
        for cls, f_report in [
            ("M1", 0.30),
            ("M1-hard", 0.30001),
            ("M2", 0.30005),
            ("M3", 0.33),
        ]:
            rows.append(
                {
                    "phase": "phase1",
                    "n": n,
                    "k": k,
                    "class": cls,
                    "f_report": f_report,
                    "baseline_f": baseline,
                }
            )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    error_path = tmp_path / "monitor_error_bars.png"
    gap_path = tmp_path / "monitor_gap_bars.png"
    plot_monitor_error_bars(csv_path, error_path)
    plot_monitor_gap_bars(csv_path, gap_path)

    assert error_path.exists()
    assert gap_path.exists()


def test_summary_figures_include_bar_charts_and_stability_table(tmp_path: Path) -> None:
    csv_path = tmp_path / "phase1.csv"
    pd.DataFrame(
        [
            {
                "phase": "phase1",
                "n": 11,
                "k": 2,
                "class": cls,
                "f_report": f_report,
                "f_mean": f_report + 0.001,
                "f_std": 0.001,
                "ci95_low": f_report,
                "ci95_high": f_report + 0.002,
                "baseline_f": 0.36,
                "delta_hard": 0.00001,
                "delta_acc": 0.00001,
                "delta_count": 0.00005,
                "delta_minimal": 0.03,
                "restart_losses_json": "[0.30, 0.31]",
                "anneal_loss_tau_1_0": 0.32 if cls == "M1-hard" else None,
                "anneal_loss_tau_0_5": 0.31 if cls == "M1-hard" else None,
                "anneal_loss_tau_0_1": 0.305 if cls == "M1-hard" else None,
                "anneal_loss_tau_0_01": 0.301 if cls == "M1-hard" else None,
                "anneal_loss_tau_0_001": 0.3005 if cls == "M1-hard" else None,
            }
            for cls, f_report in [("M1", 0.30), ("M1-hard", 0.30001), ("M2", 0.30005), ("M3", 0.33)]
        ]
    ).to_csv(csv_path, index=False)

    outputs = write_summary_figures(csv_path, tmp_path / "figures", phase="phase1")

    for key in ("monitor_error_bars", "monitor_gap_bars", "stability_summary"):
        assert key in outputs
        assert Path(outputs[key]).exists()


def test_phase1_extended_smoke_generates_lightweight_outputs(tmp_path: Path) -> None:
    from experiments.phase1_extended import main as run_phase1_extended

    manifest = run_phase1_extended(
        output_dir=tmp_path,
        n_values=[11],
        k_values=[2],
        restarts=1,
        steps=1,
        run_ga=False,
        run_exhaustive=False,
        run_ground_truth=False,
        generate_artifacts=False,
    )

    assert Path(manifest["results_csv"]).exists()
    assert "monitor_gap_bars" in manifest
    assert Path(manifest["monitor_gap_bars"]).exists()
