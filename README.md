# Machine Learning for Intelligent Agents with Limited Resources

Final-year project (School of Informatics, University of Edinburgh) on
learning memory-bounded monitors for the Majority Problem.

Supervisor: Richard Mayr  
Tutor: Aurora Constantin

## Overview

This repository studies the **Majority Problem** under strict memory limits.
Given an odd number `n` of i.i.d. fair-coin votes, a probabilistic finite-state
monitor with only `k` internal states must decide whether the majority voted
`Head`.

The codebase is built around a single differentiable evaluation pipeline:

1. define a monitor class,
2. compute its exact finite-`n` error `f(n, k)` with a dynamic program,
3. optimise parameters with gradient descent and optional auxiliary methods,
4. export comparable experiment tables and structural diagnostics.

The current architecture is designed to support the full proposal workflow:

- four static monitor classes (`M1`-`M4`),
- dynamic monitors for the static-vs-dynamic comparison,
- gradient descent as the primary optimiser,
- optional GA / exhaustive / annealing probes,
- reproducible CSV / figure / log outputs,
- structural analysis for tridiagonality, posterior monotonicity, and
  transition entropy.

## Research Questions

The repository is organised around the following empirical questions:

- **H1** Is the optimal static monitor behaviourally close to a stochastic
  counter of the `Head-Tail` difference?
- **H2** Does the optimum genuinely require randomised transitions, or does it
  collapse to a near-deterministic machine under annealing?
- **H3** What structure does the optimal dynamic monitor learn?
- **H4** How large is the dynamic advantage
  `Delta(n, k) = f_static(n, k) - f_dynamic(n, k)`?
- **H5** Does soft acceptance improve performance at small `k`?

## Monitor Classes

The static classes follow the proposal's factorial design:

| Class | Transition family | Acceptance | Purpose |
|-------|-------------------|------------|---------|
| `M1` | Unconstrained row-stochastic `H, T` | Hard acceptance at report time | Most expressive static baseline |
| `M2` | Unconstrained row-stochastic `H, T` | Soft `p(s)` | Isolates acceptance effects |
| `M3` | Counting-restricted `{stay, move}` with state-dependent `q_H(s), q_T(s)` | Soft `p(s)` | Tests the stochastic-counter hypothesis |
| `M4` | Minimal counting monitor with state-independent `q_H, q_T` | Soft `p(s)` | Smallest non-trivial counting baseline |

Dynamic monitoring is implemented in two forms:

- `DynamicMonitor`: full step-indexed transitions `(H_i, T_i)`
- `SharedDynamicMonitor`: shared-parameter dynamic monitor for larger `n`

## Current Architecture

### Core design

The architecture is centred on a small set of stable interfaces:

- `MonitorBase.transitions(n)` returns static `(k, k)` or dynamic `(n, k, k)`
  row-stochastic transitions
- `MonitorBase.acceptance()` returns per-state acceptance probabilities
- `MonitorBase.initial()` returns the initial state distribution
- `compute_error(H, T, p, n)` evaluates exact finite-`n` error

This keeps modelling, optimisation, and analysis loosely coupled.

### Repository layout

```text
src/
  core/
    recursion.py          exact differentiable forward recursion for f(n, k)
    loss.py               training-time vs report-time evaluation helpers
    boundary_checks.py    proposal sanity checks and boundary conditions
  monitors/
    base.py               common monitor interface
    m1_unconstrained.py   static M1
    m2_soft_accept.py     static M2
    m3_counting.py        static M3
    m4_minimal.py         static M4
    dynamic.py            DynamicMonitor + SharedDynamicMonitor
  optim/
    gradient.py           multi-restart Adam training with report-time stats
    genetic.py            GA cross-check for M1
    exhaustive.py         coarse exhaustive search for small (n, k)
    anneal.py             temperature-annealing probe for H2
  analysis/
    heatmap.py            transition heatmaps
    posterior_scatter.py  state-vs-posterior visualisation
    tridiagonality.py     tridiagonality score
    class_gaps.py         Delta^acc / Delta^count / Delta^minimal
    diagnostics.py        entropy + monotonicity + tridiagonality summaries
    fitting.py            bonus curve fitting over empirical f(n, k)
  baselines/
    mk_random_walk.py     Kontorovich M(k) baseline
experiments/
  common.py               shared result schema and artifact writers
  phase1_exploratory.py   exploratory static runs + optional GA/exhaustive
  phase2_core.py          core static + dynamic comparison
  phase3_stretch.py       stretch grid with SharedDynamicMonitor
  reproduce_all.py        Python entrypoint for the full pipeline
results/
  csv/                    experiment tables and diagnostic tables
  figures/                heatmaps and posterior scatter plots
  logs/                   per-run JSONL logs
tests/
  test_boundaries.py
  test_recursion.py
  test_nesting.py
  test_report_eval.py
  test_phase_outputs.py
  test_reproduce_pipeline.py
scripts/
  reproduce.sh            shell wrapper for the reproducibility pipeline
```

## Training and Evaluation Semantics

One important detail in the current implementation:

- `M1` is trained with a soft relaxation of acceptance for optimisation
  stability.
- Final reported `M1` results are evaluated through a dedicated
  **report-time hard-acceptance path**.

This means experiment CSVs now distinguish between:

- `f_train_best`: best training-time objective
- `f_report`: report-time value used for comparisons and tables

This avoids mixing the training relaxation into the final `M1 vs M2` analysis.

## Installation

### Requirements

- Python `>= 3.10`
- `torch`, `numpy`, `scipy`, `matplotlib`, `pandas`, `pytest`, `pyyaml`

### Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

POSIX shell:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start

### 1. Run the test suite

```bash
python -m pytest -q
```

This covers:

- recursion correctness,
- boundary conditions,
- nesting sanity,
- `M1` report-time hard evaluation,
- experiment output schema,
- reproducibility pipeline smoke checks.

### 2. Run the full reproducibility pipeline

```bash
python -m experiments.reproduce_all
```

or:

```bash
bash scripts/reproduce.sh
```

This writes experiment tables, diagnostics, logs, and figures under `results/`.

### 3. Run a single phase directly

```bash
python -m experiments.phase1_exploratory
python -m experiments.phase2_core
python -m experiments.phase3_stretch
```

## Experiment Entry Points

All phase runners are parameterised Python functions, so you can run a tiny
smoke configuration without editing source files.

### Phase 1: exploratory static grid

Default grid:

- `n in {11, 21}`
- `k in {2, 3}`
- static classes `M1-M4`
- optional GA, exhaustive search, and annealing

Example smoke run:

```python
from experiments.phase1_exploratory import main

main(
    output_dir="results/smoke_phase1",
    n_values=[11],
    k_values=[2],
    restarts=1,
    steps=1,
    run_ga=False,
    run_exhaustive=False,
)
```

### Phase 2: core results

Default grid:

- `n in {51, 101}`
- `k in {2, 3, 5}`
- static `M1-M4`
- dynamic `DynamicMonitor`

### Phase 3: stretch results

Default grid:

- `n = 201`
- `k in {10, 20}`
- static `M1-M4`
- dynamic `SharedDynamicMonitor`

## Outputs

### Directory structure

Each phase writes to:

- `results/csv/phaseX.csv`
- `results/csv/phaseX_diagnostics.csv`
- `results/logs/phaseX.jsonl`
- `results/figures/phaseX/*.png`

The full reproducibility pipeline also writes:

- `results/csv/fits.csv`

### Main result table schema

The shared experiment schema includes:

- run identifiers: `phase`, `n`, `k`, `class`
- optimisation summaries:
  `f_train_best`, `f_report`, `f_mean`, `f_std`, `ci95_low`, `ci95_high`,
  `restarts`, `restart_losses_json`
- class-gap diagnostics:
  `delta_acc`, `delta_count`, `delta_minimal`, `delta_dynamic`
- baseline / auxiliary search:
  `baseline_f`, `baseline_gap`, `ga_f`, `ga_gap`, `exhaustive_f`,
  `exhaustive_gap`
- annealing outputs:
  `anneal_loss_tau_1_0`, `anneal_loss_tau_0_5`, `anneal_loss_tau_0_1`,
  `anneal_loss_tau_0_01`, `anneal_loss_tau_0_001`
- structure diagnostics:
  `tridiagonality`, `transition_entropy_mean`, `transition_entropy_min`,
  `transition_entropy_max`, `posterior_monotonicity_mean`,
  `posterior_monotonicity_min`
- artifact paths:
  `heatmap_path`, `posterior_scatter_path`

### Diagnostics CSV

The per-phase diagnostics table is a compact view of:

- tridiagonality,
- transition entropy,
- posterior monotonicity.

### Figures

For each trained monitor the pipeline can emit:

- transition heatmaps,
- posterior scatter plots.

Dynamic monitors use a dedicated dynamic heatmap view that samples
representative time steps.

## Analysis and Diagnostics

The current structural analysis stack includes:

- **Class gaps**:
  `Delta^acc`, `Delta^count`, `Delta^minimal`
- **Dynamic gap**:
  `Delta(n, k) = f_static - f_dynamic`
- **Tridiagonality**:
  how close a transition matrix is to a bounded random walk
- **Transition entropy**:
  whether a trained monitor behaves deterministically or diffusely
- **Posterior monotonicity**:
  whether state indices track posterior order over time
- **Posterior scatter plots**:
  qualitative inspection of state-to-posterior structure
- **Curve fitting**:
  bonus analytic fits over empirical `M1` results

## Reproducibility Notes

The pipeline is designed to be reproducible at the artifact level:

- seeds are logged per restart,
- CSVs store aggregate statistics and restart loss lists,
- logs store per-run metadata,
- figures are generated from the trained monitor selected for reporting,
- `experiments.reproduce_all` is the single Python entrypoint that ties the
  phases together.

## Testing

Run everything:

```bash
python -m pytest -q
```

Useful subsets:

```bash
python -m pytest tests/test_boundaries.py -q
python -m pytest tests/test_recursion.py -q
python -m pytest tests/test_report_eval.py -q
python -m pytest tests/test_phase_outputs.py -q
python -m pytest tests/test_reproduce_pipeline.py -q
```

## Development Notes

- The project currently uses `float64` for the core recursion because the
  error differences relevant to nesting checks can be very small.
- `matplotlib` is configured for headless figure generation, so the plotting
  pipeline works in non-GUI environments.
- `Phase 1` currently includes the optional hooks for GA and exhaustive search;
  `Phase 2` includes optional GA; `Phase 3` focuses on the stretch dynamic run.

## Licence

MIT. See `LICENSE`.

## Acknowledgements

Claude (Anthropic) was used as a research assistant for literature discussion
and structural outlining. All code and text are the author's own work.
