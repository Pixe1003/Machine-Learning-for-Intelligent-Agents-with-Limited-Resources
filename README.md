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

- four static monitor classes (`M1`, `M1-hard`, `M2`, `M3`) under the
  supervisor's renamed hierarchy (see *Monitor Classes* below),
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

> **Naming convention.** The class names below follow the supervisor's
> renaming of the proposal's original `M1`-`M4`, *not* the proposal text.
> The renaming makes `M1` the most general model and orders the rest by how
> much expressiveness each constraint removes, so that `M1 ⊇ M2 ⊇ M3` reads
> as a clean nesting. The source *file* names still carry the old proposal
> numbering (`m1_unconstrained.py`, `m2_soft_accept.py`, ...); the mapping
> table makes the correspondence explicit. **The canonical identifier is the
> class name** — `f_M1` in every CSV, figure, and table refers to the row
> below, never to the file called `m1_*`.

| Class | Source file | Transition family | Acceptance | Free params | Role |
|-------|-------------|-------------------|------------|-------------|------|
| `M1` | `m2_soft_accept.py` (`M1Monitor`) | Unconstrained row-stochastic `H, T` | Soft `p(s)` (sigmoid) | `2k(k-1) + k` | Most general static model; lower bound `f_M1` for the hierarchy |
| `M1-hard` | `m1_unconstrained.py` (`M1HardMonitor`) | Unconstrained row-stochastic `H, T` | Hard binary `F(s) ∈ {0,1}` at report time | `2k(k-1) + k` | Hard-acceptance variant of `M1`; isolates the value of soft acceptance |
| `M2` | `m3_counting.py` (`M2Monitor`) | State-dependent counting `{stay, move}` via `q_H(s), q_T(s)` | Soft `p(s)` | `3k` | Stochastic-counter model; `Δ^count = f_M2 − f_M1` tests **H1** |
| `M3` | `m4_minimal.py` (`M3Monitor`) | Minimal counting with state-independent scalars `q_H, q_T` | Soft `p(s)` | `k + 2` | Smallest non-trivial counting baseline; `Δ^minimal = f_M3 − f_M2` |

**Nesting.** `M1 ⊇ M2 ⊇ M3` as parameter families, so under a consistent
soft evaluation `f_M1 ≤ f_M2 ≤ f_M3`. `M1-hard` is a *subset of* `M1`
(hard acceptance is a restriction of soft) but is **not** part of the
counting hierarchy, so it is tracked separately rather than slotted between
the others.

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
    m1_unconstrained.py   defines M1HardMonitor (unconstrained H,T + hard accept)
    m2_soft_accept.py     defines M1Monitor     (unconstrained H,T + soft accept)
    m3_counting.py        defines M2Monitor     (state-dependent counting + soft)
    m4_minimal.py         defines M3Monitor     (minimal counting + soft)
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
  test_monitor_renaming.py      class-name <-> module-export mapping + semantics
  test_report_eval.py
  test_phase_outputs.py
  test_experiment_entrypoints.py
  test_reproduce_pipeline.py
scripts/
  reproduce.sh            shell wrapper for the reproducibility pipeline
```

## Training and Evaluation Semantics

Two details matter for reading the result tables.

**Training vs report.** `M1-hard` is trained with a soft sigmoid relaxation of
its binary acceptance for optimisation stability, then evaluated through a
dedicated **report-time hard-acceptance path** (`report_evaluate`, which
thresholds the sigmoid at 0.5). The soft-acceptance classes (`M1`, `M2`, `M3`)
report their native soft value. Experiment CSVs therefore distinguish:

- `f_train_best`: best training-time objective
- `f_report`: report-time value used for comparisons and tables

**Dual evaluation.** To make class gaps comparable across classes that were
*selected* under different acceptance criteria, every monitor is additionally
re-scored under both semantics via `evaluate_soft` / `evaluate_hard`
(`src/core/loss.py`):

- `f_report_soft`: the monitor under continuous sigmoid acceptance
- `f_report_hard`: the same monitor with acceptance thresholded at 0.5

Computing `Δ^acc`, `Δ^count`, and `Δ^minimal` from a *consistent* eval mode
(rather than mixing each class's native mode) removes a cross-criterion
artefact that could otherwise make `Δ^acc` appear negative even though the
nesting `M1 ⊇ M1-hard` holds. The consistent-eval gaps are stored as
`delta_acc_soft`, `delta_acc_hard`, `delta_count_hard`, and
`delta_minimal_hard`; the legacy mixed-eval `delta_hard` / `delta_count` /
`delta_minimal` columns are retained for backward compatibility.

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
- the class-naming convention (`M1` / `M1-hard` / `M2` / `M3` map to the
  expected module exports and acceptance semantics),
- `M1-hard` report-time hard evaluation,
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
- `k in {2, 3, 5}` (`k=5` added per the supervisor's request to expose
  class-gap differences that `k=2,3` can mask)
- static classes `M1`, `M1-hard`, `M2`, `M3`
- a mixed cold/warm restart strategy (every `COLD_INTERVAL`-th restart is a
  cold random init; the rest warm-start from the cascade and add a Gaussian
  perturbation, so restarts explore the basin instead of collapsing onto the
  warm-start fixed point)
- optional Week 5 GA cross-check, defaulting to the required `(n=21, k=3, M1)`
  case (`ga_cases=None` runs GA for every grid point)
- optional exhaustive search, annealing, and a separate `(5, 2)` ground-truth
  check

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
    run_ground_truth=False,
)
```

### Phase 1 extended: lightweight richer grid

`experiments.phase1_extended` expands the static grid for exploratory analysis
without replacing the formal Phase 1 table.

Default grid:

- `n in {11, 21, 31}`
- `k in {2, 3, 4}`
- static classes `M1`, `M1-hard`, `M2`, `M3`
- 5 restarts, with GA/exhaustive/ground-truth disabled by default
- summary figures enabled, but per-configuration heatmaps disabled by default

Run it with:

```bash
python -m experiments.phase1_extended
```

This writes under `results/phase1_extended/` and is intended for richer
trend/gap analysis before spending compute on a full 20-restart run.

### Phase 2: core results

Default grid:

- `n in {51, 101}`
- `k in {2, 3, 5}`
- static `M1`, `M1-hard`, `M2`, `M3`
- dynamic `DynamicMonitor`

### Phase 3: stretch results

Default grid:

- `n = 201`
- `k in {10, 20}`
- static `M1`, `M1-hard`, `M2`, `M3`
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

Phase 1 additionally writes:

- `results/csv/phase1_ground_truth.csv`

`phase1.csv` is the formal `{11,21} x {2,3}` static-monitor table.  The
ground-truth CSV is separate so the small `(n=5,k=2)` exhaustive validation
does not appear as another experiment-grid row.

### Main result table schema

The shared experiment schema includes:

- run identifiers: `phase`, `n`, `k`, `class`
- optimisation summaries:
  `f_train_best`, `f_report`, `f_mean`, `f_std`, `ci95_low`, `ci95_high`,
  `restarts`, `restart_losses_json`
- dual-eval report values (same monitor, both acceptance semantics):
  `f_report_soft`, `f_report_hard`
- class-gap diagnostics:
  - legacy mixed-eval: `delta_acc`, `delta_hard`, `delta_count`,
    `delta_minimal`, `delta_dynamic`
  - consistent-eval (from the dual-eval columns): `delta_acc_soft`,
    `delta_acc_hard`, `delta_count_hard`, `delta_minimal_hard`
- baseline / auxiliary search:
  `baseline_f`, `baseline_gap`, `ga_f`, `ga_gap`, `exhaustive_f`,
  `exhaustive_gap`; Phase 1 writes the GA soft-eval result on the `M1` row and
  the corresponding hard-eval result on the `M1-hard` row
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

### Phase 1 ground-truth CSV

`phase1_ground_truth.csv` records the Week 4 exhaustive validation fields:
`n`, `k`, `class`, `gradient_f`, `gradient_mean`, `gradient_std`, `restarts`,
`steps`, `exhaustive_f`, `absolute_gap`, `relative_gap`, `gap_threshold`,
`passed`, and `action`.

By default it checks `M1` at `(n=5,k=2)` with 20 restarts and accepts the run
when the relative gap to exhaustive search is at most 1%.  If the gap is larger,
the runner retries with 30 restarts and `lr=5e-3`; if that still misses the
threshold it runs the GA cross-check and marks the row as `ga_required`.

### Figures

For each trained monitor the pipeline can emit:

- transition heatmaps,
- posterior scatter plots.

The summary figure set also includes:

- absolute grouped monitor bars,
- relative monitor-gap bars against M1,
- restart distribution plots,
- a restart-stability CSV summary.

Dynamic monitors use a dedicated dynamic heatmap view that samples
representative time steps.

## Analysis and Diagnostics

The current structural analysis stack includes:

- **Class gaps** (`src/analysis/class_gaps.py`):
  `Delta^acc`, `Delta^count`, `Delta^minimal`, each available in a
  legacy mixed-eval form and a methodologically clean consistent-eval form
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
