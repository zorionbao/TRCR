# Core Methods and Reusable Panel Examples

Code for *Task Recovery and Coordinated Resource Planning After Hip Fracture*.

This MIT-licensed distribution contains the core cohort, MOSAIC, RIFT, and task-space calculations, numerical tests, and 32 reusable panel examples. Four binary activities define 16 functional states: dressing, eating, bed transfer, and toileting.

## Installation

Use Python 3.12.10 and the versions in `requirements.txt`.

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` in a POSIX shell, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The plotting examples use Liberation Sans and the Okabe–Ito palette. A directory containing Liberation Sans font files can be supplied with `--font-dir`.

## Core cohort analysis

Prepare `hrs.csv`, `elsa.csv`, and `charls.csv` in the long-format convention specified in [INPUTS.md](INPUTS.md).

```bash
python run.py analyze --source-dir inputs/cohorts --output-dir outputs
```

The command harmonizes observations, preserves participant-level folds, fits functional and task-specific contrasts, estimates joint states, computes MOSAIC, summarizes paired turnover, runs known-truth calibration, and solves RIFT. It creates aggregate tables for the reusable panel examples. Inputs are supplied explicitly; generated files stay under the chosen output directory.

| Module | Calculation |
| --- | --- |
| `core/harmonize.py` | Common person-wave variables, living interviews, and death-date intervals |
| `core/clinical.py`, `core/domains.py` | Five participant folds and standardized functional contrasts |
| `core/mosaic.py`, `core/joint_decomposition.py` | Full-state standardization and exact five-player Shapley attribution |
| `core/benchmark.py` | Three known-generating mechanisms, 200 samples per mechanism |
| `core/turnover.py` | Observed new and resolved task difficulties |
| `core/rift.py`, `core/recovery_frontier.py`, `core/probabilistic_restoration.py` | Deterministic and probabilistic attempt-resource optimization |
| `core/recovery_identities.py` | Exact 50%-success half-risk identities and comparisons with the allocation programs |
| `core/task_transport.py` | Minimum Hamming transport, observed paired movement, and information summaries |
| `core/figure_inputs.py` | Aggregate input tables for reusable panel examples |

CHARLS cognition retains only its continuous `total_cognition` score, and frailty retains index `frailtya`. Missing values stay missing before training-fold imputation. The binary `cog_status` and alternative `frailtyb` index have distinct meanings and are not substituted into those measures.

Primary settings include five participant folds, cohort-specific seeds, 1,000 MOSAIC multiplier draws, and 0.5% uniform smoothing. The benchmark has 1,200 reference and 400 case observations per sample. Deterministic frontier uncertainty uses 1,000 multinomial portfolio draws. Planning success, attempt costs, and capacities are explicit scenario inputs.

## Portable mathematical interfaces

`core.task_transport.minimum_transport(source, target)` accepts two 16-state probability vectors and returns a minimum cost, one optimal coupling, and a primal/dual certificate. `paired_movement(table)` accepts a 16-by-16 nonnegative transition-count or mass array. It compares observed paired turnover with the minimum distributional movement and returns entropy, persistence, and information summaries. Multiple couplings can share a minimum; the observed transition table identifies the measured within-person changes.

`core.recovery_identities.half_risk_identity(probabilities, case_counts=None)` returns the independent-success minimum risk and half-risk attainability, plus the shared-uniform half-risk budget. The identities assume success probability 0.5, equal unit attempt costs, unrestricted task capacity, and sufficient total budget. Supplying integer counts preserves the exact feasibility condition. `verify_half_risk_identities(portfolios)` compares the formulas with the allocation programs for any supplied collection. The test suite uses 14 synthetic portfolios; all test cases are generated in code.

The full study archive contains the exploratory neural sensitivity, observed-score subgroup analyses, follow-up-window sensitivities, service-geography inputs, and final figure assembly programs.

## Reusable panel examples

```bash
python run.py figures --input-dir inputs/figures --output-dir outputs
python -m figures.figure1.panel_a --input-dir inputs/figures --output-dir outputs
python -m figures.figure4.panel_h --input-dir inputs/figures --output-dir outputs
```

Each module contains an independent `draw` function. The batch command exports 32 examples and four example assemblies; individual commands export one example. Append `--font-dir inputs/fonts` when needed. The example input tables and units are specified in [INPUTS.md](INPUTS.md); separate example legends are in `legends`.

| Example group | Modules | Content |
| --- | --- | --- |
| 1 | A–H | Burden, state shifts, severity, task contrasts, state dependence |
| 2 | A–H | Task histories, turnover, subsequent burden, state persistence |
| 3 | A–H | Attribution, multiplier distributions, coalitions, benchmark performance |
| 4 | A–H | Restoration, success dependence, capacity, and target budgets |

## Tests and numerical commands

```bash
python run.py test --output-dir outputs
python run.py benchmark --output-dir outputs
python run.py rift --input-dir inputs/joint_states --output-dir outputs
```

The RIFT input is an externally supplied `joint_states.csv`. Tests cover probability conservation, Shapley and derivative identities, complete success kernels, attempt resources, analytic targets, transport solutions, source-scale preservation, and execution interfaces. Tests generate their numerical cases in code. Test and execution files are written under the selected output directory.

## Data access and distribution

Obtain controlled cohort records through the [HRS portal](https://hrsdata.isr.umich.edu/), [ELSA access service](https://www.elsa-project.ac.uk/accessing-elsa-data), and [CHARLS portal](https://charls.charlsdata.com/). Each study's terms govern its records. [INPUTS.md](INPUTS.md) describes the variables required by these core methods.

The distribution consists of source code, tests, English documentation, example legends, dependency specifications, checksums, and the MIT license. Data, estimated results, images, execution products, and dependencies are supplied or generated separately. `.gitignore` excludes those files from publication.
