# Execution Inputs

## Cohort source files

`analyze` reads `hrs.csv`, `elsa.csv`, and `charls.csv` from `--source-dir`. Each file has one row per participant and survey wave. Participant-wave pairs must be unique. Use consecutively numbered waves and harmonized numeric variables. The loader recognizes each column by the stem preceding ` (` and retains the first occurrence of a duplicated stem.

The input convention uses `iwstat = 0` for a living interview and `iwstat = 1` for a death interview. Daily-activity variables use 0 for no difficulty and 1 for difficulty. Hip-fracture variables use 0 and 1. Missing observations remain missing. Convert study-specific response codes to these conventions before running the analysis.

Core fields are the participant identifier, wave, interview year, interview-survival indicator, four common task variables, cohort fracture fields, and age or birth year. ELSA requires both `hip` and `hipe`. Covariates and death-date fields can contain missing observations; unavailable covariates are represented by missing values and evaluated by the model's availability criterion.

| Variable group | HRS | ELSA | CHARLS |
| --- | --- | --- | --- |
| Participant identifier | `hhidpn` | `idauniqc` | `ID` |
| Wave | `wave` | `wave` | `wave` |
| Interview year, month | `iwendy`, `iwendm` | `iwindy`, `iwindm` | `iwy`, `iwm` |
| Interview survival | `iwstat` | `iwstat` | `iwstat` |
| Age, birth year | `ragey_e`, `rabyear` | `agey`, `rabyear` | `age`, `rabyear` |
| Hip fracture | `hipe` | `hip` for the report interval; `hipe` for prior history | `hip` |
| Four common activities | `dressa`, `eata`, `beda`, `toilta` | `dressa`, `eata`, `beda`, `toilta` | `dressa`, `eata`, `beda`, `toilta` |
| Sex, education | `ragender`, `raeducl` | `ragender`, `raeducl` | `ragender`, `raeducl` |
| Body mass index, self-rated health | `bmi`, `shlt` | `mbmi` with `bmi` fallback, `shlt` | `bmi`, `srh` |
| Smoking, marital status | `smokev`, `mstath` | `smokev`, `mstath` | `smokev`, `marry` |
| Cognition | `cogtot` | `tcog_z_z` | `total_cognition` only |
| Frailty | `frailty` | `frailty` | `frailtya` only |
| Instrumental activities | `iadl5a` | `iadltot2_e` | `iadl` |
| Depressive symptoms | `cesd` | `cesd` | `cesd10` |
| Income, wealth | `itot`, `atotb` | `itearn`, `hatotb` | `income_total`, `hhcperc` |
| Region | `cenreg` | Region is excluded from this cohort model | `region` |
| Death date | `radyear`, `radmonth` | `radyear`, `radmonth` | `radyear`, `radmonth` |

CHARLS continuous cognition and frailty index A remain separate from `cog_status` and `frailtyb`. Missing primary values are preserved for training-fold imputation; source alternatives do not replace them.

The shared chronic-disease stems are `diabe`, `hearte`, `stroke`, `cancre`, `lunge`, `arthre`, and `hibpe`. The harmonizer also recognizes the supporting variables enumerated in `core/harmonize.py`. Variables available in at least 10% of the landmark sample enter the corresponding cohort model. Median imputation and scaling are fitted within each training fold; missing indicators accompany imputed variables. Survey wave, education, region, and marital-status codes enter as categorical variables. `ragender` follows the harmonized 0/1 convention used by the input extracts. CESD retains its cohort-specific scale: 0-8 in HRS and ELSA and 0-30 in CHARLS.

Interview and death dates preserve their month or year precision. A living interview supplies the observed functional record. Death interviews use their survival indicator independently of any retrospective task answers. The landmark code requires two consecutive prior interviews, pre-interval age of at least 65 years, observed fracture-free prior history, and a reporting interval of 9-54 months. Next-visit function uses the consecutive living interview 18-42 months after reporting; the second follow-up uses the implemented 36-84 month window.

## Reusable panel-example input files

The 32 reusable panel examples read these four CSV files from `--input-dir`. `analyze` creates them under `figure_inputs` in its output directory. The `panel` column identifies the displayed observations. `cohort` uses `HRS`, `ELSA`, or `CHARLS` wherever a cohort is involved.

| File | Panel values | Displayed fields |
| --- | --- | --- |
| `Figure1_functional_burden.csv` | `Multiple daily activities affected`, `b` | `risk_difference`, `lo`, `hi`, `case_risk`, `counterfactual_reference_risk` |
| Same file | `c`, `d`, `e` | `state`, `difference` for all 16 joint states |
| Same file | `f` | `arm`, `threshold`, `risk` |
| Same file | `g` | `analysis`, `risk_difference` for the four common tasks |
| Same file | `h` | `arm`, `independent_multi`, `multi_risk` |
| `Figure2_persistence.csv` | `a`, `b`, `c` | `lag`, `task`, `risk`, `n` |
| Same file | `HRS turnover`, `ELSA turnover`, `CHARLS turnover` | `loss`, `recovery`, `n`, `fraction`, `denominator` |
| Same file | `gh` | `count`, `later_multiple`, `same_state_percent`, `denominator` |
| `Figure3_joint_composition.csv` | `HRS multiple`, `ELSA multiple`, `CHARLS multiple` | `component`, `value`, `lo`, `hi` |
| Same file | `d` | `draw`, `association_pp` |
| Same file | `e` | `coalition`, `value` |
| Same file | `f` | `method`, `scenario`, `replicate`, `estimate`, `truth` |
| Same file | `g` | `method`, `scenario`, `rmse` |
| Same file | `h` | `component`, `scenario`, `coverage` |
| `Figure4_capacity_opportunity.csv` | `A`, `C`, `E` | `method`, `budget`, `risk` |
| Same file | `A uncertainty`, `C uncertainty`, `E uncertainty` | `budget`, `lo`, `hi` |
| Same file | `B`, `D`, `F` | `q`, `budget`, `risk` |
| Same file | `G` | `budget`, `independent_minus_comonotonic_risk` |
| Same file | `H` | `q`, `minimum_attempt_budget` |

Figure 4B, 4D, and 4F also require `scenario = Equal attempt costs and unrestricted task capacity`, `dependence = Independent successes`, and `policy = Joint subset optimization`. Figure 4G requires the same `scenario` and `policy`, with `q = 0.5`. Figure 4H requires the same `scenario`, `dependence = Independent successes`, and the Boolean field `attainable = True`. Retain these fields when supplying an external figure table.

Task indices 0, 1, 2, and 3 denote dressing, eating, bed transfer, and toileting. The trajectory table stores their full English task labels. State index `s` has bit `j` set when task `j` has difficulty. The 4 x 4 state heatmaps reshape the increasing state indices in row-major order. MOSAIC coalition bits 0-4 encode dressing, eating, bed transfer, toileting, and joint association.

Risks, risk differences, coverage, and component values use probability units unless stated otherwise. Figure 1F `risk`, Figure 2H `same_state_percent`, and Figure 3D `association_pp` are already in percent or percentage-point units. Figure 2G `later_multiple` uses probability units. Plotting code performs the corresponding conversions explicitly.

Figure 2 task histories retain the number of observed values separately for each task and visit. Exact state persistence counts identical four-task patterns across visits. Empty initial-state groups retain a denominator of zero and the implemented zero-valued display. RIFT target curves retain attainable targets; the underlying target calculation reports feasibility for all evaluated probabilities.

The article's four nine-panel main figures and supplementary displays use the complete production programs in `Reproducibility_Code.zip`. Group numbers in the example table above refer to the reusable example interfaces.

## Joint-state input for resource planning

`rift` accepts a directory containing `joint_states.csv`. The required fields are `cohort`, `state`, `fracture`, and `case_count`. Each cohort has state indices 0-15 exactly once. `case_count` contains finite, nonnegative integers with a positive cohort total. `fracture` is the observed case-state probability and equals `case_count` divided by the cohort's total case count. Probabilities sum to one. The case counts determine the multinomial portfolio uncertainty calculation.

## Outputs created by execution

The output directory contains harmonized person-wave and landmark records, fitted scores, aggregate estimates, multiplier arrays, benchmark results, resource frontiers, four figure-input tables, and execution figures. Person-level products remain under the chosen output directory. The example figure command consumes the four aggregate tables, and the benchmark command generates its own seeded simulation inputs.

## Task transport and analytic recovery interfaces

`minimum_transport(source, target)` requires two finite, nonnegative vectors of length 16 whose probabilities sum to one. `paired_movement(table)` requires a finite, nonnegative 16-by-16 table with positive total mass. It accepts counts or masses, normalizes by the total, and retains row/column state meanings.

`half_risk_identity(probabilities, case_counts=None)` uses the same four-task state convention. Optional counts must be nonnegative integers with positive total and imply the same probabilities. The returned condition applies to independent 50% successes, equal unit costs, unrestricted capacity, and sufficient budget. `verify_half_risk_identities(portfolios)` accepts an iterable of valid probability vectors and returns numerical comparisons. These functions have no file-system or project-directory dependencies.
