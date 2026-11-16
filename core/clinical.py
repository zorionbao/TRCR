"""Reporting landmarks and participant-cross-fitted functional contrasts."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
from .harmonize import base


CORE = ["adl_dress", "adl_eat", "adl_bed", "adl_toilet"]
COV = ["age_at_iw", "male", "edu3", "bmi", "srh", "smokev", "married", "cognition",
       "frailty", "diab", "heart", "stroke", "cancer", "lung", "arthr", "hibp", "iadl",
       "cesd", "income", "wealth", "region"]
SEEDS = {"HRS": 2026092201, "ELSA": 2026092202, "CHARLS": 2026092203}


def build(cohort: str, source_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Select reporting landmarks with two preceding fracture-free interviews."""
    panel_path = Path(output_dir) / "harmonized" / f"panel_{cohort.lower()}.csv"
    frame = (pd.read_csv(panel_path, dtype={"id": str}, low_memory=False)
             .sort_values(["id", "wave"]).reset_index(drop=True))
    frame["core4"] = frame[CORE].sum(axis=1, min_count=4)
    frame["ever"] = frame["hip_ever"]
    if cohort == "ELSA":
        source = pd.read_csv(Path(source_dir) / "elsa.csv", low_memory=False,
                             usecols=lambda column: base(column) in
                             ["idauniqc", "wave", "hip", "hipe"])
        source.columns = [base(column) for column in source.columns]
        source = source.loc[:, ~source.columns.duplicated()]
        source["id"] = source.idauniqc.astype(str)
        frame = frame.merge(source[["id", "wave", "hip", "hipe"]],
                            on=["id", "wave"], validate="one_to_one")
        frame["ever"] = frame.hipe
        frame["recent"] = frame.hip

    grouped = frame.groupby("id", sort=False)
    for lag in [1, 2]:
        for column in ["t", "wave", "core4", "ever", "living_interview"] + CORE + COV:
            if column in frame:
                frame[f"p{lag}_{column}"] = grouped[column].shift(lag)
    for lead in [1, 2]:
        for column in ["t", "wave", "core4", "living_interview"] + CORE:
            frame[f"n{lead}_{column}"] = grouped[column].shift(-lead)
    frame["prior_positive"] = grouped["ever"].transform(
        lambda values: values.eq(1).cummax().shift(fill_value=False))
    current = frame.recent if cohort == "ELSA" else frame.ever
    frame["a"] = current.eq(1).astype(int)
    criteria = [
        ("source_rows", pd.Series(True, index=frame.index)),
        ("two_prior_consecutive_interviews",
         (frame.wave - frame.p1_wave == 1) & (frame.p1_wave - frame.p2_wave == 1)),
        ("preinterval_age_65plus", frame.p1_age_at_iw >= 65),
        ("observed_preinterval_function",
         frame.p1_core4.notna() & frame.p2_core4.notna() &
         frame.p1_living_interview.eq(True) & frame.p2_living_interview.eq(True)),
        ("observed_fracture_free_history", frame.p1_ever.eq(0) & ~frame.prior_positive),
        ("observed_current_fracture", current.isin([0, 1])),
        ("observed_current_function", frame.core4.notna() & frame.living_interview),
        ("positive_valid_interval", (frame.t - frame.p1_t).between(.75, 4.5)),
    ]
    mask = pd.Series(True, index=frame.index)
    flow = []
    for label, criterion in criteria:
        mask &= criterion
        flow.append({"cohort": cohort, "step": label, "rows": int(mask.sum()),
                     "persons": int(frame.loc[mask, "id"].nunique()),
                     "cases": int(frame.loc[mask, "a"].sum())})
    landmarks = frame.loc[mask].copy()
    for visit in [1, 2]:
        landmarks[f"dt{visit}"] = landmarks[f"n{visit}_t"] - landmarks.t
        lower, upper = (1.5, 3.5) if visit == 1 else (3.0, 7.0)
        landmarks[f"obs{visit}"] = (
            (landmarks[f"n{visit}_wave"] - landmarks.wave == visit) &
            landmarks[f"dt{visit}"].between(lower, upper) &
            landmarks[f"n{visit}_core4"].notna() &
            landmarks[f"n{visit}_living_interview"].eq(True))
    landmarks["cohort"] = cohort
    for column in COV:
        if f"p1_{column}" in landmarks:
            landmarks["h_" + column] = landmarks[f"p1_{column}"]
    landmarks["h_core4"] = landmarks.p1_core4
    landmarks["h_slope"] = (landmarks.p1_core4 - landmarks.p2_core4) / (landmarks.p1_t - landmarks.p2_t)
    landmarks["h_year"] = landmarks.p1_t
    landmarks["h_age2"] = ((landmarks.p1_age_at_iw - 75) / 10) ** 2
    landmarks["h_interval"] = landmarks.t - landmarks.p1_t
    return landmarks, flow


def design(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct pre-interval predictors in their recorded column order."""
    columns = [column for column in frame if column.startswith("h_") and
               frame[column].notna().mean() >= .10]
    matrix = frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    for column in ["h_income", "h_wealth"]:
        if column in matrix:
            matrix[column] = np.sign(matrix[column]) * np.log1p(np.abs(matrix[column]))
    categorical = ["wave"] + [column for column in ["h_edu3", "h_region", "h_married"]
                              if column in matrix]
    for column in categorical:
        values = frame[column] if column == "wave" else matrix.pop(column)
        indicators = pd.get_dummies(values.fillna(-99).astype(str), prefix=column, dtype=float)
        matrix = pd.concat([matrix, indicators], axis=1)
    return matrix


def pipeline():
    """Create training-fold imputation, scaling, and regularized regression."""
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=1, max_iter=1500, solver="lbfgs", tol=1e-7),
    )


def estimate(frame: pd.DataFrame, yname: str, seed: int, label: str,
             endpoint: str | None = None, specification: str | None = None,
             followup_visit: int = 1):
    """Estimate the augmented inverse-odds contrast and participant scores."""
    frame = frame.loc[frame[yname].notna()].reset_index(drop=True)
    exposure = frame.a.to_numpy(int)
    outcome = frame[yname].to_numpy(float)
    if min(exposure.sum(), (1 - exposure).sum()) < 15:
        return None, None
    matrix = design(frame)
    identifiers = frame.id.astype(str).to_numpy()
    unique = np.unique(identifiers)
    random = np.random.default_rng(seed)
    unique = random.permutation(unique)
    assignment = {identifier: index % 5 for index, identifier in enumerate(unique)}
    folds = np.array([assignment[identifier] for identifier in identifiers])
    propensity = np.zeros(len(frame))
    reference_model = np.zeros(len(frame))
    with threadpool_limits(limits=1):
        for fold in range(5):
            train = folds != fold
            test = ~train
            exposure_model = pipeline()
            exposure_model.fit(matrix.loc[train], exposure[train])
            propensity[test] = exposure_model.predict_proba(matrix.loc[test])[:, 1]
            outcome_model = pipeline()
            controls = train & (exposure == 0)
            if len(np.unique(outcome[controls])) < 2:
                reference_model[test] = outcome[controls].mean()
            else:
                outcome_model.fit(matrix.loc[controls], outcome[controls])
                reference_model[test] = outcome_model.predict_proba(matrix.loc[test])[:, 1]
    unbounded = propensity.copy()
    propensity = np.clip(propensity, .001, .8)
    odds_weight = (1 - exposure) * propensity / (1 - propensity)
    cases = exposure.sum()
    weight_sum = odds_weight.sum()
    residual = outcome - reference_model
    case_residual = np.sum(exposure * residual) / cases
    control_residual = np.sum(odds_weight * residual) / weight_sum
    difference = case_residual - control_residual
    influence = (exposure * (residual - case_residual) / cases -
                 odds_weight * (residual - control_residual) / weight_sum)
    cluster = pd.Series(influence).groupby(identifiers).sum().to_numpy()
    standard_error = np.sqrt(np.square(cluster).sum() * len(cluster) / (len(cluster) - 1))
    case_risk = outcome[exposure == 1].mean()
    endpoint = endpoint or label
    median_followup = float(frame[f"dt{followup_visit}"].median()) if followup_visit else 0.0
    result = dict(
        cohort=frame.cohort.iloc[0], analysis=label, endpoint=endpoint,
        specification=specification or label, followup_visit=followup_visit,
        n=len(frame), persons=len(unique), cases=int(cases), case_risk=case_risk,
        counterfactual_reference_risk=case_risk - difference, risk_difference=difference,
        se=standard_error, lo=difference - 1.95996398454 * standard_error,
        hi=difference + 1.95996398454 * standard_error,
        p=2 * norm.sf(abs(difference / standard_error)), crude_case=case_risk,
        crude_reference=outcome[exposure == 0].mean(),
        or_weight_ess=float(weight_sum * weight_sum / np.square(odds_weight).sum()),
        max_propensity=float(unbounded.max()), trimmed=int(np.sum(unbounded != propensity)),
        median_followup=median_followup,
        case_mean_age=float(frame.loc[exposure == 1, "p1_age_at_iw"].mean()),
        covariates="|".join(matrix.columns),
    )
    scores = frame[["cohort", "id", "wave", "a"]].copy()
    scores["y"] = outcome
    scores["propensity"] = propensity
    scores["m0"] = reference_model
    scores["influence"] = influence
    scores["control_odds_weight"] = odds_weight
    scores["fold"] = folds
    scores["endpoint"] = endpoint
    return result, scores


def run(source_dir: Path, output_dir: Path) -> None:
    """Write the two severity contrasts and their participant-level scores."""
    harmonized = Path(output_dir) / "harmonized"
    statistics = Path(output_dir) / "stats" / "clinical"
    harmonized.mkdir(parents=True, exist_ok=True)
    statistics.mkdir(parents=True, exist_ok=True)
    frames = []
    results = []
    for cohort, seed in SEEDS.items():
        landmarks, _ = build(cohort, source_dir, output_dir)
        frames.append(landmarks)
        observed = landmarks.loc[landmarks.obs1].copy()
        for threshold, endpoint, label, suffix in [
            (1, "any4", "primary_any4", ""),
            (2, "multiple4", "two_or_more4", "_multiple4"),
        ]:
            observed["outcome"] = observed.n1_core4.ge(threshold).astype(float)
            result, scores = estimate(observed, "outcome", seed, label,
                                      endpoint=endpoint, specification="primary")
            if result is None:
                raise ValueError(f"{cohort} has fewer than 15 records in an exposure group")
            results.append(result)
            scores.to_csv(harmonized / f"clinical_scores{suffix}_{cohort.lower()}.csv", index=False)
            print(f"{cohort} {endpoint}: n={result['n']}, cases={result['cases']}, "
                  f"risk difference={result['risk_difference']:.12g}", flush=True)
    effects = pd.DataFrame(results)
    effects["multiplicity_family"] = "six_cohort_by_severity"
    effects["p_Holm_severity_family"] = multipletests(effects.p, method="holm")[1]
    effects.to_csv(statistics / "clinical_effects.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(harmonized / "clinical_landmarks.csv", index=False)
