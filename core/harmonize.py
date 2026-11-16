"""Harmonize cohort extracts into longitudinal functional panels."""

from pathlib import Path

import numpy as np
import pandas as pd


COHORT_COLUMNS = {
    "HRS": {
        "id": "hhidpn", "wave": "wave", "iyear": "iwendy", "imonth": "iwendm",
        "iwstat": "iwstat", "age": "ragey_e", "male": "ragender",
        "birthyear": "rabyear", "edu3": "raeducl", "adl": "adl6a", "iadl": "iadl5a",
        "hip_ever": "hipe", "dyear": "radyear", "dmonth": "radmonth", "region": "cenreg",
        "srh": "shlt", "bmi": "bmi", "smokev": "smokev", "frailty": "frailty",
        "cognition": "cogtot", "married": "mstath", "adl_dress": "dressa",
        "adl_eat": "eata", "adl_bed": "beda", "adl_toilet": "toilta",
        "diab": "diabe", "heart": "hearte", "stroke": "stroke", "cancer": "cancre",
        "lung": "lunge", "arthr": "arthre", "hibp": "hibpe", "cesd": "cesd",
        "income": "itot", "wealth": "atotb",
    },
    "ELSA": {
        "id": "idauniqc", "wave": "wave", "iyear": "iwindy", "imonth": "iwindm",
        "iwstat": "iwstat", "age": "agey", "male": "ragender", "birthyear": "rabyear",
        "edu3": "raeducl", "adl": "adltot6", "iadl": "iadltot2_e", "hip_ever": "hip",
        "hip_ever_dr": "hipe", "dyear": "radyear", "dmonth": "radmonth", "region": None,
        "srh": "shlt", "bmi": "mbmi", "smokev": "smokev", "frailty": "frailty",
        "cognition": "tcog_z_z", "married": "mstath", "adl_dress": "dressa",
        "adl_eat": "eata", "adl_bed": "beda", "adl_toilet": "toilta",
        "diab": "diabe", "heart": "hearte", "stroke": "stroke", "cancer": "cancre",
        "lung": "lunge", "arthr": "arthre", "hibp": "hibpe", "cesd": "cesd",
        "income": "itearn", "wealth": "hatotb",
    },
    "CHARLS": {
        "id": "ID", "wave": "wave", "iyear": "iwy", "imonth": "iwm", "iwstat": "iwstat",
        "age": "age", "male": "ragender", "birthyear": "rabyear", "edu3": "raeducl",
        "adl": "adlab_c", "iadl": "iadl", "hip_ever": "hip", "dyear": "radyear",
        "dmonth": "radmonth", "region": "region", "srh": "srh", "bmi": "bmi",
        "smokev": "smokev", "frailty": "frailtya", "cognition": "total_cognition",
        "married": "marry", "adl_dress": "dressa", "adl_eat": "eata", "adl_bed": "beda",
        "adl_toilet": "toilta", "diab": "diabe", "heart": "hearte", "stroke": "stroke",
        "cancer": "cancre", "lung": "lunge", "arthr": "arthre", "hibp": "hibpe",
        "cesd": "cesd10", "income": "income_total", "wealth": "hhcperc",
    },
}
FALLBACK_COLUMNS = {
    "HRS": {},
    "ELSA": {"hip_ever": "hipe", "bmi": "bmi"},
    # The continuous cognition score and frailty index A each retain one scale.
    # A binary cognition status and frailty index B are different constructs.
    "CHARLS": {},
}


def base(column: str) -> str:
    """Return the variable stem preceding a parenthetical label."""
    return column.split(" (")[0].strip()


def load(cohort: str, source_dir: Path) -> pd.DataFrame:
    """Read the variables used in the functional analysis."""
    mapping = COHORT_COLUMNS[cohort]
    keys = {value for value in mapping.values() if value is not None}
    keys.update(FALLBACK_COLUMNS[cohort].values())
    frame = pd.read_csv(Path(source_dir) / f"{cohort.lower()}.csv",
                        low_memory=False, usecols=lambda column: base(column) in keys)
    frame = frame.rename(columns={column: base(column) for column in frame.columns})
    return frame.loc[:, ~frame.columns.duplicated()]


def numeric(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is not None and column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def build(cohort: str, source_dir: Path) -> pd.DataFrame:
    """Apply cohort variable mappings and source-specific fallbacks."""
    frame = load(cohort, source_dir)
    mapping = COHORT_COLUMNS[cohort]
    columns = {"cohort": cohort, "id": frame[mapping["id"]].astype(str)}
    for output, source in mapping.items():
        if output == "id":
            continue
        values = numeric(frame, source)
        fallback = FALLBACK_COLUMNS[cohort].get(output)
        if fallback is not None:
            values = values.where(values.notna(), numeric(frame, fallback))
        columns[output] = values
    return pd.DataFrame(columns)


def finalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct interview times, living observations, and death-date intervals."""
    frame = frame.sort_values(["id", "wave"]).reset_index(drop=True)
    frame["iyear"] = frame["iyear"].astype(float)
    frame["imonth_raw"] = frame["imonth"]
    interview_month = frame["imonth"].between(1, 12) & frame["imonth"].eq(np.floor(frame["imonth"]))
    frame["interview_time_precision"] = np.where(interview_month, "month", "year")
    frame["interview_t_lower"] = np.where(
        interview_month, frame["iyear"] + (frame["imonth"] - 1) / 12, frame["iyear"])
    frame["interview_t_upper"] = np.where(
        interview_month, frame["iyear"] + frame["imonth"] / 12, frame["iyear"] + 1)
    frame["imonth"] = frame["imonth"].fillna(6.0).astype(float)
    frame["t"] = frame["iyear"] + (frame["imonth"] - 0.5) / 12.0
    frame["age_at_iw"] = frame["age"].where(
        frame["age"].notna(), frame["iyear"] - frame["birthyear"])
    if not frame["iwstat"].dropna().isin([0, 1]).all() or frame["iwstat"].isna().all():
        raise ValueError("iwstat must identify living interviews as 0 and death interviews as 1")
    frame["living_interview"] = frame["iwstat"].eq(0) & frame["t"].notna()

    previous = frame.groupby("id")["hip_ever"].shift(1)
    frame["hip_prev"] = previous
    frame["hip_inc"] = (frame["hip_ever"].eq(1) & previous.eq(0)).astype(int)
    first_positive = (frame.loc[frame["hip_ever"].eq(1)]
                      .groupby("id")["wave"].min().rename("hip_wave"))
    frame = frame.merge(first_positive, on="id", how="left")
    frame["post_hip"] = (frame["wave"].ge(frame["hip_wave"]) & frame["hip_wave"].notna()).astype(int)

    valid_year = frame["dyear"].between(1900, 2100) & frame["dyear"].eq(np.floor(frame["dyear"]))
    valid_month = frame["dmonth"].between(1, 12) & frame["dmonth"].eq(np.floor(frame["dmonth"]))
    frame["death_year_raw"] = frame["dyear"]
    frame["death_month_raw"] = frame["dmonth"]
    lower = pd.Series(np.where(valid_month, frame.dyear + (frame.dmonth - 1) / 12,
                               frame.dyear), index=frame.index)
    upper = pd.Series(np.where(valid_month, frame.dyear + frame.dmonth / 12,
                               frame.dyear + 1), index=frame.index)
    source = pd.DataFrame({"id": frame.id, "lower": lower.where(valid_year),
                           "upper": upper.where(valid_year)})
    dates = source.groupby("id").agg(death_t_lower_raw=("lower", "max"),
                                    death_t_upper_raw=("upper", "min"))
    frame = frame.merge(dates, on="id", how="left", validate="many_to_one")
    last_alive = (frame.loc[frame.living_interview].groupby("id")
                  .agg(last_alive_t_lower=("interview_t_lower", "max"),
                       last_alive_t_upper=("interview_t_upper", "max")))
    frame = frame.merge(last_alive, on="id", how="left", validate="many_to_one")
    frame["death_recorded"] = frame.death_t_lower_raw.notna()
    frame["death_source_interval_conflict"] = (
        frame.death_recorded & frame.death_t_lower_raw.ge(frame.death_t_upper_raw))
    frame["death_before_living_interview_conflict"] = (
        frame.death_recorded & frame.death_t_upper_raw.le(frame.last_alive_t_lower))
    frame["death_record_conflict"] = (
        frame.death_source_interval_conflict | frame.death_before_living_interview_conflict)
    frame["death_time_precision"] = np.select(
        [~frame.death_recorded, frame.death_source_interval_conflict,
         (frame.death_t_upper_raw - frame.death_t_lower_raw).lt(.1)],
        ["unrecorded", "conflicting_source_intervals", "month"], default="year")
    frame["death_t_lower"] = frame.death_t_lower_raw.where(~frame.death_record_conflict)
    frame["death_t_upper"] = frame.death_t_upper_raw.where(~frame.death_record_conflict)
    frame["death_t_lower"] = np.fmax(frame.death_t_lower, frame.last_alive_t_lower)
    frame.loc[~frame.death_recorded | frame.death_record_conflict, "death_t_lower"] = np.nan
    frame["adl"] = frame["adl"].clip(lower=0, upper=6)
    return frame


def run(source_dir: Path, output_dir: Path) -> None:
    """Write one harmonized person-wave panel for each cohort."""
    destination = Path(output_dir) / "harmonized"
    destination.mkdir(parents=True, exist_ok=True)
    for cohort in COHORT_COLUMNS:
        panel = finalise(build(cohort, source_dir))
        panel.to_csv(destination / f"panel_{cohort.lower()}.csv", index=False)
        print(f"{cohort}: {len(panel)} person-wave records", flush=True)
