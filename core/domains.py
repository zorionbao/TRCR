"""Case-standardized risk differences for four daily activities."""

from pathlib import Path

import pandas as pd

from . import clinical


def run(source_dir: Path, output_dir: Path) -> None:
    """Write next-interview task contrasts using the cohort estimator."""
    rows = []
    for cohort, seed in clinical.SEEDS.items():
        landmarks, _ = clinical.build(cohort, source_dir, output_dir)
        for task in clinical.CORE:
            observed = landmarks.loc[landmarks.obs1 & landmarks["n1_" + task].notna()].copy()
            observed["outcome"] = observed["n1_" + task]
            result, _ = clinical.estimate(observed, "outcome", seed, task,
                                          endpoint=task, specification="task_specific")
            if result is None:
                raise ValueError(f"{cohort} has fewer than 15 records in an exposure group")
            rows.append(result)
            print(f"{cohort} {task}: risk difference={result['risk_difference']:.12g}", flush=True)
    destination = Path(output_dir) / "stats" / "clinical"
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination / "clinical_domains.csv", index=False)
