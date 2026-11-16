"""Paired task losses and recoveries between reporting and follow-up."""
from pathlib import Path
import numpy as np
import pandas as pd
from .clinical import CORE, SEEDS


def run(output_dir: Path):
    """Summarize joint task turnover in observed fracture-report records."""
    output_dir = Path(output_dir)
    frame = pd.read_csv(output_dir/'harmonized/clinical_landmarks.csv', dtype={'id': str}, low_memory=False)
    frame = frame.loc[frame.obs1 & frame.a.eq(1)]
    rows = []
    for cohort in SEEDS:
        group = frame.loc[frame.cohort.eq(cohort)]
        before = group[CORE].to_numpy(float)
        after = group[['n1_'+task for task in CORE]].to_numpy(float)
        valid = np.isfinite(before).all(1) & np.isfinite(after).all(1)
        before, after = before[valid], after[valid]
        lost = ((before == 0) & (after == 1)).sum(1)
        recovered = ((before == 1) & (after == 0)).sum(1)
        denominator = len(before)
        if not denominator:
            raise ValueError('No observed paired task records for '+cohort)
        for loss in range(5):
            for recovery in range(5-loss):
                n = int(((lost == loss) & (recovered == recovery)).sum())
                rows.append(dict(cohort=cohort, arm=1, interval='Report to follow-up',
                                 loss=loss, recovery=recovery, n=n,
                                 denominator=denominator, fraction=n/denominator))
    outdir = output_dir/'stats/rift'
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir/'turnover_phase.csv', index=False)
    return outdir
