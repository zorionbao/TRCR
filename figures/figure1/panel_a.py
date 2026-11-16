"""Adjusted excess risk of multiple daily-activity difficulties."""
import numpy as np
from figures.common import COHORTS, COL, subset, title, key, panel_main


def draw(ax, frame):
    data = subset(frame, "Multiple daily activities affected")
    data = data.set_index("cohort").reindex(COHORTS).reset_index()
    bounds = (min(-5, 5*np.floor(data.lo.min()*20)),
              max(30, 5*np.ceil(data.hi.max()*20)))
    for index, row in data.iterrows():
        if np.isfinite(row.risk_difference):
            error = np.array([[row.risk_difference-row.lo],
                              [row.hi-row.risk_difference]])*100
            ax.errorbar(row.risk_difference*100, index, xerr=error, fmt="o",
                        color=COL[row.cohort], ms=4, lw=1.2, capsize=2)
    ax.axvline(0, color="black", lw=.5, ls=":")
    ax.set(yticks=range(3), yticklabels=COHORTS,
           xlabel="Adjusted risk difference (percentage points)", xlim=bounds)
    ax.invert_yaxis()
    title(ax, "Multiple daily activities affected")
    key(ax, COHORTS, [COL[cohort] for cohort in COHORTS])


if __name__ == "__main__":
    panel_main(1, "A", draw)
