"""Figure 3A: HRS task and joint-association Shapley contributions."""
from figures import common


def draw(ax, frame):
    import numpy as np
    cohort = "HRS"
    data = common.subset(frame, cohort + " multiple").set_index("component").reindex(common.PLAYERS)
    scope = frame.loc[frame["panel"].isin([c + " multiple" for c in common.COHORTS])]
    bounds = (min(-5, 5*np.floor(scope.lo.min()*20)), max(12, 5*np.ceil(scope.hi.max()*20)))
    ax.errorbar(data.value*100, range(5), xerr=np.vstack([data.value-data.lo, data.hi-data.value])*100,
                fmt="o", color=common.COL[cohort], ms=3, lw=1, capsize=2)
    ax.axvline(0, color="black", lw=.5, ls=":")
    ax.set(yticks=range(5), yticklabels=["Dressing", "Eating", "Bed transfer", "Toileting", "Association"],
           xlabel="Contribution (percentage points)", xlim=bounds)
    ax.invert_yaxis()
    common.title(ax, "HRS functional composition")


if __name__ == "__main__":
    common.panel_main(3, "a", draw)
