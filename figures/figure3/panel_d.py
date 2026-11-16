"""Figure 3D: participant-multiplier association component distributions."""
from figures import common


def draw(ax, frame):
    data = common.subset(frame, "d")
    for cohort in common.COHORTS:
        values = data.loc[data.cohort.eq(cohort), "association_pp"].to_numpy()
        common.density(ax, values, common.COL[cohort], cohort)
    ax.axvline(0, color="black", lw=.5, ls=":")
    ax.set(xlabel="Association contribution (pp)", ylabel="Multiplier density")
    common.title(ax, "Joint association component")
    common.cohort_legend(ax)


if __name__ == "__main__":
    common.panel_main(3, "d", draw)
