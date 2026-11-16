"""Cumulative severity risks across complete functional states."""
from figures.common import COHORTS, COL, subset, title, key, panel_main


def draw(ax, frame):
    from matplotlib.lines import Line2D
    data = subset(frame, "f")
    for cohort in COHORTS:
        for arm, style in [("fracture", "-"), ("reference_projected", "--")]:
            values = data.loc[data.cohort.eq(cohort) & data.arm.eq(arm)].sort_values("threshold")
            ax.plot(values.threshold, values.risk, style, marker="o", ms=2.5,
                    color=COL[cohort], lw=1)
    ax.set(xlabel="Minimum number of difficulties", ylabel="Standardised risk (%)",
           xticks=range(1, 5), ylim=(0, 65))
    ax.legend([Line2D([], [], color="black"), Line2D([], [], color="black", ls="--")],
              ["Fracture", "Reference"], fontsize=7)
    title(ax, "Severity across the full distribution")
    key(ax, COHORTS, [COL[cohort] for cohort in COHORTS])


if __name__ == "__main__":
    panel_main(1, "F", draw)
