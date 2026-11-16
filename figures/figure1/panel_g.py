"""Adjusted risk differences for the four common daily activities."""
from figures.common import COHORTS, COL, subset, title, key, panel_main


def draw(ax, frame):
    data = subset(frame, "g")
    tasks = ["adl_dress", "adl_eat", "adl_bed", "adl_toilet"]
    for cohort in COHORTS:
        values = data.loc[data.cohort.eq(cohort)].set_index("analysis").reindex(tasks)
        ax.plot(range(4), values.risk_difference*100, "o-", color=COL[cohort], lw=1, ms=3)
    ax.set(xticks=range(4), xticklabels=["Dressing", "Eating", "Bed transfer", "Toileting"],
           ylabel="Adjusted difference (pp)", ylim=(0, 22))
    ax.tick_params(axis="x", rotation=60)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    title(ax, "Task-specific burden profiles")
    key(ax, COHORTS, [COL[cohort] for cohort in COHORTS])


if __name__ == "__main__":
    panel_main(1, "G", draw)
