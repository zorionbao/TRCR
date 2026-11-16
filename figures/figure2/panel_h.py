"""Exact four-task state persistence by reporting burden."""
from figures.common import COHORTS, COL, subset, title, key, panel_main


def draw(ax, frame):
    data = subset(frame, "gh")
    for cohort in COHORTS:
        values = data.loc[data.cohort.eq(cohort)].sort_values("count")
        ax.plot(values["count"], values.same_state_percent, "o-",
                color=COL[cohort], ms=3)
    ax.set(xlabel="Difficulties at report", ylabel="Same task state (%)",
           xticks=range(5), ylim=(0, 100))
    title(ax, "Persistence of the exact task pattern")
    key(ax, COHORTS, [COL[cohort] for cohort in COHORTS])


if __name__ == "__main__":
    panel_main(2, "H", draw)
