"""Observed HRS task trajectories around fracture reporting."""
from figures.common import OKABE, subset, title, key, panel_main


def draw(ax, frame):
    data = subset(frame, "a")
    tasks = ["Dressing", "Eating", "Bed transfer", "Toileting"]
    for index, task in enumerate(tasks):
        values = data.loc[data.task.eq(task)].sort_values("lag")
        ax.plot(values.lag, values.risk*100, "o-", color=OKABE[index], ms=2.5, lw=1.1)
    ax.set(xticks=[-2, -1, 0, 1, 2], xticklabels=["−2", "−1", "Report", "+1", "+2"],
           ylim=(0, 65), xlabel="Interview relative to fracture report",
           ylabel="Observed task difficulty (%)")
    title(ax, "HRS: task trajectories")
    key(ax, tasks, OKABE[:4])


if __name__ == "__main__":
    panel_main(2, "A", draw)
