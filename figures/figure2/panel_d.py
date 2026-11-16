"""Concurrent newly difficult and resolved tasks in HRS."""
import numpy as np
from figures.common import COL, subset, title, panel_main


def draw(ax, frame):
    data = subset(frame, "HRS turnover")
    observed = data.loc[data.n > 0]
    ax.scatter(observed.loss, observed.recovery, s=500*np.sqrt(observed.fraction),
               c=COL["HRS"], alpha=.7, edgecolor="white", lw=.5)
    ax.set(xticks=range(5), yticks=range(5), xlim=(-.5, 4.6), ylim=(-.5, 4.6),
           xlabel="New difficulties", ylabel="Resolved difficulties")
    for fraction, vertical in [(.01, 4.1), (.1, 3.2), (.5, 2.15)]:
        ax.scatter([3.4], [vertical], s=500*np.sqrt(fraction), color=COL["HRS"],
                   alpha=.7, edgecolor="white", lw=.5)
        ax.text(3.9, vertical, f"{fraction*100:g}%", fontsize=7, va="center")
    title(ax, "HRS: concurrent functional change")


if __name__ == "__main__":
    panel_main(2, "D", draw)
