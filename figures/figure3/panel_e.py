"""Figure 3E: all 32 marginal-and-association replacement coalitions."""
from figures import common


def draw(ax, frame):
    data = common.subset(frame, "e")
    values = data.pivot(index="cohort", columns="coalition", values="value").reindex(common.COHORTS).sort_index(axis=1).to_numpy()*100
    common.heat(ax, values, [str(i) for i in range(32)], common.COHORTS,
                "The complete composition landscape", "Multiple-difficulty risk (%)", 0, 40)
    ax.set(xticks=[0, 7, 15, 23, 31], xticklabels=["0", "7", "15", "23", "31"], xlabel="Replacement coalition (0-31)")
    ax.tick_params(axis="x", rotation=0)


if __name__ == "__main__":
    common.panel_main(3, "e", draw)
