"""Figure 3G: RMSE across three known-truth generating mechanisms."""
from figures import common


def draw(ax, frame):
    data = common.subset(frame, "g")
    scenarios = ["Marginal change", "Association change", "Combined change"]
    methods = ["MOSAIC joint", "Independent tasks"]
    for method, color in zip(methods, common.OKABE[:2]):
        group = data.loc[data.method.eq(method)].set_index("scenario").reindex(scenarios)
        ax.plot(range(3), group.rmse*100, "o-", color=color, ms=3, label=method)
    ax.set(xticks=range(3), xticklabels=["Marginal", "Association", "Combined"],
           ylabel="RMSE (percentage points)", ylim=(0, 5))
    ax.tick_params(axis="x", rotation=30)
    common.title(ax, "Accuracy across distinct mechanisms")
    common.key(ax, methods, common.OKABE[:2], inline=True, loc="upper left")


if __name__ == "__main__":
    common.panel_main(3, "g", draw)
