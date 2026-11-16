"""Figure 3H: analytic component-interval coverage in 200 samples per mechanism."""
from figures import common


def draw(ax, frame):
    import numpy as np
    data = common.subset(frame, "h")
    scenarios = ["Marginal change", "Association change", "Combined change"]
    for j, scenario in enumerate(scenarios):
        group = data.loc[data.scenario.eq(scenario)].set_index("component").reindex(common.PLAYERS)
        ax.plot(np.arange(5)+(j-1)*.08, group.coverage*100, "o", color=common.OKABE[j], ms=3, label=scenario.split()[0])
    ax.axhline(95, color="black", ls=":", lw=.6)
    ax.set(xticks=range(5), xticklabels=["Dressing", "Eating", "Bed transfer", "Toileting", "Association"],
           ylabel="95% interval coverage (%)", ylim=(80, 100))
    ax.tick_params(axis="x", rotation=60)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    common.title(ax, "Analytic uncertainty calibration")
    common.key(ax, ["Marginal", "Association", "Combined"], common.OKABE[:3], markers=["o"]*3, styles=[""]*3, inline=True, loc="lower left")


if __name__ == "__main__":
    common.panel_main(3, "h", draw)
