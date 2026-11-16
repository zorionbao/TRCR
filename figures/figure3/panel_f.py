"""Figure 3F: estimator distributions under an association-only shift."""
from figures import common


def draw(ax, frame):
    import numpy as np
    data = common.subset(frame, "f")
    methods = ["MOSAIC joint", "Independent tasks"]
    for method, color in zip(methods, common.OKABE[:2]):
        group = data.loc[data.method.eq(method)]
        common.density(ax, group.estimate.to_numpy()*100, color, method)
    ax.axvline(data.truth.iloc[0]*100, color="black", ls=":", lw=.8)
    ax.set(xlabel="Estimated risk difference (pp)", ylabel="Density")
    peak = max(float(np.max(line.get_ydata())) for line in ax.lines if len(line.get_ydata()) > 2)
    ax.set_ylim(0, 1.65*peak)
    common.title(ax, "Recovering a pure association shift")
    common.key(ax, methods, common.OKABE[:2], inline=True, loc="upper center")
    legend = ax.get_legend()
    legend.set_frame_on(True)
    legend.get_frame().set(facecolor="white", edgecolor="none", alpha=1)


if __name__ == "__main__":
    common.panel_main(3, "f", draw)
