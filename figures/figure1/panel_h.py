"""Joint-state burden and the matching product-of-marginals risk."""
from figures.common import COHORTS, COL, subset, title, key, panel_main


def draw(ax, frame):
    from matplotlib.lines import Line2D
    data = subset(frame, "h")
    for cohort in COHORTS:
        for arm, marker in [("reference", "o"), ("fracture", "s")]:
            row = data.loc[data.cohort.eq(cohort) & data.arm.eq(arm)].iloc[0]
            ax.scatter(row.independent_multi*100, row.multi_risk*100,
                       s=27, marker=marker, color=COL[cohort])
    ax.plot([0, 45], [0, 45], color="black", ls=":", lw=.5)
    ax.set(xlabel="Risk under task independence (%)", ylabel="Joint-state risk (%)",
           xlim=(0, 45), ylim=(0, 45))
    handles = [Line2D([], [], color="black", marker="o", lw=0),
               Line2D([], [], color="black", marker="s", lw=0)]
    ax.legend(handles, ["Reference", "Fracture"], fontsize=7, loc="upper left")
    title(ax, "State dependence reshapes burden")
    key(ax, COHORTS, [COL[cohort] for cohort in COHORTS])


if __name__ == "__main__":
    panel_main(1, "H", draw)
