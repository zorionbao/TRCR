"""Absolute case and case-standardized reference risks."""
from figures.common import COHORTS, COL, subset, title, key, panel_main


def draw(ax, frame):
    from matplotlib.lines import Line2D
    data = subset(frame, "b").set_index("cohort")
    for index, cohort in enumerate(COHORTS):
        row = data.loc[cohort]
        reference = row.counterfactual_reference_risk*100
        case = row.case_risk*100
        ax.plot([reference, case], [index, index], color=COL[cohort], lw=1.4)
        ax.scatter([reference], [index], s=24, facecolor="white", edgecolor=COL[cohort])
        ax.scatter([case], [index], s=24, color=COL[cohort])
    ax.set(yticks=range(3), yticklabels=COHORTS,
           xlabel="Multiple-difficulty risk (%)", xlim=(0, 45))
    ax.invert_yaxis()
    handles = [Line2D([], [], marker="o", color="black", lw=0, mfc="white"),
               Line2D([], [], marker="o", color="black", lw=0)]
    ax.legend(handles, ["Reference", "Fracture"], fontsize=7,
              loc="upper right", bbox_to_anchor=(1, .87))
    title(ax, "Absolute functional burden")
    key(ax, COHORTS, [COL[cohort] for cohort in COHORTS])


if __name__ == "__main__":
    panel_main(1, "B", draw)
