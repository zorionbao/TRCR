"""CHARLS joint-state probability differences."""
from figures.common import subset, heat, panel_main


def draw(ax, frame):
    data = subset(frame, "e").sort_values("state")
    values = data.difference.to_numpy().reshape(4, 4)*100
    heat(ax, values, ["Neither", "Dressing", "Eating", "Both"],
         ["Neither", "Bed transfer", "Toileting", "Both"],
         "CHARLS: changes in joint states", "Difference (pp)",
         vmin=-20, vmax=20, div=True)


if __name__ == "__main__":
    panel_main(1, "E", draw)
