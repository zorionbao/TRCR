"""Render 32 reusable panel examples in four example assemblies."""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
from figures import common


def combined(figure_number, modules, frame, output_dir):
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    figure = plt.figure(figsize=(183/25.4, 300/25.4))
    axes = []
    for i, module in enumerate(modules):
        x = 33 if i % 2 == 0 else 121
        y = 300 - 33 - 40 - (i//2)*(300-33-32-40)/3
        ax = figure.add_axes([x/183, y/300, 46/183, 40/300])
        ax._fixed_rect = list(ax.get_position().bounds)
        figure.text((7 if i % 2 == 0 else 95)/183, (y+47)/300, chr(65+i),
                    fontweight="bold", fontsize=10, va="bottom")
        module.draw(ax, frame)
        axes.append(ax)
    if figure_number in [1, 3]:
        figure.legend([Line2D([], [], color=common.COL[c], marker="o", lw=0, ms=4) for c in common.COHORTS], common.COHORTS, loc="upper center", bbox_to_anchor=(.53, .994), ncol=3)
    elif figure_number == 2:
        figure.legend([Line2D([], [], color=color, lw=1.2) for color in common.OKABE[:4]], ["Dressing", "Eating", "Bed transfer", "Toileting"], loc="upper center", bbox_to_anchor=(.53, .994), ncol=4)
        figure.legend([Line2D([], [], color=common.COL[c], lw=1.2) for c in common.COHORTS], common.COHORTS, title="G,H | Cohorts", loc="lower center", bbox_to_anchor=(.53, .01), ncol=3)
    elif figure_number == 4:
        for labels, colors, y, scope in [(["Joint states", "Independent tasks", "One task per person"], common.OKABE[:3], .997, "A,C,E"), (["q = 0.25", "q = 0.50", "q = 0.75", "q = 1.00"], common.OKABE[:4], .978, "B,D,F")]:
            figure.legend([Line2D([], [], color=color, lw=1.3) for color in colors], labels, loc="upper center", bbox_to_anchor=(.57, y), ncol=len(labels))
            figure.text(.045, y-.008, scope, fontsize=7.5, va="center")
        figure.legend([Line2D([], [], color=common.COL[c], lw=1.2) for c in common.COHORTS], common.COHORTS, title="G,H | Cohorts", loc="lower center", bbox_to_anchor=(.53, .01), ncol=3)
    return common.export(figure, output_dir, f"Figure{figure_number}")


def run(input_dir: Path, output_dir: Path, font_dir=None):
    """Export all panels and figures from the supplied aggregate tables."""
    output_dir = Path(output_dir)
    runtime = output_dir / ".runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
    common.configure(font_dir)
    destination = output_dir / "figures"
    records = []
    for number in range(1, 5):
        frame = common.read_frame(input_dir, number)
        modules = [importlib.import_module(f"figures.figure{number}.panel_{letter}") for letter in "abcdefgh"]
        for letter, module in zip("abcdefgh", modules):
            records.append(common.draw_panel(number, letter, module.draw, frame, destination))
        records.append(combined(number, modules, frame, destination))
    (destination / "figure_geometry_checks.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--font-dir", type=Path)
    args = parser.parse_args()
    records = run(args.input_dir, args.output_dir, args.font_dir)
    print(json.dumps({"figures": len(records), "formats": ["PDF", "SVG", "PNG"]}, indent=2))


if __name__ == "__main__":
    main()
