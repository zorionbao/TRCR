"""Typography, data selection, mathematical curves and figure export."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import textwrap

COHORTS = ["HRS", "ELSA", "CHARLS"]
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
COL = dict(zip(COHORTS, OKABE[:3]))
PLAYERS = ["Dressing", "Eating", "Bed transfer", "Toileting", "Joint association"]
FILES = {
    1: "Figure1_functional_burden.csv",
    2: "Figure2_persistence.csv",
    3: "Figure3_joint_composition.csv",
    4: "Figure4_capacity_opportunity.csv",
}


def configure(font_dir=None):
    """Select the scientific figure style and require Liberation Sans."""
    import matplotlib as mpl
    mpl.use("Agg")
    from matplotlib import font_manager
    if font_dir is not None:
        directory = Path(font_dir)
        if not directory.is_dir():
            raise FileNotFoundError("The font directory does not exist")
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() in {".ttf", ".otf"}:
                font_manager.fontManager.addfont(str(path))
    for weight in ["normal", "bold"]:
        try:
            font_manager.findfont(font_manager.FontProperties(family="Liberation Sans", weight=weight), fallback_to_default=False)
        except ValueError as error:
            raise RuntimeError("Liberation Sans is required; install it or supply --font-dir") from error
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": .6, "ytick.major.width": .6,
        "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
        "savefig.bbox": None, "savefig.dpi": 300,
    })


def subset(frame, panel):
    """Select the aggregate rows assigned to one source-data panel."""
    result = frame.loc[frame["panel"].eq(panel)].copy()
    if result.empty:
        raise ValueError(f"The source-data panel is empty: {panel}")
    return result


def read_frame(input_dir, figure_number):
    import pandas as pd
    path = Path(input_dir) / FILES[figure_number]
    if not path.is_file():
        raise FileNotFoundError(f"Required source-data file: {FILES[figure_number]}")
    frame = pd.read_csv(path)
    if "panel" not in frame:
        raise ValueError(f"The panel column is required in {FILES[figure_number]}")
    return frame


def title(ax, text):
    ax.set_title("\n".join(textwrap.wrap(text, 28, break_long_words=False, break_on_hyphens=False)), loc="left", pad=7)


def key(ax, labels, colors, markers=None, styles=None, inline=False, loc="upper right", faces=None):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=color, lw=1.2,
                      marker=markers[i] if markers else None,
                      linestyle=styles[i] if styles else "-", ms=3,
                      markerfacecolor=faces[i] if faces else color)
               for i, color in enumerate(colors)]
    if inline:
        ax.legend(handles, labels, loc=loc, fontsize=8)
    else:
        ax._figure_key = (handles, list(labels))
    return handles


def cohort_legend(ax):
    return key(ax, COHORTS, [COL[c] for c in COHORTS])


def heat(ax, values, xlabels, ylabels, title_text, label, vmin=None, vmax=None, div=False):
    from matplotlib.colors import LinearSegmentedColormap
    colors = [OKABE[0], "#FFFFFF", OKABE[1]] if div else ["#FFFFFF", OKABE[5], OKABE[0]]
    cmap = LinearSegmentedColormap.from_list("okabe_div" if div else "okabe_blue", colors)
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
    ax.set(xticks=range(len(xlabels)), xticklabels=xlabels, yticks=range(len(ylabels)), yticklabels=ylabels)
    ax.tick_params(axis="x", rotation=30)
    title(ax, title_text)
    colorbar = ax.figure.colorbar(image, ax=ax, label=label, fraction=.05, pad=.03)
    colorbar.ax._source_axes = ax
    return image


def density(ax, values, color, label=None):
    import numpy as np
    from scipy.stats import gaussian_kde
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or len(values) < 2:
        raise ValueError("A density requires at least two finite observations")
    sd = np.std(values)
    if sd < 1e-10:
        return ax.axvline(np.mean(values), color=color, label=label)
    x = np.linspace(values.min() - sd*.3, values.max() + sd*.3, 240)
    y = gaussian_kde(values)(x)
    line, = ax.plot(x, y, color=color, lw=1.15, label=label)
    ax.fill_between(x, 0, y, color=color, alpha=.09)
    return line


def new_panel(letter):
    from matplotlib import pyplot as plt
    figure = plt.figure(figsize=(105/25.4, 105/25.4))
    ax = figure.add_axes([32/105, 30/105, 46/105, 40/105])
    ax._fixed_rect = list(ax.get_position().bounds)
    figure.text(7/105, 77/105, letter.upper(), fontweight="bold", fontsize=10, va="bottom")
    return figure, ax


def prepare(figure):
    for ax in figure.axes:
        if hasattr(ax, "_fixed_rect"):
            for label in [ax.xaxis.label, ax.yaxis.label]:
                text = label.get_text()
                if "\n" not in text:
                    label.set_text("\n".join(textwrap.wrap(text, 30, break_long_words=False, break_on_hyphens=False)))
            for lower, upper, ticks, setter in [(*sorted(ax.get_xlim()), ax.get_xticks(), ax.set_xticks),
                                                 (*sorted(ax.get_ylim()), ax.get_yticks(), ax.set_yticks)]:
                keep = ticks[(ticks >= lower) & (ticks <= upper)]
                if len(keep):
                    setter(keep)
            ax.set_position(ax._fixed_rect)
        source = getattr(ax, "_source_axes", None)
        if source is not None:
            p = source._fixed_rect
            ax.set_box_aspect(None)
            width_mm = figure.get_figwidth()*25.4
            ax.set_position([p[0] + p[2] + 1.5/width_mm, p[1], 2/width_mm, p[3]])
    figure.canvas.draw()


def geometry(figure, name):
    import matplotlib as mpl
    renderer = figure.canvas.get_renderer()
    data_axes = [ax for ax in figure.axes if hasattr(ax, "_fixed_rect")]
    outside, crossing, overlaps = [], [], []
    texts = []
    for item in figure.findobj(mpl.text.Text):
        if not item.get_visible() or not item.get_text():
            continue
        box = item.get_window_extent(renderer)
        if box.width > 0 and box.height > 0:
            texts.append((item.get_text(), box))
        if box.width > 0 and box.height > 0 and (box.x0 < -2 or box.y0 < -2 or box.x1 > figure.bbox.width+2 or box.y1 > figure.bbox.height+2):
            outside.append(item.get_text())
    for i, ax in enumerate(data_axes):
        labels = [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]
        for label in labels:
            if not label.get_visible() or not label.get_text():
                continue
            box = label.get_window_extent(renderer)
            for j, other in enumerate(data_axes):
                if i != j and box.overlaps(other.get_window_extent(renderer)):
                    crossing.append({"panel": i+1, "other_panel": j+1, "text": label.get_text()})
    for i, (left_text, left_box) in enumerate(texts):
        for right_text, right_box in texts[i+1:]:
            width = min(left_box.x1, right_box.x1) - max(left_box.x0, right_box.x0)
            height = min(left_box.y1, right_box.y1) - max(left_box.y0, right_box.y0)
            if width > 1 and height > 1:
                overlaps.append([left_text, right_text])
    sizes = [[round(ax.get_position().width*figure.get_figwidth()*25.4, 5), round(ax.get_position().height*figure.get_figheight()*25.4, 5)] for ax in data_axes]
    return {"figure": name, "outside_canvas_text": outside, "cross_panel_text": crossing, "text_overlaps": overlaps,
            "data_axes_mm": sizes, "width_mm": round(figure.get_figwidth()*25.4, 3), "height_mm": round(figure.get_figheight()*25.4, 3)}


def export(figure, output_dir, name):
    from matplotlib import pyplot as plt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare(figure)
    record = geometry(figure, name)
    record["files"] = [name + "." + ext for ext in ["pdf", "svg", "png"]]
    for ext in ["pdf", "svg", "png"]:
        figure.savefig(output_dir / (name + "." + ext), dpi=300)
    plt.close(figure)
    return record


def draw_panel(figure_number, panel_letter, draw, frame, output_dir):
    figure, ax = new_panel(panel_letter)
    draw(ax, frame)
    if hasattr(ax, "_figure_key"):
        handles, labels = ax._figure_key
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .99), ncol=min(2, len(labels)), fontsize=7.5)
    return export(figure, output_dir, f"Figure{figure_number}{panel_letter.upper()}")


def panel_main(figure_number, panel_letter, draw):
    parser = argparse.ArgumentParser(description=f"Reproduce Figure {figure_number}{panel_letter.upper()}")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--font-dir", type=Path)
    args = parser.parse_args()
    runtime = args.output_dir / ".runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
    configure(args.font_dir)
    frame = read_frame(args.input_dir, figure_number)
    record = draw_panel(figure_number, panel_letter, draw, frame, args.output_dir)
    print(json.dumps(record, indent=2))
