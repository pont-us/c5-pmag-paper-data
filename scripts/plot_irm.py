#!/usr/bin/env python3

import sys
from clgplot import DataSeries, IrmCurves, gradient
import pathlib
import matplotlib
from matplotlib import pyplot
from numpy import log10, arange
from plot_settings import set_matplotlib_parameters


def main():
    set_matplotlib_parameters()

    nrows = 2
    ncols = 3
    fig, axess = pyplot.subplots(nrows, ncols,
                                 sharex=False, sharey=False,
                                 squeeze=False,
                                 figsize=(180 / 25.4, 100 / 25.4))
    filenames_all = ("a-35", "a-85",
                     "b-30", "b-85", "b-90",
                     "c-10", "c-50",
                     "d-10", "d-30", "d-35", "d-70", "d-85",
                     "e-10", "e-35", "e-50", "e-90",
                     "f-10", "f-50", "f-70", "f-90",
                     "g-40", "g-70", "g-90",
                     "h-01", "h-10", "h-30", "h-50")
    filenames_selected = ("d-35", "e-10", "e-90", "f-50", "g-40", "h-30")
    filenames = filenames_selected
    display_names = filenames_to_sample_depths(filenames)
    print("All sample depths:",
          ", ".join(map(str, sorted(map(int,
                               filenames_to_sample_depths(filenames_all))))))
    
    for i in range(len(filenames)):
        sample_filename = filenames[i]
        data_path = pathlib.Path("../data/rock-mag/irmunmix/combined")
        data_points = DataSeries.read_file(
            str(data_path / "measurements" / ("c5" + sample_filename)))
        curves = IrmCurves.read_file(
            str(data_path / "fits" / ("c5" + sample_filename + "-1comp")))
        row = i // ncols
        col = i % ncols
        axes = axess[row, col]
        make_one_plot(data_points, curves, display_names[i], axes)
        axes.set_xlim(0, 3.0)
        axes.set_ylim(0, 1.4)
        if row < nrows - 1:
            axes.xaxis.set_ticklabels([])
        else:
            if col < ncols - 1:
                axes.xaxis.get_major_ticks()[-1].label1.set_visible(False)
        if col > 0:
            axes.yaxis.set_ticklabels([])
        else:
            if row == nrows - 1:
                axes.yaxis.get_major_ticks()[-1].label1.set_visible(False)

    axess[0][0].set_xlim(0, 3)
    fig.text(0.5, 0.01, "log10(Applied field (mT))", ha="center")
    fig.text(0.01, 0.5, "Gradient of magnetization", va="center",
             rotation="vertical")
    pyplot.tight_layout(h_pad=0, w_pad=-0.2, rect=(0.025, 0.025, 1.02, 1.02))
    pyplot.savefig(str(pathlib.Path("..", "script-output", "fig-irm.pdf")))


def filenames_to_sample_depths(filenames):
    sample_codes = tuple(map(lambda x: (x[0].upper(), int(x[2:4])),
                             filenames))
    core_bottom_depths = {
        "H":  52,
        "G": 152,
        "F": 252,
        "E": 352,
        "D": 452,
        "C": 504,
        "B": 604,
        "A": 703
        }
    return ["{}".format(core_bottom_depths[sample_code[0]] -
                                 sample_code[1])
                     for sample_code in sample_codes]


def make_one_plot(data_points, curves, name, axes):
    sirm = curves.sirm
    xs = list(map(log10, data_points.data[0][1:]))
    ys = data_points.data[1][1:]
    axes.plot(xs, gradient(xs, ys) / sirm, marker="o",
              ls="", color="black", markerfacecolor="none", markersize=6)

    xs = arange(0.1, 3, 0.02)
    ys = [curves.evaluate(x, True) for x in xs]
    axes.plot(xs, ys, linewidth=1.0, color="black")
    for curve in curves.components:
        ys2 = [curve.evaluate(x) for x in xs]
        axes.plot(xs, ys2, linewidth=0.5, color="black")
    axes.annotate(name, xy=(2.2, 1.2))
    

if __name__ == "__main__":
    main()
