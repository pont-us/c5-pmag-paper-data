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
    matplotlib.rcParams.update({'font.size': 12})

    fig, axess = pyplot.subplots(1, 3, sharey=True,
                                 figsize=(200/25.4, 80/25.4))
    filenames = ("b-85-1", "d-35", "e-35")
    sample_codes = (("B", 85), ("D", 35), ("E", 35))
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
    display_names = ["{}".format(core_bottom_depths[sample_code[0]] -
                                 sample_code[1])
                     for sample_code in sample_codes]
    
    for i in range(len(filenames)):
        sample_filename = filenames[i]
        data_path = pathlib.Path("../data/rock-mag/irm/")
        data_points = DataSeries.read_file(str(data_path /
                                               "cleaned" / sample_filename))
        curves = IrmCurves.read_file(str(data_path /
                                         "unmixed" / sample_filename))
        make_one_plot(data_points, curves, display_names[i], axess[i])

    axess[1].set_xlabel("log10(Applied field (mT))")
    axess[0].set_ylabel("Gradient of magnetization")
    pyplot.tight_layout()
    pyplot.savefig(str(pathlib.Path("..", "script-output", "irm-plot.pdf")))


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
