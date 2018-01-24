#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Produce Day plot from VSM data files.
"""

from __future__ import (absolute_import, division, print_function)
from builtins import *
import sys
import re
import os.path
from matplotlib import pyplot
import matplotlib
from matplotlib.pyplot import Rectangle
import argparse
import codecs


def main():
    fields_in_file = ["Initial slope", "Saturation",
                      "Remanence", "Coercivity", "S*", "Coercivity (remanent)"]
    fields_to_calculate = ["Bcr/Bc", "Mrs/Ms"]
    fields_all = fields_in_file + fields_to_calculate

    command_line_arguments = parse_command_line_arguments()
    sample_dict = read_micromag_files(command_line_arguments.hystfile,
                                      fields_in_file)
    calculate_day_plot_parameters_and_write_to_dict(sample_dict)
    write_fields_and_params_to_stdout(sample_dict, fields_all)
    xs, ys = get_day_plot_parameters_from_dict(sample_dict)
    make_plot(command_line_arguments.language[0], xs, ys)


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="Create a Day plot from VSM data")
    parser.add_argument("--language", type=str, nargs=1,
                        default=["it"], choices=["it", "en"],
                        help="language for labels")
    parser.add_argument("hystfile", type=str, nargs="+",
                        help="hysteresis parameter file")
    args = parser.parse_args()
    return args


def read_micromag_files(filenames, fields_in_file):
    sample_dict = {}
    for filename in filenames:
        sample_code = os.path.basename(filename)[0:6]
        if sample_code not in sample_dict:
            sample_dict[sample_code] = {}
        params = sample_dict[sample_code]
        with codecs.open(filename, "r", "iso-8859-1") as file_object:
            for line in file_object.readlines():
                parts = re.split("  +", line.strip())
                if parts[0] in fields_in_file and parts[1] != "N/A":
                    value_flt = float(parts[1])
                    # if parts[0]=="Remanence": value_flt /= 10.
                    params[parts[0]] = value_flt
    return sample_dict


def calculate_day_plot_parameters_and_write_to_dict(sample_dict):
    for sample, params in sample_dict.items():
        params["Mrs/Ms"] = params["Remanence"] / params["Saturation"]
        params["Bcr/Bc"] = (params["Coercivity (remanent)"] /
                            params["Coercivity"])


def get_day_plot_parameters_from_dict(sample_dict):
    xs, ys = [], []
    for key in sample_dict.keys():
        xs.append(sample_dict[key]["Bcr/Bc"])
        ys.append(sample_dict[key]["Mrs/Ms"])
    return xs, ys


def write_fields_and_params_to_stdout(sample_dict, fields_all):
    file_handle = sys.stdout
    separator = ","
    file_handle.write(separator.join(["File"] + fields_all) + "\n")

    for filename, params in sample_dict.items():
        def getval(x): return str(params.get(x))

        values = [filename] + list(map(getval, fields_all))
        file_handle.write(separator.join(values) + "\n")


def make_plot(language, xs, ys):

    settings = dict(
        xmin=0,
        xmax=5,
        ymin=0,
        ymax=0.6,
        magmin=0.05,
        magmax=0.5,
        coercivitymin=1.5,
        coercivitymax=4
    )

    settings.update(settings)
    
    
    fontname = "NimbusSanL"
    font = {"family": fontname,
            "weight": "normal",
            "size": 22}
    matplotlib.rc("font", **font)
    matplotlib.rcParams["mathtext.fontset"] = "custom"
    matplotlib.rcParams["mathtext.it"] = fontname + ":italic"
    matplotlib.rcParams["mathtext.rm"] = fontname
    matplotlib.rcParams["mathtext.tt"] = fontname
    matplotlib.rcParams["mathtext.bf"] = fontname
    matplotlib.rcParams["mathtext.cal"] = fontname
    matplotlib.rcParams["mathtext.sf"] = fontname

    colour_main = "#ffffff"
    colour_highlight = "#ffffff"
    pyplot.xlabel("$B_{\mathrm{cr}} / B_\mathrm{c}$")
    pyplot.ylabel("$M_{\mathrm{rs}} / M_{\mathrm{s}}$")
    pyplot.xlim(settings["xmin"], settings["xmax"])
    pyplot.ylim(settings["ymin"], settings["ymax"])
    pyplot.gca().add_patch(Rectangle((0, 0), settings["xmax"],
                                     settings["ymax"], fc=colour_main))
    pyplot.gca().add_patch(Rectangle((settings["coercivitymin"],
                                      settings["magmin"]),
                                     settings["coercivitymax"] -
                                     settings["coercivitymin"],
                                     settings["magmax"] -
                                     settings["magmin"],
                                     fc=colour_highlight))
    pyplot.annotate("SD", (0.05, 0.53))
    pyplot.annotate("PSD", (1.60, 0.42))
    pyplot.annotate({"it": "PD", "en": "MD"}[language], (4.1, 0.01))
    pyplot.hlines([settings["magmin"], settings["magmax"]], 0., 5.)
    pyplot.vlines([settings["coercivitymin"], settings["coercivitymax"]],
                  0.0, 0.6)
    pyplot.subplots_adjust(left=0.12, bottom=0.14, right=0.97, top=0.97)
    pyplot.plot(xs, ys, "o", color="none", mew=2., ms=12., mec="none",
                mfc="black", alpha=0.5)
    pyplot.savefig(os.path.join("..", "script-output", "day-plot.pdf"),
                   transparent=True)


if __name__ == "__main__":
    main()
