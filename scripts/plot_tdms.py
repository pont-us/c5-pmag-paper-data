#!/usr/bin/python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os
import os.path
import re

from tdmagsus import Furnace
from tdmagsus import MeasurementCycle

from plot_settings import set_matplotlib_parameters


def main():
    set_matplotlib_parameters()
    section_bottom_depths = create_section_depth_table()
    data_dir = "../data/rock-mag/tdms"
    furnace_filename = "FURMAR19.CUR"
    furnace = Furnace(os.path.join(data_dir, furnace_filename))
    measurement_cycles = {}
    sample_names = []
    for filename in sorted(os.listdir(data_dir), reverse=True):
        if furnace_filename == filename:
            continue
        sample_name = filename[:-4]
        sample_names.append(sample_name)
        measurement_cycles[sample_name] = \
            MeasurementCycle(furnace, os.path.join(data_dir, filename),
                             0.25, 10.0)

    assert(8 == len(sample_names))
    fig = plt.figure(figsize=(180 / 25.4, 70 / 25.4))
    depths = []
    for i in range(len(sample_names)):
        axes = fig.add_subplot(2, 4, i + 1)
        sample_name = sample_names[i]
        cycle = measurement_cycles[sample_name] 
        data = cycle.data
        depth = sample_id_to_depth(section_bottom_depths, sample_name)
        depths.append(depth)

        axes.yaxis.set_ticklabels([])
        axes.set_xlim(0, 7)
        
        axes.plot(data[0][0] / 100, data[0][1], color="#d95f02", lw=0.5)
        axes.plot(data[1][0] / 100, data[1][1], color="#1b9e77", dashes=(2, 2),
                  lw=0.5)
        curie_fit = cycle.curie_paramag(550, 600)
        
        print("{} {} {:.1f} {:.2f}".format(sample_name, depth, curie_fit[0],
                                           curie_fit[1]))
        axes.axvline(curie_fit[0] / 100, ymin=-0.05, ymax=0.1, color="#7570b3",
                     lw=2)

        if i > 3:
            axes.xaxis.get_major_ticks()[-1].label1.set_visible(False)
            axes.xaxis.get_major_ticks()[0].label1.set_visible(False)
        else:
            axes.xaxis.set_ticklabels([])

        axes.text(0.05, 0.5,
                  "{}$\\,$cmbsf".format(depth),
                  color="gray",
                  horizontalalignment="left",
                  verticalalignment="center", transform=axes.transAxes)

        axes.tick_params(axis="x", color="gray", length=3)
        axes.tick_params(left=False, right=False)
        for side in "bottom", "left", "top", "right":
            axes.spines[side].set_linewidth(0.5)
            axes.spines[side].set_color("gray")

    fig.text(0.5, 0.03, "Temperature (°C / 100)", ha="center")
    fig.subplots_adjust(left=0.07, right=0.995, top=0.99, bottom=0.15,
                        wspace=0.02, hspace=0.03)
    fig.text(0.01, 0.5, "Magnetic susceptibility",
             va="center", rotation="vertical")
    fig.text(0.04, 0.5, "(arbitrary units)",
             va="center", rotation="vertical")
    fig.savefig("../script-output/fig-tdms.pdf")

    print("Depths:", ", ".join(map(str, sorted(depths))))

    
def create_section_depth_table():
    section_lengths = \
        [("H", 52),
         ("G", 100),
         ("F", 100),
         ("E", 100),
         ("D", 100),
         ("C", 52),
         ("B", 100),
         ("A", 99)]

    section_bottom_depths = {}
    current_bottom = 0
    for section_code, length in section_lengths:
        current_bottom += length
        section_bottom_depths[section_code] = current_bottom

    return section_bottom_depths


def sample_id_to_depth(section_bottom_depths, sample_id):
    match = re.match(r"^C5(.)-(\d+)$", sample_id)
    return section_bottom_depths[match.group(1)] - int(match.group(2))

    
if __name__ == "__main__":
    main()
