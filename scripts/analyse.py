#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""Perform inclination and RPI analysis for C5 data.

This script reads the C5 inclination and RPI data files, as
written by PuffinPlot. It also reads various reference data
sets. It uses Match (via Scoter) to tune the C5 data to a
reference curve, and produces various graphs and output files.

This script can't be directly integrated with
calc-dirs-and-rpi.py, because this one has to be run in CPython
(for access to numpy, matplotlib, etc.) and calc-dirs-and-rpi.py
has to be run in Jython (for access to PuffinPlot).

"""

import sys
sys.path.append("../libraries/python/")
from matplotlib.backends.backend_svg import FigureCanvasSVG
from scoter.series import Series
from scoter.plot import Page, Plot, Axes, Line
from scoter.match import MatchConf, MatchSeriesConf
from scoter.scoter import find_executable
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import ConnectionPatch
import numpy as np
import os.path
from curveread import read_dataset, Location
import logging
import logging.config

logger = logging.getLogger(__name__)

# depth (cm), year, error bar
ties_adbc = (
    ( 58,    1906,    0),  # Vesuvius
    ( 93,    1718,   10),  # G. trunc. peak
    (166,    1200,   30),  # d18O/G. ruber/C90
    (199,     901,   38),  # d18O/G. ruber/C90
    (232,     650,   38),  # d18O/G. ruber/C90
    (260,     337,    3),  # d18O/G. ruber/C90
    (299,    -421,   29),  # d18O/G. ruber/C90
    (328,    -750,   48),  # Top acme G. quad.
    (406,   -1750,   48),  # Base acme G. quad.
    (420,   -2248,  100),  # Astroni 3 tephra
    (436,   -2470,   58),  # Agnano M. Spina
)

PRESENT_YEAR = 1950
SHALLOWEST = 52
DEEPEST = 434
YOUNGEST = PRESENT_YEAR - ties_adbc[0][1]
OLDEST = PRESENT_YEAR - ties_adbc[-1][1] + 50

TIEPOINT_OFFSET = -6 # convert from composite to C5 depth

ties_with_margin = \
    tuple([(cm + TIEPOINT_OFFSET, PRESENT_YEAR - year, margin)
           for (cm, year, margin) in ties_adbc])

ties = tuple([(cm, age) for cm, age, _ in ties_with_margin])

colourlist = "black green orange blue gray purple red".split(" ")

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def outpath(*path_components):
    """Join path components and resolve them relative to the output directory,
    which is hard-coded within this function. If path_components is empty,
    return the output directory itself.

    path_components : iterable containing path components"""
    basedir = os.path.join("..", "script-output")
    return os.path.join(basedir, *path_components)


def standardize(s):
    return s.interpolate(600).subtract_mean().scale_std_to(1)


def _make_line_tuple(record, target, extras):
    lines1 = [Line(record, color="black", zorder=2.2),
              Line(target, color="#0060ff", zorder=2.1)]
    lines2 = [Line(extra, lw=0.5, zorder=0.5, ls="dashed", dashes=(2, 1))
              for extra in extras]
    return tuple(lines1 + lines2)


def do_match(inc_target, rpi_target, inc_record, rpi_record,
             inc_extras=(), rpi_extras=()):

    speeds = \
        "1:9,1:8,1:7,1:6,1:5,1:4,1:3,2:5,1:2,3:5,2:3,3:4,4:5,1:1," + \
        "5:4,4:3,3:2,5:3,2:1,5:2,3:1,4:1,5:1,6:1,7:1,8:1,9:1"

    params = {
        "inc": dict(
            nomatch=100,
            speedpenalty=0.0,
            targetspeed="1:1",
            speedchange=100.0,
            tiepenalty=1e6,
            gappenalty=1,
            speeds=speeds),
        "rpi": dict(
            nomatch=100,
            speedpenalty=0.1,
            targetspeed="1:1",
            speedchange=100.0,
            tiepenalty=1e6,
            gappenalty=1,
            speeds=speeds),
        "tan": dict(
            nomatch=10,
            speedpenalty=0.0,
            targetspeed="1:1",
            speedchange=3.8,
            tiepenalty=0.08,
            gappenalty=1,
            speeds=speeds)}

    # If True, match will not actually be run, and the cached results
    # from the previous run will be used.
    options_no_run = False

    conf_record = dict(intervals=200, begin=SHALLOWEST, end=DEEPEST)
    conf_target = dict(intervals=200, begin=YOUNGEST, end=OLDEST)

    # find_executable will try to find the match binary on the current
    # path. If it's in some other location, match_path must be set
    # to the full path.
    match_path = find_executable("match")

    inc_target_st = standardize(inc_target)
    inc_target.name = "Inclination reference curve"
    rpi_target_st = standardize(rpi_target)
    rpi_target.name = "RPI reference curve"
    inc_record_st = standardize(inc_record)
    rpi_record_st = standardize(rpi_record)

    for tie_points, tie_tag in ((None, "noties"), (ties, "ties")):

        confs = {
            "inc": MatchConf(MatchSeriesConf((inc_record_st, ), **conf_record),
                             MatchSeriesConf((inc_target_st, ), **conf_target),
                             params["inc"], tie_points=tie_points),
            "rpi": MatchConf(MatchSeriesConf((rpi_record_st, ), **conf_record),
                             MatchSeriesConf((rpi_target_st, ), **conf_target),
                             params["rpi"], tie_points=tie_points),
            "tan": MatchConf(
                MatchSeriesConf((inc_record_st, rpi_record_st), **conf_record),
                MatchSeriesConf((inc_target_st, rpi_target_st), **conf_target),
                params["tan"], tie_points=tie_points)}

        matches = {}

        md_tmpl = outpath("match", "match-%%s-%s" % tie_tag)
        for conf in confs:
            matches[conf] = confs[conf].run_match(match_path,
                                                  md_tmpl % conf,
                                                  options_no_run)

        for conf in matches:
            logger.debug("Checking match results for error...")
            match = matches[conf]
            if hasattr(match, "error") and match.error:
                logger.error("Error running match on : " + md_tmpl % conf)
                logger.error(match.stderr)
                sys.exit(1)

        smoothed = {"inc": inc_record.interpolate().smooth(3),
                    "rpi": rpi_record.interpolate().smooth(3)}

        if tie_points is not None:
            depth_to_age_map = matches["tan"].match.mapping()
            print("{:>10} {:>10} {:>10} {:>10} {:>10}".format(
                "depth", "orig_age", "warped_age", "offset", "margin"))
            for depth, age, margin in ties_with_margin:
                warped_age = round(depth_to_age_map(depth))
                age_offset = round(age - warped_age)
                warning = "" if abs(age_offset) <= margin else "EXCESSIVE"
                print("{:>10g} {:>10g} {:>10g} {:>10g} {:>10g} {:>10}".format(
                      float(depth), float(age), float(warped_age),
                      float(age-warped_age), float(margin), warning))
        
        # Create warped data sets -- every combination of:
        # warpees (smoothed input data to be warped): inc, rpi
        # warpers (match results to use for warping): inc, rpi, tan
        warped = {}
        for warpee in "inc", "rpi":
            warped[warpee] = {}
            for warper in "inc", "rpi", "tan":
                w = smoothed[warpee].warp_using(matches[warper].match)
                if warpee == "rpi":
                    w = w.scale_to(rpi_target)
                w.name = "C5 core %s" % {"inc": "inclination", "rpi": "RPI"}[warpee]
                warped[warpee][warper] = w
                w.write(outpath("%s.txt" % w.name, ))

        rates = {}
        for conf in "inc", "rpi", "tan":
            rate = matches[conf].match.rate()
            rates[conf] = rate
            rate.name = "Sedimentation rate"
            rate.write(outpath("match", "rate-%s-%s" % (conf, tie_tag)))

        tie_dates = [] if tie_points is None else [t[1] for t in tie_points]

        logger.debug("Oldest: "+str(OLDEST))

        def mark_levantine_spike(axes):
            axes.annotate("*", xy=(2947, 2.2), color="red", fontsize="large",
                          horizontalalignment="center")

        line_opts = dict(lw=1.0)
        axes_opts = dict(xlim=(0, OLDEST),
                         vlines=tie_dates,)
        tandem_page = \
            Page((
                  Plot(Axes(_make_line_tuple(warped["inc"]["tan"],
                                             inc_target, inc_extras),
                            legend_loc="lower right",
                            bbox_to_anchor=(1.02, 0.75),
                            ylabel=u"inclination (°)",
                            **axes_opts)),
                  Plot(Axes((Line(rates["tan"], color="black"),),
                            legend_loc="upper right",
                            bbox_to_anchor=(1.02, 0.95),
                            ylabel="sed. rate (cm/yr)",
                            **axes_opts)),
                  Plot(Axes(_make_line_tuple(warped["rpi"]["tan"],
                                             rpi_target, rpi_extras),
                            legend_loc="upper right",
                            bbox_to_anchor=(1.02, 0.31),
                            ylabel="RPI (normalized)",
                            xlabel="Age (years BP)",
                            customize=mark_levantine_spike,
                            **axes_opts)),
                  ),
                 filename=outpath("matched-%s" % tie_tag, )
                 )
        tandem_page.add_line_args(line_opts, False)

        if tie_tag == "ties":
            tandem_page.plot(print_params=dict(bbox_inches="tight",
                                               facecolor="none"),
                             gridspec=dict(left=0.06, right=0.99, wspace=0.01),
                             filetype="pdf", figsize=(11, 6.0))

        inc_page = Page((Plot(Axes((Line(warped["inc"]["inc"]),
                                    Line(inc_target),),
                                   legend_loc=(0.82, 0.1),
                                   xlabel=u"Age (years BP)",
                                   ylabel=u"Geomagnetic inclination (°)",
                                   **axes_opts), ),
                         ),
                        filename=outpath("inc-inc-%s" % tie_tag))
        inc_page.add_line_args(line_opts)
        # inc_page.plot(print_params=dict(bbox_inches="tight", facecolor="none"),
        #               gridspec=dict(left=0.06, right=0.99, wspace=0.01),
        #               figsize=(11, 3))


def comparison_plot(age_datasets, depth_datasets, parameter,
                    ylims=None, legendpos=None):

    axes = []
    ylabels = {"incs": u"Inclination (°)", "rpis": "RPI (normalized)"}
    for plot_index, datasets in enumerate((age_datasets, depth_datasets)):
        axis1 = plt.subplot(2, 1, plot_index+1)
        axes.append(axis1)
        if plot_index == 0:
            axis1.xaxis.tick_top()
            axis1.xaxis.set_label_position("top")
            axis1.xaxis.set_label_text("Age (years BP)")
        else:
            axis1.xaxis.set_label_text("Depth (cm)")

        axis1.set_xlim([0, 450])
        axis1.yaxis.set_label_text(ylabels[parameter])
        axis1.patch.set_facecolor("white")
        for dataset_index, dataset in enumerate(datasets):
            label = dataset.name
            if parameter == "rpis" and plot_index == 1:
                label = "C5 RPI (mag. sus.)"
            line = axis1.plot(dataset.xs, getattr(dataset, parameter),
                              color=colourlist[dataset_index],
                              lw=1,
                              mec=colourlist[dataset_index],
                              mfc=colourlist[dataset_index],
                              markersize=1,
                              label=label)
            if (parameter == "incs" and plot_index == 1 and
               dataset.has_named_data("mad3")):
                axis2 = axis1.twinx()
                axis2.set_xlim([0, 450])
                axis2.set_ylim([0, 5])
                axis2.plot(dataset.xs, dataset.get_named_data("mad3"),
                           color="#009060", lw=0.5)
                axis2.yaxis.set_label_text(u"MAD (°)")

            if parameter == "rpis" and plot_index == 1:
                axis1.plot(dataset.xs, dataset.get_named_data("rpi-arm-ratio"),
                           color=colourlist[1], label="C5 RPI (ARM ratio)")
                axis1.plot(dataset.xs, dataset.get_named_data("rpi-arm-slope"),
                           color=colourlist[2], label="C5 RPI (ARM slope)")
                axis2 = axis1.twinx()
                axis2.set_xlim([0, 450])
                axis2.set_ylim([0.95, 1])
                axis2.plot(dataset.xs, dataset.get_named_data("rpi-rsquared"),
                           color="#009060", lw=0.5)
                axis2.yaxis.set_label_text(
                    u"$\mathregular{R}^{\mathregular{2}}$")
                axis2.annotate(u"$\mathregular{R}^{\mathregular{2}}$",
                               xy=(6, 0.993), color="#009060")

        if parameter == "incs" and plot_index == 1:
            axis1.annotate("inclination", xy=(90, 68), color="black")
            axis1.annotate("MAD", xy=(90, 32), color="#009060")
            marked_xs = [48, 97, 147, 230, 329, 384]
            all_xs = line[0].get_xdata()
            all_ys = line[0].get_ydata()
            indices = [np.where(all_xs == marked_x)[0][0]
                       for marked_x in marked_xs]
            marked_ys = all_ys[indices]
            axis1.plot(marked_xs, marked_ys,
                       ls="None", marker="o",
                       mec="red", mfc="none", mew=1)

    axes[0].set_xlim((0, OLDEST))

    if ylims:
        axes[0].set_ylim(ylims)
        axes[1].set_ylim(ylims)

    for core_boundary in (52, 152, 252, 352):
        axes[1].axvline(core_boundary,
                        color="blue", alpha=1.0, lw=1, zorder=0,
                        ymin=0, ymax=0.2)

    for tiepoint_index, tiepoint_pair in enumerate(ties):
        depth, age = tiepoint_pair
        con = ConnectionPatch(
            (depth, axes[1].get_ylim()[1]), (age, axes[0].get_ylim()[0]),
            "data", "data", arrowstyle="-", axesA=axes[1], axesB=axes[0],
            color="brown", lw=1.5, alpha=0.5)
        axes[0].axvline(age, color="brown", alpha=0.5, lw=1.5, zorder=-10)

        # A bit of a hack to overlay the tie-point numbers.
        axes[0].annotate("|", # just to paint out the vertical line
                         xy=(age, {"incs": 79,"rpis": -2.9}[parameter]),
                         horizontalalignment="center",
                         verticalalignment="center",
                         color="white",
                         fontsize="large",
                         bbox=dict(pad=0, facecolor="white", edgecolor="none"))
        axes[0].annotate(str(tiepoint_index+1), # now write the actual number
                         xy=(age, {"incs": 78,"rpis": -3}[parameter]),
                         horizontalalignment="center",
                         verticalalignment="center",
        )

        axes[1].axvline(depth, color="brown", alpha=0.5,  lw=1.5, zorder=-10)
        axes[1].add_artist(con)

    if parameter == "rpis":

        legend0 = axes[0].legend(loc=(0.515, 0.04), ncol=1, handletextpad=0.1,
                                columnspacing=0.5, fontsize="small",frameon=True)
        frame = legend0.get_frame()
        frame.set_facecolor("white")
        frame.set_alpha(1.0)
        frame.set_zorder(11)

        legend1 = axes[1].legend(loc=(0.22, 0.48), ncol=1, handletextpad=0.1,
                                columnspacing=0.5, fontsize="small",frameon=True)
        frame = legend1.get_frame()
        frame.set_facecolor("white")
        frame.set_alpha(1.0)
        frame.set_zorder(11)

    if parameter == "incs":
        axes[0].legend(loc=(0.2, 0.04), ncol=3, handletextpad=0.1,
                       columnspacing=0.5, fontsize="small")

    plt.subplots_adjust(left=0.07, right=0.92, top=0.9, bottom=0.11)
    plt.gcf().set_size_inches(8, 4.5)
    plt.draw()
    plt.savefig(outpath("plot-" + parameter + ".pdf"), format="pdf",
                facecolor="none")
    plt.close()


def assemble(datasets, param):
    xs = []
    ys = []
    for dataset in datasets:
        for i, x in enumerate(dataset.xs):
            if len(xs) == 0 or x > xs[-1]:
                xs.append(x)
                ys.append(getattr(dataset, param)[i])
    return np.array([xs, ys])


def main():

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    lh = logging.StreamHandler()
    lh.setLevel(logging.DEBUG)
    root.addHandler(lh)

    mpl.rcParams["font.family"] = "Arial"

    c5_loc = Location(40. + 58./60. + 24.953/3600.,
                      13. + 47./60. + 02.514/3600.)

    # Read the reference data    
    keys = ("cals10k2", "cals3k4", "sha-dif-14k", "uk",
            "c5", "augusta", "igrf_12", "ty1", "ty2",
            "salerno_incs", "salerno_rpis")
    data = {key: read_dataset(key, c5_loc) for key in keys}

    c5 = data["c5"]

    # Create output directories
    match_dir = outpath("match")
    if not os.path.exists(match_dir):
        os.makedirs(match_dir)

    def data_list(keystring):
        return [data[key] for key in keystring.split(" ")]

    # Plot comparisons between C5 and reference data
    comparison_plot(data_list("sha-dif-14k cals3k4 cals10k2 uk augusta"),
                    (c5,), "incs",
                    (30, 85), (1.01, 0))
    comparison_plot(data_list("sha-dif-14k cals3k4 cals10k2 augusta"),
                    (c5, ), "rpis", None,
                    (1.01, 0))

    # Create composite reference curves
    inc_comp = Series(assemble(data_list("igrf_12 augusta sha-dif-14k"),
                               "incs"),
                      name="incl. ref. curve")
    rpi_comp = Series(assemble(data_list("igrf_12 augusta sha-dif-14k"),
                               "rpis"),
                      name="RPI ref. curve")

    logger.debug("Inc. reference curve positions: " +
                 str(inc_comp.positions()))

    rpi_comp = rpi_comp.clip((0, OLDEST+100))
    inc_comp = inc_comp.clip((0, OLDEST+100))

    def incseries(name):
        return data[name].inc_series(name)

    def rpiseries(name):
        return data[name].rpi_series(name)

    def create_combination_specifiers(series_picker):
        return Series.combine_series(
        (("single", 0, series_picker("cals10k2")),
         ("transition", 155, series_picker("cals10k2"), series_picker("augusta")),
         ("single", 250, series_picker("augusta")),
         ("transition", 4050, series_picker("augusta"), series_picker("cals10k2")),
         ("single", 4150, series_picker("cals10k2")),
         ("end", OLDEST + 100)),
        10)

    print("ref curve end", OLDEST+100)

    inc_comp_2 = create_combination_specifiers(incseries)
    rpi_comp_2 = create_combination_specifiers(rpiseries)

    # Perform match and plot results
    do_match(inc_comp_2,
             rpi_comp_2,
             c5.inc_series("c5-inc"), c5.rpi_series("c5-rpi"),
             (data["sha-dif-14k"].inc_series("SHA.DIF.14k model"),
              data["ty1"].inc_series("ET91-18 core"),
              data["ty2"].inc_series("ET95-4 core"),
              data["salerno_incs"].inc_series("C1201 core"),
              ),
             (data["sha-dif-14k"].rpi_series("SHA.DIF.14k model"),
              data["cals10k2"].rpi_series("CALS10k.2 model"),
              data["salerno_rpis"].rpi_series("C1201 core"),
              ))


if __name__ == "__main__":
    main()
