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
from scoter.series import Series
from scoter.plot import Page, Plot, Axes, Line
from scoter.match import MatchConf, MatchSeriesConf
from scoter.scoter import find_executable
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import os.path
from curveread import read_dataset, Location
import logging
import logging.config
from plot_settings import set_matplotlib_parameters, PLOT_WIDTH
import scipy

logger = logging.getLogger(__name__)


class Constraints(object):

    def __init__(self):
        self.present_year = 1950
        self.tiepoint_offset = -6  # convert from composite to C5 depth
        d18o = r"$\delta^{18}\mathrm{O}$ correlation"
        lirer2014 = "lirer2014planktonic"
        lirer2013 = "lirer2013integrated"
        self.ties_adbc = \
            (  # depth (cm), year, error bar, description, reference
                ( 58,  1906,   0,
                  r"Vesuvius \newline tephra layer", "margaritelli2016marine"),
                ( 93,  1718,  10,
                  r"Abundance peak \newline "
                  r"\emph{Globorotalia \newline truncatulinoides}", lirer2014),
                (166,  1200,  30, d18o, lirer2014),  # d18O/G. ruber/C90
                (199,   901,  38, d18o, lirer2014),  # d18O/G. ruber/C90
                (232,   650,  38, d18o, lirer2014),  # d18O/G. ruber/C90
                (260,   337,  30, d18o, lirer2014),  # d18O/G. ruber/C90
                (299,  -421,  29, d18o, lirer2014),  # d18O/G. ruber/C90
                (328,  -750,  48,
                 r"Top acme \newline \emph{Globigerinoides quadrilobatus}",
                 lirer2013),  # Top acme G. quad.
                (406, -1750,  48,
                 r"Base acme \newline \emph{Globigerinoides quadrilobatus}",
                 lirer2013),  # Base acme G. quad.
                (420, -2248, 100,
                 r"Astroni 3 \newline tephra layer",
                 "smith2011tephrostratigraphy"),  # Astroni 3 tephra
                (436, -2470,  58,
                 r"Agnano M. Spina \newline tephra layer", lirer2013),
            )
        self.shallowest = 52
        self.deepest = 434
        self.youngest = self.present_year - self.ties_adbc[0][1]
        self.oldest = self.present_year - self.ties_adbc[-1][1] + 50

        # We use generators rather than list comprehensions here because, in
        # Python 2, variable bindings leak out of the comprehension into the
        # surrounding scope.
        self.ties_with_margin = \
            tuple((cm + self.tiepoint_offset, self.present_year - year, margin,
                   desc, ref)
                  for (cm, year, margin, desc, ref) in self.ties_adbc)
        self.ties = tuple((cm, age)
                          for cm, age, _, _, _ in self.ties_with_margin)


CONSTRAINTS = Constraints()


class Style(object):
    def __init__(self):
        self.lineprops_record = dict(color="black", zorder=2.2, lw=1.25)
        self.lineprops_record2 = dict(color="darkgrey", zorder=2.0, lw=1.25)
        self.lineprops_target = dict(color="#0060ff", zorder=2.1, lw=0.75)
        self.lineprops_extras = dict(lw=0.5, zorder=0.5, ls="dashed",
                                     dashes=(2, 1))
        self.lineprops_points = dict(zorder=0.5, ls="None", marker="o",
                                     mfc="None", mew=0.2, ms=2)
        self.colourlist = "black green orange blue grey purple red brown olive magenta".split(" ")


STYLE = Style()

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


def make_line_tuple(record, target, extras=None, lines_rec2=None, points=None):
    if extras is None:
        extras = []
    if points is None:
        points = []
    lines_main = [Line(record, **STYLE.lineprops_record),
                  Line(target, **STYLE.lineprops_target)]
    lines_extra = [Line(extra, **STYLE.lineprops_extras)
                   for extra in extras]
    lines_rec2 = [Line(lines_rec2, **STYLE.lineprops_record2)] \
        if lines_rec2 else []
    lines_points = []
    for i in range(len(points)):
        lines_points.append(Line(points[i], mec=STYLE.colourlist[i + 1],
                                 **STYLE.lineprops_points))
    return tuple(lines_main + lines_rec2 + lines_extra + lines_points)


def mark_levantine_spike(axes):
    axes.annotate("*", xy=(2950, 0.960),
                  xycoords=("data", "axes fraction"),
                  color="red", fontsize="large",
                  horizontalalignment="center",
                  verticalalignment="top")


def linear_age_model(dec_target, inc_target, rpi_target,
                     dec_record, inc_record, rpi_record,
                     dec_extras=(), inc_extras=(), rpi_extras=(),
                     dec_points=(), inc_points=(), rpi_points=(),
                     mad3s=None):
    transformed_inc = \
        inc_record.clip((CONSTRAINTS.shallowest, CONSTRAINTS.deepest)). \
        linear_position_transform(CONSTRAINTS.ties).smooth(5)
    transformed_inc.name = "C5 core"
    transformed_rpi = \
        rpi_record.clip((CONSTRAINTS.shallowest, CONSTRAINTS.deepest)). \
        linear_position_transform(CONSTRAINTS.ties).smooth(5)
    transformed_rpi.name = "C5 core"
    transformed_dec = \
        dec_record.clip((CONSTRAINTS.shallowest, CONSTRAINTS.deepest)). \
        linear_position_transform(CONSTRAINTS.ties). \
        wrap_values().smooth(5)
    transformed_dec.name = "C5 core"
    mad3s_clipped = mad3s.clip((CONSTRAINTS.shallowest, CONSTRAINTS.deepest))

    def mean_abs_difference(series1, series2):
        return np.mean(np.abs(series1.interpolate(500).values() -
                              series2.interpolate(500).values()))

    print("dec difference: {:.2f}".format(mean_abs_difference(transformed_dec, dec_target)))
    print("inc difference: {:.2f}".format(mean_abs_difference(transformed_inc, inc_target)))
    print("RPI difference: {:.2f}".format(mean_abs_difference(transformed_rpi, rpi_target)))

    incs_for_comparison_record = transformed_inc.interpolate(400)
    incs_for_comparison_target = inc_target.interpolate(400)
    print(np.mean(np.abs(incs_for_comparison_record.values() - incs_for_comparison_target.values())))

    tie_dates = [t[1] for t in CONSTRAINTS.ties]
    line_opts = dict(lw=1.0)
    axes_opts = dict(xlim=(0, CONSTRAINTS.oldest),
                     vlines=tie_dates,)

    np.savetxt(outpath("decs-linear.csv"), np.transpose(transformed_dec.data),
               delimiter=",")
    np.savetxt(outpath("incs-linear.csv"), np.transpose(transformed_inc.data),
               delimiter=",")
    np.savetxt(outpath("rpis-linear.csv"), np.transpose(transformed_rpi.data),
               delimiter=",")

    for record_depth, record_age in [(inc_record, transformed_inc),
                                     (dec_record, transformed_dec),
                                     (rpi_record, transformed_rpi)]:
        record_clipped = record_depth.clip((CONSTRAINTS.shallowest, CONSTRAINTS.deepest))
        for i in range(record_clipped.npoints()):
            depth = record_clipped.data[0, i]
            if 230 < depth < 250:
                record_age.data[1, i] = np.nan

    def add_mad3_area_inc(ax):
        ax.fill_between(transformed_inc.positions(), transformed_inc.values() - mad3s_clipped.values(),
                        transformed_inc.values() + mad3s_clipped.values(), facecolor="#c0c0c0",
                        zorder=0, color="None")

    def add_mad3_area_dec(ax):
        ax.fill_between(transformed_dec.positions(),
                        map(wrap_declination, transformed_dec.values() - mad3s_clipped.values()),
                        map(wrap_declination, transformed_dec.values() + mad3s_clipped.values()),
                        facecolor="#c0c0c0",
                        zorder=0, color="None")

    page = \
        Page((
              Plot(Axes(make_line_tuple(transformed_inc,
                                        inc_target, extras=inc_extras,
                                        points=inc_points),
                        legend_loc="center left",
                        bbox_to_anchor=(1.002, 0.5),
                        ylabel=u"inclination (°)",
                        customize=add_mad3_area_inc,
                        **axes_opts)),
              Plot(Axes(make_line_tuple(transformed_dec,
                                        dec_target, extras=dec_extras,
                                        points=dec_points),
                        legend_loc="center left",
                        bbox_to_anchor=(1.002, 0.5),
                        ylabel=u"declination (°)",
                        customize=add_mad3_area_dec,
                        **axes_opts)),
              Plot(Axes(make_line_tuple(transformed_rpi,
                                        rpi_target, extras=rpi_extras,
                                        points=rpi_points),
                        legend_loc="center left",
                        bbox_to_anchor=(1.002, 0.5),
                        ylabel="RPI (normalized)",
                        xlabel="Age (years BP)",
                        customize=mark_levantine_spike,
                        **axes_opts)),
              ),
             filename=outpath("fig-linear")
             )
    page.add_line_args(line_opts, False)

    page.plot(gridspec=dict(bottom=0.07, top=0.98, hspace=0.13,
                            left=0.08, right=0.90, wspace=0.01),
              filetype="pdf", figsize=(PLOT_WIDTH, PLOT_WIDTH * 0.81))

    return transformed_dec, transformed_inc, transformed_rpi


def do_match(dec_target, inc_target, rpi_target,
             dec_record, inc_record, rpi_record,
             dec_linear, inc_linear, rpi_linear):

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
            speedchange=5.5,
            tiepenalty=0.065,
            gappenalty=1,
            speeds=speeds)}

    # If True, match will not actually be run, and the cached results
    # from the previous run will be used.
    options_no_run = False

    conf_record = dict(intervals=200,
                       begin=CONSTRAINTS.shallowest, end=CONSTRAINTS.deepest)
    conf_target = dict(intervals=200,
                       begin=CONSTRAINTS.youngest, end=CONSTRAINTS.oldest)

    # find_executable will try to find the match binary on the current
    # path. If it's in some other location, match_path must be set
    # to the full path.
    match_path = find_executable("match")

    dec_target_st = standardize(dec_target)
    inc_target_st = standardize(inc_target)
    rpi_target_st = standardize(rpi_target)
    dec_record_st = standardize(dec_record)
    inc_record_st = standardize(inc_record)
    rpi_record_st = standardize(rpi_record)

    for tie_points, tie_tag in ((None, "noties"), (CONSTRAINTS.ties, "ties")):

        confs = {
            "inc": MatchConf(MatchSeriesConf((inc_record_st, ), **conf_record),
                             MatchSeriesConf((inc_target_st, ), **conf_target),
                             params["inc"], tie_points=tie_points),
            "rpi": MatchConf(MatchSeriesConf((rpi_record_st, ), **conf_record),
                             MatchSeriesConf((rpi_target_st, ), **conf_target),
                             params["rpi"], tie_points=tie_points),
            "tan": MatchConf(
                MatchSeriesConf((dec_record_st, inc_record_st, rpi_record_st),
                                **conf_record),
                MatchSeriesConf((dec_target_st, inc_target_st, rpi_target_st),
                                **conf_target),
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

        series_map = {
            "dec": dec_record.wrap_values(),
            "inc": inc_record,
            "rpi": rpi_record
        }

        if tie_points is not None:
            depth_to_age_map = matches["tan"].match.mapping()
            print("{:>10} {:>10} {:>10} {:>10} {:>10}".format(
                "depth", "orig_age", "warped_age", "offset", "margin"))
            with open("../script-output/tie-point-table.tex", "w") as fh:
                index = 1
                for depth, age, margin, desc, ref in CONSTRAINTS.ties_with_margin:
                    warped_age = round(depth_to_age_map(depth))
                    age_offset = round(age - warped_age)
                    warning = "" if abs(age_offset) <= margin else "EXCESSIVE"
                    cite = r"\citet{%s}" % ref
                    offset_math = "$%.0f$" % age_offset
                    print("{:>10g} {:>10g} {:>10g} {:>10g} {:>10g} {:>10}".format(
                          float(depth), float(age), float(warped_age),
                          float(age-warped_age), float(margin), warning))
                    fh.write("{:>2d} & {:>4g} & {:>4g} & {:>4g} & "
                             "{:50} & {:30} & {:>5g} & {:>5}\\\\\n".format(
                        index, float(depth), float(age), float(margin),
                        desc, cite, float(warped_age), offset_math
                    ))
                    index += 1
        
        # Create warped data sets -- every combination of:
        # warpees (smoothed input data to be warped): inc, rpi
        # warpers (match results to use for warping): inc, rpi, tan
        warped = {}
        for warpee in "dec", "inc", "rpi":
            warped[warpee] = {}
            for warper in "inc", "rpi", "tan":
                w = series_map[warpee].warp_using(matches[warper].match)
                w.name = "C5 matched"
                warped[warpee][warper] = w
                w.write(outpath("%s-matched.txt" % warpee))

        rates = {}
        for conf in "inc", "rpi", "tan":
            rate = matches[conf].match.rate()
            rates[conf] = rate
            rate.name = "Sedimentation rate"
            rate.write(outpath("match", "rate-%s-%s" % (conf, tie_tag)))

        tie_dates = [] if tie_points is None else [t[1] for t in tie_points]

        logger.debug("oldest: " + str(CONSTRAINTS.oldest))

        line_opts = dict(lw=1.0)
        axes_opts = dict(xlim=(0, CONSTRAINTS.oldest),
                         vlines=tie_dates,)

        erase_from = 166
        erase_to = 185
        warped_inc = warped["inc"]["tan"].smooth(5)
        warped_inc.data[1, erase_from:erase_to] = np.nan
        warped_dec = warped["dec"]["tan"].smooth(5)
        warped_dec.data[1, erase_from:erase_to] = np.nan
        warped_rpi = warped["rpi"]["tan"].smooth(5)
        warped_rpi.data[1, erase_from:erase_to] = np.nan
        sed_rate = rates["tan"].scale_values_without_offset(10)
        sed_rate.data[1, 52:63] = np.nan

        tandem_page = \
            Page((
                  Plot(Axes(make_line_tuple(warped_inc,
                                            inc_target, lines_rec2=inc_linear),
                            legend_loc="center left",
                            bbox_to_anchor=(1.002, 0.5),
                            ylabel=u"inclination (°)",
                            **axes_opts)),
                  Plot(Axes(make_line_tuple(warped_dec,
                                            dec_target, lines_rec2=dec_linear),
                            legend_loc="center left",
                            bbox_to_anchor=(1.002, 0.5),
                            ylabel=u"declination (°)",
                            **axes_opts)),
                  Plot(Axes(make_line_tuple(warped_rpi,
                                            rpi_target, lines_rec2=rpi_linear),
                            legend_loc="center left",
                            bbox_to_anchor=(1.002, 0.5),
                            ylabel="RPI (normalized)",
                            customize=mark_levantine_spike,
                            **axes_opts)),
                  Plot(Axes((Line(sed_rate,
                                  color="black"),),
                            legend_loc=None,
                            ylabel="sed. rate (mm/yr)",
                            xlabel="Age (years BP)",
                            **axes_opts)),
                  ),
                 filename=outpath("matched-%s" % tie_tag, )
                 )
        tandem_page.add_line_args(line_opts, False)

        if tie_tag == "ties":
            tandem_page.plot(gridspec=dict(bottom=0.06, top=0.98, hspace=0.13,
                                           left=0.08, right=0.90, wspace=0.01),
                             filetype="pdf",
                             figsize=(PLOT_WIDTH, PLOT_WIDTH * 1.08))

        inc_page = Page((Plot(Axes((Line(warped["inc"]["inc"]),
                                    Line(inc_target),),
                                   legend_loc=(0.82, 0.1),
                                   xlabel=u"Age (years BP)",
                                   ylabel=u"Geomagnetic inclination (°)",
                                   **axes_opts), ),
                         ),
                        filename=outpath("inc-inc-%s" % tie_tag))
        inc_page.add_line_args(line_opts)


def wrap_declination(dec):
    return dec if dec < 300 else dec - 360


def comparison_plot(age_datasets, depth_datasets, parameter,
                    ylims=None, plot_as_points_from=999):
    axes = []
    ylabels = {"decs": u"Declination (°)",
               "incs": u"Inclination (°)",
               "rpis": "RPI (normalized)"}

    for plot_index, datasets in enumerate((age_datasets, depth_datasets)):
        axis1 = plt.subplot(2, 1, plot_index + 1)
        axes.append(axis1)
        if plot_index == 0:
            axis1.xaxis.tick_top()
            axis1.xaxis.set_label_position("top")
            axis1.xaxis.set_label_text("Age (years BP)")
        else:
            axis1.xaxis.set_label_text("Depth (cm)")

        axis1.set_xlim([0, 450])
        axis1.yaxis.set_label_text(
            ("C5 " if plot_index == 1 else "") + ylabels[parameter])
        axis1.patch.set_facecolor("white")
        for dataset_index, dataset in enumerate(datasets):
            label = dataset.name
            if parameter == "rpis" and plot_index == 1:
                label = "MS"
            ys = getattr(dataset, parameter)
            if parameter == "decs":
                ys = map(wrap_declination, ys)
            if dataset_index < plot_as_points_from:
                line = axis1.plot(dataset.xs, ys,
                                  color=STYLE.colourlist[dataset_index],
                                  lw=1,
                                  mec=STYLE.colourlist[dataset_index],
                                  mfc=STYLE.colourlist[dataset_index],
                                  markersize=1,
                                  label=label)
            else:
                line = axis1.plot(dataset.xs, ys,
                                  linestyle="None",
                                  marker="o",
                                  mec=STYLE.colourlist[dataset_index],
                                  mfc="None",
                                  mew=0.3,
                                  markersize=3,
                                  label=label)

            if (parameter in ("incs", "decs") and plot_index == 1 and
               dataset.has_named_data("mad3")):
                axis2 = axis1.twinx()
                axis2.set_xlim([0, 450])
                axis2.set_ylim([0, 5])
                axis2.plot(dataset.xs, dataset.get_named_data("mad3"),
                           color="#009060", lw=0.5)
                axis2.yaxis.set_label_text(u"MAD (°)")
                if parameter == "incs":
                    axis1.fill_between(dataset.xs,
                                       dataset.get_named_data("inc-min"),
                                       dataset.get_named_data("inc-max"),
                                       facecolor="#b0b0b0",
                                       color="None")
                if parameter == "decs":
                    axis1.fill_between(dataset.xs,
                                       map(wrap_declination,
                                           dataset.get_named_data("dec-min")),
                                       map(wrap_declination,
                                           dataset.get_named_data("dec-max")),
                                       facecolor="#b0b0b0",
                                       color="None")

            if parameter == "rpis" and plot_index == 1:
                axis1.plot(dataset.xs, dataset.get_named_data("rpi-arm-ratio"),
                           color=STYLE.colourlist[1], label="ARM-R")
                axis1.plot(dataset.xs, dataset.get_named_data("rpi-arm-slope"),
                           color=STYLE.colourlist[2], label="ARM-S")
                axis1.plot(dataset.xs, dataset.get_named_data("rpi-irm-ratio"),
                           color=STYLE.colourlist[3], label="IRM-R")
                axis1.plot(dataset.xs, dataset.get_named_data("rpi-irm-slope"),
                           color=STYLE.colourlist[4], label="IRM-S")
                axis2 = axis1.twinx()
                axis2.set_xlim([0, 450])
                axis2.set_ylim([0.95, 1])
                axis2.plot(dataset.xs,
                           dataset.get_named_data("rpi-arm-rsquared"),
                           color="#009060", lw=0.5)
                axis2.yaxis.set_label_text(
                    r"$\mathregular{R}^{\mathregular{2}}$")
                axis2.annotate(r"$\mathregular{R}^{\mathregular{2}}$",
                               xy=(6, 0.993), color="#009060")

        if parameter in ("incs", "decs") and plot_index == 1:
            marked_xs = [48, 97, 147, 230, 329, 384]
            all_xs = line[0].get_xdata()
            all_ys = line[0].get_ydata()
            indices = [np.where(all_xs == marked_x)[0][0]
                       for marked_x in marked_xs]
            marked_ys = all_ys[indices]
            axis1.plot(marked_xs, marked_ys,
                       ls="None", marker="o",
                       mec="red", mfc="none", mew=1)

        if parameter == "incs" and plot_index == 1:
            axis1.annotate("inclination", xy=(90, 68), color="black")
            axis1.annotate("MAD", xy=(90, 32), color="#009060")

        if parameter == "decs" and plot_index == 1:
            axis1.annotate("declination", xy=(120, 12), color="black")
            axis1.annotate("MAD", xy=(165, -25), color="#009060")

    axes[0].set_xlim((0, CONSTRAINTS.oldest))

    if ylims is not None:
        axes[0].set_ylim(ylims[0])
        axes[1].set_ylim(ylims[1])

    for core_boundary in (52, 152, 252, 352):
        axes[1].axvline(core_boundary,
                        color="blue", alpha=1.0, lw=1, zorder=0,
                        ymin=0, ymax=0.2)

    for tiepoint_index, tiepoint_pair in enumerate(CONSTRAINTS.ties):
        depth, age = tiepoint_pair
        con = ConnectionPatch(
            (depth, axes[1].get_ylim()[1]), (age, axes[0].get_ylim()[0]),
            "data", "data", arrowstyle="-", axesA=axes[1], axesB=axes[0],
            color="brown", lw=1.5, alpha=0.5)
        axes[0].axvline(age, color="brown", alpha=0.5, lw=1.5, zorder=-10)

        # A bit of a hack to overlay the tie-point numbers.
        axes[0].annotate("|",  # just to paint out the vertical line
                         xy=(age,
                             {"decs": 50, "incs": 79, "rpis": -2.9}[parameter]),
                         horizontalalignment="center",
                         verticalalignment="center",
                         color="white",
                         fontsize="large",
                         bbox=dict(pad=0, facecolor="white", edgecolor="none"))
        axes[0].annotate(str(tiepoint_index+1),  # now write the actual number
                         xy=(age,
                             {"decs": 49, "incs": 78, "rpis": -3}[parameter]),
                         horizontalalignment="center",
                         verticalalignment="center",
                         )

        axes[1].axvline(depth, color="brown", alpha=0.5,  lw=1.5, zorder=-10)
        axes[1].add_artist(con)

    if parameter == "rpis":

        legend0 = axes[0].legend(ncol=1, handletextpad=0.1,
                                 columnspacing=0.5, fontsize="small",
                                 frameon=True,
                                 loc="center left",
                                 bbox_to_anchor=(1.002, 0.5)
                                 )
        frame = legend0.get_frame()
        frame.set_facecolor("white")
        frame.set_alpha(1.0)
        frame.set_zorder(11)

        legend1 = axes[1].legend(loc=(0.02, 0.35), ncol=2, handletextpad=0.0,
                                 columnspacing=0.2, fontsize="small",
                                 frameon=True)
        frame = legend1.get_frame()
        frame.set_facecolor("white")
        frame.set_alpha(1.0)
        frame.set_zorder(11)

    if parameter == "incs":
        axes[0].legend(ncol=1, handletextpad=0.1,
                       columnspacing=0.5, fontsize="small",
                       loc="center left",
                       bbox_to_anchor=(1.002, 0.5)
                       )

    if parameter == "decs":
        axes[0].legend(ncol=1, handletextpad=0.1,
                       columnspacing=0.5, fontsize="small",
                       loc="center left",
                       bbox_to_anchor=(1.002, 0.5)
                       )

    plt.subplots_adjust(left=0.08, right=0.83, top=0.89, bottom=0.11)
    plt.gcf().set_size_inches(PLOT_WIDTH, PLOT_WIDTH * 0.48)
    plt.draw()
    plt.savefig(outpath("fig-" + parameter[:-1] + "-depth.pdf"), format="pdf",
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


def measure_distances(c5):
    print("Distances to other sites:")
    other_locs = dict(
        Augusta = Location.from_dms((37, 12.69), (15, 15.19)),
        ET91_18 = Location(42.6, 9.9),
        ET95_4 = Location(42.9, 9.9),
        Salerno = Location.from_dms((40, 28.918), (14, 42.236)),
        Paris = Location(48.9, 2.3)
        )

    for loc in other_locs:
        print("%15s %.2f " % (loc, c5.distance_km(other_locs[loc])))


def depth_to_age_linear(tie_points, depths):
    tie_points_flipped = zip(*tie_points)
    transform_xs = tie_points_flipped[0]
    transform_ys = tie_points_flipped[1]

    f = scipy.interpolate.interp1d(transform_xs, transform_ys,
                                   fill_value="extrapolate")
    ages = f(np.array(depths))
    return ages


def main():

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    lh = logging.StreamHandler()
    lh.setLevel(logging.DEBUG)
    root.addHandler(lh)

    set_matplotlib_parameters()

    c5_loc = Location(40. + 58./60. + 24.953/3600.,
                      13. + 47./60. + 02.514/3600.)

    # Read the reference data    
    keys = ("cals10k2", "cals3k4", "sha-dif-14k", "uk",
            "c5", "augusta", "igrf_12", "ty1", "ty2",
            "salerno_decs", "salerno_incs", "salerno_rpis",
            "w-europe", "geomagia-italy-psv", "geomagia-italy-rpi",
            "rmd1", "rmd8", "taranto-mp49-psv", "taranto-mp49-rpi")
    data = {key: read_dataset(key, c5_loc, 4500) for key in keys}

    c5 = data["c5"]

    # Create output directories
    match_dir = outpath("match")
    if not os.path.exists(match_dir):
        os.makedirs(match_dir)

    def data_list(keystring):
        return [data[key] for key in keystring.split(" ")]

    # Plot comparisons between C5 and reference data
    comparison_plot(data_list("augusta salerno_decs sha-dif-14k ty1 ty2 "
                              "w-europe geomagia-italy-psv taranto-mp49-psv "
                              "rmd1 rmd8"),
                    (c5, ), "decs",
                    ((-30, 60), (-32, 32)), 6)
    comparison_plot(data_list("augusta salerno_incs sha-dif-14k ty1 ty2 "
                              "w-europe geomagia-italy-psv taranto-mp49-psv "
                              "rmd1 rmd8"),
                    (c5, ), "incs",
                    ((30, 85), (30, 80)), 6)
    comparison_plot(data_list("augusta salerno_rpis sha-dif-14k cals10k2 "
                              "geomagia-italy-rpi taranto-mp49-rpi"),
                    (c5, ), "rpis", None, 4)

    def incseries(name):
        return data[name].inc_series(name)

    def decseries(name):
        return data[name].dec_series(name)

    def rpiseries(name):
        return data[name].rpi_series(name)

    def create_combination_specifiers(series_picker):
        return Series.combine_series(
            (("single", 0, series_picker("cals10k2")),
             ("transition", 155,
              series_picker("cals10k2"), series_picker("augusta")),
             ("single", 250, series_picker("augusta")),
             ("transition", 4050,
              series_picker("augusta"), series_picker("cals10k2")),
             ("single", 4150, series_picker("cals10k2")),
             ("end", CONSTRAINTS.oldest + 100)),
            10)

    print("ref curve end", CONSTRAINTS.oldest + 100)

    inc_comp_2 = create_combination_specifiers(incseries)
    dec_comp_2 = create_combination_specifiers(decseries)
    rpi_comp_2 = create_combination_specifiers(rpiseries)

    dec_extras = (
        data["salerno_decs"].dec_series("Salerno"),
        data["sha-dif-14k"].dec_series("SHA.DIF.14k"),
        data["ty1"].dec_series("ET91-18"),
        data["ty2"].dec_series("ET95-4"),
        data["w-europe"].dec_series("W. Europe"),
    )

    dec_points = (
        data["geomagia-italy-psv"].dec_series("Italy"),
        data["taranto-mp49-psv"].dec_series("MP49"),
        data["rmd1"].dec_series("RMD1"),
        data["rmd8"].dec_series("RMD8"),
    )

    inc_extras = (
        data["salerno_incs"].inc_series("Salerno"),
        data["sha-dif-14k"].inc_series("SHA.DIF.14k"),
        data["ty1"].inc_series("ET91-18"),
        data["ty2"].inc_series("ET95-4"),
        data["w-europe"].inc_series("W. Europe"),
    )

    inc_points = (
        data["geomagia-italy-psv"].inc_series("Italy"),
        data["taranto-mp49-psv"].inc_series("MP49"),
        data["rmd1"].inc_series("RMD1"),
        data["rmd8"].inc_series("RMD8"),
    )

    rpi_extras = (
        data["salerno_rpis"].rpi_series("Salerno"),
        data["sha-dif-14k"].rpi_series("SHA.DIF.14k"),
        data["cals10k2"].rpi_series("CALS10k.2"),
    )

    rpi_points = (
        data["geomagia-italy-rpi"].rpi_series("Italy"),
        data["taranto-mp49-rpi"].rpi_series("MP49")
    )

    inc_comp_2.name = "Reference"
    dec_comp_2.name = "Reference"
    rpi_comp_2.name = "Reference"
    
    # Create linear age model and plot results
    linear_dec, linear_inc, linear_rpi = \
        linear_age_model(data["augusta"].dec_series("Augusta"),
                         data["augusta"].inc_series("Augusta"),
                         data["augusta"].rpi_series("Augusta"),
                         c5.dec_series("c5-dec"),
                         c5.inc_series("c5-inc"),
                         c5.rpi_series("c5-rpi"),
                         dec_extras, inc_extras, rpi_extras,
                         dec_points, inc_points, rpi_points,
                         c5.get_series("mad3", "mad3"))

    linear_dec.name = "C5 linear"
    linear_inc.name = "C5 linear"
    linear_rpi.name = "C5 linear"

    # Perform match and plot results
    do_match(dec_comp_2, inc_comp_2, rpi_comp_2,
             c5.dec_series("c5-inc"), c5.inc_series("c5-inc"),
             c5.rpi_series("c5-rpi"),
             linear_dec,  linear_inc, linear_rpi)

    measure_distances(c5_loc)

    print(depth_to_age_linear(CONSTRAINTS.ties, [48, 97, 147, 230, 329, 384]))
    print(depth_to_age_linear(CONSTRAINTS.ties, [230, 250]))


if __name__ == "__main__":
    main()
