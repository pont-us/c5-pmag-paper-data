# -*- coding: utf-8 -*-

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = >
# Copyright 2014 Pontus Lurcock
#
# This file is part of Scoter.
#
# Scoter is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Scoter is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Scoter.  If not, see <http://www.gnu.org/licenses/>.
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = <

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
from matplotlib.backends.backend_svg import FigureCanvasSVG


font_props = None


def set_font_properties():
    global font_props
    font_props = FontProperties()
    font_props.set_size("x-small")
    font_props.set_family("Arial")


class WarpLine(object):
    def __init__(self, bwarp, scale=None,
                 subseries=0, invert=False, **args):
        self.bwarp = bwarp
        self.args = args
        if scale is None:
            self.scale = (bwarp.series[1].series[subseries].end() /
                          bwarp.series[0].series[subseries].end()
                          )
        else:
            self.scale = scale
        self.subseries = subseries
        self.invert = invert

    def plot(self, axes, xoffset=0, yoffset=0):
        xs, ys = self.bwarp.get_rates(scale=self.scale,
                                      invert=self.invert)
        axes.plot(xs, ys, label=self.bwarp.name, **self.args)

    def add_args(self, new_args):
        self.args.update(new_args)


class Line(object):
    def __init__(self, series, **args):
        self.series = series
        self.args = args

    def plot(self, axes, yoffset=0, xoffset=0):
        s = self.series
        axes.plot(s.data[0] + xoffset, s.data[1] + yoffset,
                  label=s.name, **self.args)

    def add_args(self, new_args, overwrite=True):
        if overwrite:
            self.args.update(new_args)
        else:
            combined_args = new_args.copy()
            combined_args.update(self.args)
            self.args = combined_args


class Axes(object):
    def __init__(self, lines, invert=False, spread=0, xspread=0,
                 xlim=(None, None), ylim=(None, None),
                 xlabel=None, ylabel=None, legend_loc="upper left",
                 bbox_to_anchor=(0.95, 1), vlines=[],
                 customize=None):
        if isinstance(lines, (list, tuple)):
            self.lines = lines
        else:
            self.lines = (lines,)
        self.invert = invert
        self.spread = spread
        self.xspread = xspread
        self.xlim = xlim
        self.ylim = ylim
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend_loc = legend_loc
        self.bbox_to_anchor = bbox_to_anchor
        self.vlines = vlines
        self.customize = customize

    def plot(self, axes):
        i = 0
        for vline in self.vlines:
            axes.axvline(vline, color="brown", ymin=0, ymax=0.3)
            axes.axvline(vline, color="brown", ymin=0.7, ymax=1.0)
        for line in self.lines:
            line.plot(axes,
                      xoffset=self.xspread * i,
                      yoffset=self.spread * i)
            i += 1
        if self.invert:
            axes.invert_yaxis()
        box = axes.get_position()
        axes.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        if self.xlabel is not None: axes.set_xlabel(self.xlabel)
        if self.ylabel is not None: axes.set_ylabel(self.ylabel)
        if self.xlim[0] is not None: axes.set_xlim(left=self.xlim[0])
        if self.xlim[1] is not None: axes.set_xlim(right=self.xlim[1])
        if self.ylim[0] is not None: axes.set_ylim(bottom=self.ylim[0])
        if self.ylim[1] is not None: axes.set_ylim(top=self.ylim[1])
        if self.legend_loc is not None:
            axes.legend(prop=font_props,
                        loc=self.legend_loc, bbox_to_anchor=self.bbox_to_anchor)

        if self.customize is not None:
            self.customize(axes)

class Plot(object):
    def __init__(self, ax_spec1, ax_spec2=None):
        self.ax_spec1 = ax_spec1
        self.ax_spec2 = ax_spec2

    def plot(self, fig, gridspec, index):
        ax1 = fig.add_subplot(gridspec[index, :])
        self.ax_spec1.plot(ax1)
        if self.ax_spec2:
            ax2 = ax1.twinx()
            self.ax_spec2.plot(ax2)


class Page(object):
    def __init__(self, plotspecs, filename="temp", title=None):
        self.plotspecs = plotspecs
        self.filename = filename
        self.title = title

    def add_line_args(self, new_args, overwrite = True):
        for p in self.plotspecs:
            for a in (p.ax_spec1, p.ax_spec2):
                if not a:
                    continue
                for l in a.lines:
                    l.add_args(new_args, overwrite)

    def plot(self, gridspec=None, figsize=(11, 8.5), filetype="pdf"):
        if gridspec is None:
            gridspec = dict(left=0.05, right=0.94, wspace=0.05)
        nplots = len(self.plotspecs)
        fig = mpl.figure.Figure(figsize=figsize)
        if filetype.lower() == "pdf":
            FigureCanvasPdf(fig)
        elif filetype.lower() == "svg":
            FigureCanvasSVG(fig)
        else:
            raise ValueError("Unknown filetype: %s" % filetype)
        if self.title:
            fig.suptitle(self.title)
        gs = GridSpec(nplots, 1)
        gs.update(**gridspec)
        for i in xrange(0, nplots):
            self.plotspecs[i].plot(fig, gs, i)
        fig.set_size_inches(figsize)
        fig.savefig(self.filename + "." + filetype,
                    format=filetype.lower(), facecolor="none")


def make_plot(seriess, filename, title=None, invert=False):
    f = mpl.figure.Figure(figsize=(11, 8.5))
    canvas = FigureCanvasPdf(f)
    if title:
        f.suptitle(title)
    
    gs = GridSpec(len(seriess), 1)
    gs.update(left=0.05, right=0.94, wspace=0.05)
    for i in range(0, len(seriess)):
        s = seriess[i]
        if isinstance(s, (list, tuple)):
            ax1 = f.add_subplot(gs[i, :])
            ax2 = ax1.twinx()
            ax2.plot(s[1].data[0], s[1].data[1],
                     color='#aaaaaa', linewidth=5.0)
            ax1.set_ylabel(s[0].get_name())
            ax1.plot(s[0].data[0], s[0].data[1], color='black')
            ax1.set_zorder(ax2.get_zorder()+1)  # put ax in front of ax2
            ax1.patch.set_visible(False)  # hide the 'canvas'
            if invert:
                ax1.invert_yaxis()
                ax2.invert_yaxis()
        else:
            ax = f.add_subplot(gs[i, :])
            ax.set_ylabel(s.get_name())
            ax.plot(s.data[0], s.data[1], color='black')
            if invert:
                ax.invert_yaxis()
    
    canvas.print_figure(filename)


def plot_2(plotdata, filename, title, use_offset=False):
    f = mpl.figure.Figure(figsize=(11, 8.5))
    canvas = FigureCanvasPdf(f)
    f.suptitle(title)
    gs = GridSpec(len(plotdata), 1)
    gs.update(left=0.05, right=0.94, wspace=0.05)
    i = 0
    offset = 0
    for newdata, refdata, invert, series_no in plotdata:
        ax1 = f.add_subplot(gs[i, :])
        ax2 = ax1.twinx()
        data = newdata.match.rate().data
        ax2.plot(data[0], data[1], color='#e0b040', linewidth=2.0,
                 linestyle='-', marker='+', markeredgewidth=2.0)
        ax2max = ax2.get_ylim()[1]
        ax2.set_ylim([0, ax2max * 2])
        ax1.set_ylabel(newdata.series1[series_no].get_name())
        data = newdata.series1[series_no].data
        if use_offset:
            offset = -newdata.series1[series_no].std()*2
            if invert:
                offset = -offset
        ax1.plot(refdata.data[0], refdata.data[1]+offset, linewidth=2.0,
                 color='#9090ff')
        ax1.plot(data[0], data[1], color='black', linewidth=0.75)
        ax1.set_zorder(ax2.get_zorder()+1)  # put ax in front of ax2
        ax1.patch.set_visible(False)  # hide the 'canvas'
        if invert:
            ax1.invert_yaxis()
        i = i + 1
    canvas.print_figure(filename)


class WarpPlotter(object):

    def __init__(self, nblocks, target, interval=100,
                 live=True, pdf_file=None, scale=1.):
        self.interval = interval
        self.lines = []
        self.live = live
        self.scale = scale
        self.pdf_file = pdf_file
        if self.pdf_file:
            self.pdf_pages = PdfPages(self.pdf_file)
        colours = ('yellow', 'red')
        plt.ion()
        self.fig = plt.figure()
        self.fg_colour = '#ffccaa'
        self.bg_colour = '#241C1C'  # '#483737'
        mpl.rc('axes', edgecolor=self.fg_colour, labelcolor=self.fg_colour,
               facecolor=self.bg_colour)
        mpl.rc("font", family="VenturisSans ADF", size='14')
        mpl.rc('axes', edgecolor=self.fg_colour,
               labelcolor=self.fg_colour,
               facecolor=self.bg_colour)
        mpl.rc('xtick', color=self.fg_colour)
        mpl.rc('ytick', color=self.fg_colour)

        self.ax = self.fig.add_subplot(111)
        if target:
            self.ax.plot(target[0], target[1], color='blue',
                         lw=12., ls='--')
        for line in 0, 1:
            xs = range(nblocks)
            ys = range(nblocks)
            linetuple = self.ax.plot(xs, ys, color=colours[line], lw=3.)
            self.lines.append(linetuple[0])

    def finish(self):
        self.lines.pop(1).remove()
        self.fig.canvas.draw()
        if hasattr(self, 'pdf_pages'):
            plt.savefig(self.pdf_pages,
                        facecolor=self.bg_colour, format='pdf')
            self.pdf_pages.close()

    def replot(self, soln_current, soln_new, step):
        if step % self.interval:
            return
        for i, soln in ((0, soln_current), (1, soln_new)):
            xs, ys = soln.get_coords()
            self.lines[i].set_xdata([x * self.scale for x in xs])
            self.lines[i].set_ydata([y * self.scale for y in ys])
        self.fig.canvas.draw()
        if hasattr(self, 'pdf_pages'):
            plt.savefig(self.pdf_pages,
                        facecolor=self.bg_colour, format='pdf')
