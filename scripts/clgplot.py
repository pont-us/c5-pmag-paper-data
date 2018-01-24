#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# clgplot is Copyright 2012-2017 Pontus Lurcock (pont at talvi dot net)
# and released under the MIT license:

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
A simple interactive application for plotting IRM acquisition data.

clgplot reads two types of file: raw IRM acquisition data (as a two-column
file containing applied field and measured magnetization), and output files
from the IrmUnmix program [1] which contain a representation of IRM
acquisition data as a sum of cumulative log-Gaussian (CLG) curves. clgplot
produces a plot of the data and/or curves using pyplot. The plot can be viewed
on-screen and saved to a file.

clgplot is copyright 2012-2017 by Pontus Lurcock, who may be contacted at
pont talvi net
    @     .

[1] See http://www.geo.uu.nl/~forth/Software/irmunmix/ and
http://dx.doi.org/10.1046/j.0956-540x.2001.01558.x
"""

import tkinter
import os
from matplotlib import pyplot
import re
from math import sqrt, erf
from numpy import pi, exp, log10, array, arange
import argparse
from os.path import basename
from tkinter.filedialog import askopenfilename


def gradient(xs, ys):
    """Return a new array containing the gradients at all the x points"""
    result = []
    for i in range(0, len(xs)):
        grad1, grad2 = None, None
        if i < len(xs) - 1:
            grad1 = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
        if i > 0:
            grad2 = (ys[i] - ys[i - 1]) / (xs[i] - xs[i - 1])
        if grad1 is None:
            grad1 = grad2
        if grad2 is None:
            grad2 = grad1
        result.append((grad1 + grad2) / 2)
    return array(result)


def x_for_half_max_y(xs, ys):
    """Return the x value for which the corresponding y value is half
    of the maximum y value. If there is no exact corresponding x value,
    one is calculated by linear interpolation from the two
    surrounding values.

    :param xs: x values
    :param ys: y values corresponding to the x values
    :return:
    """

    if len(xs) != len(ys):
        raise ValueError("xs and ys must be of equal length")

    half_max_y = max(ys) / 2
    for i in range(len(xs)-1):
        if ys[i+1] >= half_max_y:
            x_dist = xs[i+1] - xs[i]
            y_dist = ys[i+1] - ys[i]
            y_offset = half_max_y - ys[i]
            if y_offset == 0:
                return xs[i]
            else:
                x_offset = y_offset / y_dist * x_dist
                return xs[i] + x_offset
    return None


class DataSeries:
    """A lightly wrapped 2-column matrix with a method for reading
    it from a file."""

    def __init__(self, data, name=None, filename=None):
        self.data = data
        self.filename = filename
        self.name = name
        if name is not None:
            self.name = name
        else:
            if filename is not None:
                self.name = os.path.basename(filename)
            else:
                self.name = None

    @staticmethod
    def read_file(filename, col1=0, col2=1, name=None):
        """Reads a series from a two-column whitespace-delimited text file. If
        there are more than two columns, the extra ones are ignored. If there
        is a header line (or any other non-numeric line), it is ignored."""

        rows = []
        # "U" for universal newlines
        with open(filename, "U") as fh:
            for line in fh.readlines():
                parts = line.split()
                try:
                    position = float(parts[col1])
                    if len(parts) > col2:
                        value = float(parts[col2])
                    else:
                        print("WARNING: missing data at " + str(position))
                        value = 0
                    rows.append([position, value])
                except ValueError:
                    pass  # ignore non-numeric lines
        data = array(rows).transpose()
        return DataSeries(data, name=name, filename=filename)


class Gaussian:
    def __init__(self, m_abs, m, bhalf, dp):
        self.m_abs = m_abs  # absolute contribution (not used)
        self.m = m  # relative contribution (size of peak)
        self.a = m / (dp * (2 * pi) ** 0.5)  # corrected for dispersion
        self.bhalf = bhalf  # mean log of field (position of peak)
        self.dp = dp  # dispersion parameter (width of peak)

    def evaluate(self, x):
        (a, b, c) = (self.a, self.bhalf, self.dp)
        return a * exp(-(((x - b) ** 2) / (2 * (c ** 2))))

    def cdf(self, x):
        (a, b, c) = (self.a, self.bhalf, self.dp)
        return 0.5 * (1 + erf((x - b) / (sqrt(2 * c ** 2))))

    def to_csv_line(self):
        return "%.2f,%.2f,%.2f,%.2f,%.2f" % \
               (self.m_abs, self.m, self.a, self.bhalf, self.dp)

    @staticmethod
    def csv_header():
        return "M_abs,m_rel,a,Bhalf,DP"


class IrmCurves:
    """A collection of cumulative log-gaussian functions
    which can be evaluated to give a modelled IRM remanence
    for a specified applied field."""

    def __init__(self, name, sirm, params):
        self.name = name
        self.sirm = sirm
        self.components = [Gaussian(*p) for p in params]

    def evaluate(self, x, normalize=False):
        result = sum([g.evaluate(x) for g in self.components])
        if not normalize:
            result = result * self.sirm
        return result

    @staticmethod
    def read_file(filename):
        re1 = re.compile(r"^ True SIRM= +([0-9.E-]+)")
        re2 = re.compile(r"^ Abs Cont= +([0-9.E-]+)")
        re3 = re.compile(r"^ Rel Cont= +([0-9.E-]+) +Mean= +([0-9.E-]+) +" +
                         r"DP= +([0-9.E-]+)\s+$")
        infile = open(filename)
        sirm = float(re1.search(infile.readline()).groups()[0])
        params = []
        infile.readline()
        while True:
            comp = infile.readline()
            if not comp.startswith(" Component"):
                break
            param = [float(re2.search(infile.readline()).groups()[0])]
            line3 = infile.readline()
            param += map(float, re3.search(line3).groups())
            params.append(param)
            infile.readline()  # skip blank line
        return IrmCurves(basename(filename), sirm, params)

    def to_csv_line(self):
        result = self.name + ","
        result += ",".join([c.to_csv_line() for c in self.components])
        return result

    def csv_header(self):
        return "Sample," + \
               ",".join([Gaussian.csv_header()] * len(self.components))


def plot_clg_fit(series, curves, output_filename=None):
    sirm = 1
    if curves:
        sirm = curves.sirm

    if series:
        xs = list(map(log10, series.data[0][1:]))
        ys = series.data[1][1:]
        pyplot.plot(xs, gradient(xs, ys) / sirm, marker="o",
                    ls="", color="black", markerfacecolor="none", markersize=6)

    if curves:
        xs = arange(0.1, 3, 0.02)
        ys = [curves.evaluate(x, True) for x in xs]
        pyplot.plot(xs, ys, linewidth=1.0, color="black")
        for curve in curves.components:
            ys2 = [curve.evaluate(x) for x in xs]
            pyplot.plot(xs, ys2, linewidth=0.5, color="black")

    pyplot.ylim(ymin=0)
    pyplot.xlabel("log10(Applied field (mT))")
    pyplot.ylabel("Gradient of magnetization")
    if output_filename:
        pyplot.savefig(output_filename)
    else:
        pyplot.ion()
        pyplot.show()


class App:
    def __init__(self, master, data=None, curves=None,
                 plot_now=False):

        self.series = data
        self.curves = curves

        master.title("CLG Plot")
        frame = tkinter.Frame(master)
        frame.grid(padx=20, pady=15)

        self.data_button = \
            tkinter.Button(frame, text="Choose Data file",
                           command=self.choose_data_file)
        self.data_button.grid(row=0, pady=5)

        self.irmunmix_button = \
            tkinter.Button(frame, text="Choose IrmUnmix file",
                           command=self.choose_curves_file)
        self.irmunmix_button.grid(row=1, pady=5)

        self.plot_button = \
            tkinter.Button(frame, text="Plot data", command=self.plot)
        self.plot_button.grid(row=2, pady=5)

        self.quit_button = tkinter.Button(frame, text="Quit",
                                          command=frame.quit)
        self.quit_button.grid(row=3, pady=5)

        master.update_idletasks()

        w = master.winfo_screenwidth()
        h = master.winfo_screenheight()
        mastersize = tuple(
            int(_) for _ in master.geometry().split("+")[0].split("x"))
        x = w / 2 - mastersize[0] / 2
        y = h / 2 - mastersize[1] / 2
        master.geometry("%dx%d+%d+%d" % (mastersize + (x, y)))

        if plot_now:
            self.plot()

    def choose_curves_file(self):
        input_file = \
            askopenfilename(title="Select IrmUnmix parameter file")
        if input_file:
            self.curves = IrmCurves.read_file(input_file)

    def choose_data_file(self):
        input_file = askopenfilename(title="Select IRM data file")
        if input_file:
            self.series = DataSeries.read_file(input_file)

    def plot(self):
        plot_clg_fit(self.series, self.curves)


def main():
    parser = argparse.ArgumentParser(description="usage: clgplot [options]")
    parser.add_argument("-d", "--data", dest="data_file",
                        help="Read IRM intensities from FILE.", metavar="FILE")
    parser.add_argument("-c", "--curves", dest="curves_file",
                        help="Read curve parameters from FILE.", metavar="FILE")
    parser.add_argument("-p", "--plot", action="store_true", dest="plot_now",
                        default=False, help="Plot in GUI at once")
    parser.add_argument("-n", "--no-gui", action="store_true",
                        default=False, help="Don't start the GUI")
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="Write plot to specified file")
    args = parser.parse_args()

    data = None
    if args.data_file:
        data = DataSeries.read_file(args.data_file)
        hpcr = x_for_half_max_y(data.data[0], data.data[1])
        print("H'cr = {:.5g}".format(hpcr))

    curves = None
    if args.curves_file:
        curves = IrmCurves.read_file(args.curves_file)

    if args.output:
        plot_clg_fit(data, curves, args.output)

    if not args.no_gui:
        root = tkinter.Tk()
        App(root, data=data, curves=curves,
            plot_now=args.plot_now)
        root.mainloop()


if __name__ == "__main__":
    main()
