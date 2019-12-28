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

import logging
import numpy as np
import os.path
import scipy.signal
import scipy.stats
from math import ceil, floor, log
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class Series(object):
    """A time series for a single value.

    Consists of a series of position-value pairs (position may be time or
    depth). The data is intended to be immutable (though this is not
    enforced): the various transformation methods return updated copies
    and do not alter the original series."""

    def __init__(self, data, name="unknown", filename=None, parameter=None):
        if len(data.shape) != 2:
            raise ValueError("Data must be a rank-2 array.")
        elif data.shape[0] != 2:
            raise ValueError("Length of first dimension of data must be 2.")
        self.data = data
        self.filename = filename
        self.parameter = parameter
        if name is not None:
            self.name = name
        else:
            if filename is not None:
                self.name = os.path.basename(filename)
            else:
                self.name = None

    @classmethod
    def read(cls, filename, col1=0, col2=1, name=None, parameter=None):
        """Read a series from a text file.

        Read a series from a columnar whitespace-delimited text file. One column
        specifies the position, and another the value. If there
        is a header line (or any other non-numeric line), it is ignored.
        
        Args:
            filename: full pathname of file to read
            col1: index (0-based) of column containing position
            col2: index (0-based) of column containing value
            name: a name for the series
            parameter: name of the parameter represented by the value column
        
        Returns:
            A Series representing the data read from the file.
        
        """

        rows = []
        # 'U' for universal newlines
        with open(filename, "U") as fh:
            line_number = -1
            for line in fh.readlines():
                line_number += 1
                parts = line.split()
                if col1 >= len(parts) or col2 >= len(parts):
                    logger.warning(
                        "Not enough data fields on line %d of %s." % (
                            line_number + 1, filename))
                    continue
                if parts[col1] == "" or parts[col2] == "":
                    logger.warning("Blank data field on line %d of %s" % (
                        line_number + 1, filename))
                    continue
                try:
                    position = float(parts[col1])
                    value = float(parts[col2])
                    rows.append([position, value])
                except ValueError:
                    if line_number > 0:
                        # We don't warn if the first line is non-numeric,
                        # because it's probably a header.
                        logger.warning("Non-numeric data on line %d of %s" % (
                            line_number + 1, filename))
                    pass
        data = np.array(rows).transpose()
        return Series(data, name=name, filename=filename, parameter=parameter)

    @classmethod
    def pink1d(cls, n, rvs=scipy.stats.norm.rvs):
        k = min(int(floor(log(n) / log(2))), 6)
        pink = np.zeros((n,), float)
        m = 1
        for _ in range(k):
            p = int(ceil(float(n) / m))
            pink += np.repeat(rvs(size=p), m, axis=0)[:n]
            m <<= 1
        return pink / k

    @classmethod
    def pink1d_interp(cls, real_points=1500, interp_points=18000):
        n = interp_points
        m = real_points
        depths = np.linspace(0, 130, n)
        values_uninterp = Series.pink1d(m)
        pos = np.linspace(0, 1, m)
        values_f = interp1d(pos, values_uninterp)
        values = values_f(np.linspace(0, 1, n))
        return depths, values

    @classmethod
    def make_pink(cls, real_points=1500, interp_points=18000):
        depths, values = Series.pink1d_interp(real_points, interp_points)
        return Series(np.array([depths, values]), name="pink")

    @classmethod
    def random_rate(cls, real_points=50, interp_points=18000):
        s = Series.make_pink(real_points, interp_points)
        s = s.offset_values_by(1.0)
        return s

    def write(self, filename):
        """Write this series to the specified filename."""
        with open(filename, "w") as fh:
            for row in self.data.transpose():
                fh.write("\t".join(map(str, row)) + '\n')

    def write_to_dir(self, dir_path):
        """Write this series into the specified directory.

        The name attribute is used as the filename."""
        filename = os.path.join(dir_path, self.name)
        with open(filename, "w") as fh:
            for row in self.data.transpose():
                fh.write("\t".join(map(str, row)) + "\n")

    def copy(self, data=None, name=None, filename=None,
             suffix=None):
        """Return an identical or modified copy of this series.

        Return a copy of this series, optionally modified using one or more
        of the optional parameters."""

        # Using "is" not "==" to avoid triggering an attempted
        # element-wise object comparison in future scipy versions.
        if data is None:
            data = self.data.copy()
        if name is None:
            name = self.name
        if filename is None:
            filename = self.filename
        if suffix:
            name = name + suffix
        return Series(data, name, filename, self.parameter)

    def accumulate(self):
        new_data = self.data.copy()
        new_data[1] = np.cumsum(new_data[1])
        return self.copy(data=new_data, suffix="-ac")

    def truncate(self, limit):
        """Truncate this series at the given point.

        Remove all position-value pairs with a position greater than the
        specified limit."""
        positions = np.nonzero(self.data[0] > limit)[0]
        if positions.size > 0:
            new_data = self.data[:, :positions[0]].copy()
        else:
            new_data = self.data
        return self.copy(data=new_data, suffix='-tr')

    def clip(self, limits):
        """Clip the series to the given x-value (position) limits.

        Assumes that positions are monotonically increasing.
        The value None may be used to disable clipping at
        one end of the range."""
        xmin, xmax = limits
        start = 0
        if xmin is not None:
            while start < self.npoints() and self.data[0][start] < xmin:
                start += 1
        end = self.npoints() - 1
        if xmax is not None:
            while end >= start and self.data[0][end] > xmax:
                end -= 1
        new_data = self.data.copy()[:, start:end + 1]
        return self.copy(data=new_data, suffix="-cl")

    def clip_values(self, limits):
        assert (False)
        pass

    def npoints(self):
        """Return the number of points in this series."""
        return self.data.shape[1]

    def diff_interp(self, other, npoints=1000, norm=True):
        """Calculate a difference score using interpolation.

        Sum vertical distances between corresponding points on
        this and another curve. The sum is calculated over the
        overlapping portion of the curves.
        """
        ip1 = interp1d(self.data[0], self.data[1])
        ip2 = interp1d(other.data[0], other.data[1])
        t = 0
        start = max(self.start(), other.start())
        end = min(self.end(), other.end())
        step = (end - start) / npoints
        for i in xrange(0, npoints):
            x = start + i * step
            t += abs(ip1(x) - ip2(x))
        if norm:
            t /= npoints
        return t

    def start(self):
        return self.data[0][0]

    def end(self):
        return self.data[0][-1]

    def length(self):
        return self.end() - self.start()

    def contains(self, x, strict=True):
        """Determine whether a position is contained in this series."""
        if strict:
            return self.start() < x < self.end()
        else:
            return self.start() <= x <= self.end()

    def mean(self):
        """Return the mean of the values in this series.

        Any NaN values will be ignored."""
        return np.nanmean(self.data[1])

    def std(self):
        """Return the standard deviation of the values in this series.

        Any NaN values will be ignored."""
        return np.nanstd(self.data[1])

    def positions(self):
        """Return the values in this series."""
        return self.data[0]

    def values(self):
        """Return the values in this series."""
        return self.data[1]

    def detrend(self):
        """Return a detrended version of this series.

        Return a transformation of this series with the values linearly
        detrended. NOTE: assumes constant sampling interval."""
        new_data = self.data.copy()
        new_data[1] = scipy.signal.detrend(new_data[1])
        return self.copy(data=new_data, suffix='-dt')

    def rate(self, use_midpoints=False):
        """Turn a depth/age series into an age/rate series.

        Intended for use with the output of the Match program.
        Assumes this series is a series of depth/age correlations
        (N.B. depth is the 'x' value). Returns a series where
        the x values are ages and the y value at each age
        is the sedimentation rate at that age."""
        d = self.data[1]
        if use_midpoints:
            midpoints = 0.5 * (d[:-1] + d[1:])
            rates = np.diff(self.data[0]) / np.diff(self.data[1])
            new_data = np.array([midpoints, rates])
        else:
            xs, ys = [], []
            for i in range(len(d) - 1):
                y_diff = self.data[0][i + 1] - self.data[0][i]
                x_diff = self.data[1][i + 1] - self.data[1][i]
                rate = y_diff / x_diff
                xs.append(self.data[1][i])
                xs.append(self.data[1][i + 1])
                ys.append(rate)
                ys.append(rate)
                new_data = np.array([xs, ys])
        return self.copy(new_data, suffix="-rate")

    def mapping(self):
        """Assuming that this series consists of match output data (i.e. a
        mapping from one series to another), this method returns a function
        which converts the position (given in the scale of the first series)
        to the corresponding position in the scale of the second series."""
        return interp1d(self.data[0], self.data[1], fill_value="extrapolate")

    def warp_using(self, match_series):
        """Return a warped version of this series.

        Using the supplied series as warping tie-points, return a
        transformation of this series with the same values but with
        positions shifted according to the warping transformation."""
        mapping = match_series.mapping()
        new_series = self.copy().clip((mapping.x[0], mapping.x[-1]))
        new_series.data[0] = mapping(new_series.data[0])
        new_series.name = self.name + "-warp"
        return new_series

    def scale_to(self, series):
        """Return this series, linearly scaled to match another series.

        Transform the values of this series linearly so that their mean and
        standard deviation are equal to those of the supplied series, and
        return the new series."""
        offset = series.mean() - self.mean()
        values_offs = self.values() + offset
        scale = series.std() / self.std()
        offset_mean = np.mean(values_offs)
        values_offs_scale = \
            (values_offs - offset_mean) * scale + offset_mean
        new_data = self.data.copy()
        new_data[1] = values_offs_scale
        return self.copy(data=new_data, suffix="-sc")

    def scale_extrema_to(self, series):
        """Scale values in this series to match the extrema of another series.

        Scale the values of this series linearly in such a way that the
        minimum and maximum values are the same as the minimum and maximum
        values of another series.

        :param series: the series from which to take the minimum and maximum
        :return: a copy of this series, with the values scaled to the
           minimum and maximum of ``series``.
        """
        min_this = min(self.values())
        max_this = max(self.values())
        min_other = min(series.values())
        max_other = max(series.values())
        scale = (max_other - min_other) / (max_this - min_this)
        offset = min_other - scale * min_this
        new_values = scale * self.values() + offset
        new_data = self.data.copy()
        new_data[1] = new_values
        return self.copy(data=new_data, suffix="-sc")

    def scale_positions_by(self, factor):
        """Return this series, with the positions linearly scaled by the
        supplied factor.
        """

        new_data = self.data.copy()
        new_data[0] = self.data[0] * factor
        return self.copy(data=new_data, suffix="-psc")

    def scale_values_by(self, factor):
        """Return this series, with the values linearly scaled by the
        supplied factor.

        The scaling is relative to the mean, so the scaled values will have
        the same mean value as the original values."""
        mean = np.nanmean(self.values())
        values_offset = self.values() - mean
        values_offset_scaled = values_offset * factor
        new_data = self.data.copy()
        new_data[1] = values_offset_scaled + mean
        return self.copy(data=new_data, suffix="-sc")

    def scale_values_without_offset(self, factor):
        """Return this series, with the values linearly scaled by the
        supplied factor."""
        new_data = self.data.copy()
        new_data[1] = self.data[1] * factor

        return self.copy(new_data, suffix="-sc")

    def scale_to_other_series(self, target_series, reference_range):
        """Return this series, scaled to match another series.
        
        This method is intended to aid in splicing together two overlapping
        subseries derived from an original series, possibly with different
        scalings and offsets applied.        
        """

        target_chunk = target_series.clip(reference_range)
        this_chunk = self.clip(reference_range)
        means = [series.mean() for series in (target_chunk, this_chunk)]
        stds = [series.std() for series in (target_chunk, this_chunk)]
        scale = stds[0] / stds[1]
        new_values = (self.values() - means[1]) * scale + means[0]
        new_data = self.data.copy()
        new_data[1] = new_values
        return self.copy(data=new_data, suffix='-sc')

    def scale_std_to(self, new_std):
        """Return this series, linearly scaled.

        The scaling is relative to the mean, so the scaled values will have
        the same mean value as the original values. The scaling factor is
        such that the new series will have a standard deviation equal to
        new_std."""
        return self.scale_values_by(new_std / self.std())

    def offset_values_by(self, offset):
        """Add a constant offset to the values of this series.

        Return a new series formed by adding the supplied offset
        to each value in this series."""
        new_data = self.data.copy()
        new_data[1] = new_data[1] + offset
        return self.copy(data=new_data, suffix='-of')

    def subtract_mean(self):
        """Shift the mean value of this series to zero.

        Return a new series formed by offsetting the values of this
        series so that their mean is zero."""
        return self.offset_values_by(-self.mean())

    def smooth(self, window=5):
        """Return this series, smoothed with a running mean."""
        weightings = np.repeat(1.0, window) / window
        end_zone = int(window / 2)
        new_data = self.data.copy()
        d = new_data[1]
        new_data[1] = np.concatenate((
            d[:end_zone],
            np.convolve(d, weightings, 'valid'),
            d[-end_zone:]))
        return self.copy(data=new_data, suffix='-sm' + str(window))

    def interpolate(self, npoints=None, kind="linear"):
        """Interpolate this series to give regularly spaced x values.
        
        Args:
            npoints: number of points to interpolate to. If None,
                use the current number of points.
            kind: kind of interpolation to use. ‘linear’, ‘nearest’,
                ‘zero’, ‘slinear’, ‘quadratic, ‘cubic’ where ‘slinear’,
                ‘quadratic’ and ‘cubic’ refer to a spline interpolation
                of first, second or third order. Default: ‘linear’.
        
        Returns:
            A new series containing the interpolated data.
        """
        f = interp1d(self.data[0], self.data[1], kind=kind)
        if not npoints:
            npoints = self.npoints()
        new_xs = np.linspace(self.start(), self.end(), npoints)
        new_ys = f(new_xs)
        new_data = np.vstack((new_xs, new_ys))
        return self.copy(data=new_data, suffix='-rs')

    def get_name(self):
        """Return the name of this series."""
        if self.name is not None:
            return self.name
        else:
            return os.path.basename(self.filename)

    @classmethod
    def stack(cls, series, weights=None, dyn_weights=None,
              length=10000, name="stack"):
        """Stack the supplied collection of series."""

        starts = [s.start() for s in series]
        ends = [s.end() for s in series]
        start = min(starts)
        end = max(ends)
        if weights is None:
            weights = np.ones(len(series))
        N = length
        ips = [interp1d(s.data[0], s.data[1])
               for s in series]
        dyn_ips = None
        if dyn_weights:
            dyn_ips = [interp1d(s.data[0], s.data[1])
                       for s in dyn_weights]
        xs = np.linspace(start, end, N)
        ys = []
        for x in xs:
            nrecs = 0
            total = 0
            weight_total = 0
            for i in range(0, len(series)):
                if starts[i] <= x <= ends[i]:
                    weighted = ips[i](x) * weights[i]
                    if dyn_ips:
                        weighted *= dyn_ips[i](x)
                    total += weighted
                    weight_total += weights[i]
                    nrecs += 1
            mean = total / weight_total
            ys.append(mean)
        data = np.array([xs, ys])
        return Series(data, name=name)

    def resolution(self):
        """Return the mean resolution of this series.

        The mean resolution is calculated as the difference
        (in the X domain) between the first and last values,
        divided by the total number of records."""
        return self.length() / float(self.npoints())

    def is_even(self, tolerance=0.000001):
        """Check whether x-values are evenly spaced."""
        i = 0
        abs_tolerance = self.length() * tolerance
        for x in np.linspace(0, self.length(), self.npoints()):
            actual_x = self.data[0][i] - self.start()
            if abs(actual_x - x) > abs_tolerance:
                return False
            i += 1
        return True

    def wiggliness(self):
        """Return the wiggliness of this series."""
        ys = self.data[1]
        total = np.sum(np.abs(ys[1:] - ys[:-1]))
        return total / self.length()

    def local_wiggle(self, window=301):
        """Return a local-wiggliness series for this series."""
        # First calculate an ‘uncorrected wiggle’: for each point,
        # sum of its y-offsets from the previous and next points.
        # (At the endpoints we double the single offset which is
        # available.)
        ys = self.data[1]
        offsets = np.abs(ys[1:] - ys[:-1])
        left = np.concatenate((offsets, np.zeros(1)))
        left[0] = left[0] * 2
        right = np.concatenate((np.zeros(1), offsets))
        right[-1] = right[-1] * 2
        raw_wiggle = left + right
        # Now we have a 1-d array representing how much each
        # y-value differs from its neighbours
        wiggle = np.ones(len(ys))
        halfw = int(window / 2)
        for i in np.arange(halfw, len(ys) - halfw):
            a = i - halfw
            b = i + halfw
            total = np.sum(raw_wiggle[a:b + 1])
            distance = self.data[0][b] - self.data[0][a]
            wiggle[i] = total / distance
        # TODO do a better job on edges
        new_data = self.data.copy()
        new_data[1] = wiggle
        return self.copy(new_data, suffix="-df")

    def similar_to(self, series, tolerance=1e-6):
        """Report whether supplied series is similar to this one."""

        if self.data.shape != series.data.shape:
            return False
        max_diff = np.max(np.abs(self.data - series.data))
        return max_diff < tolerance

    def data_slice(self, start, end):
        """Return a slice of the data.

        Return the slice of the data between start and end, which are indices
        into the data array. The returned slice will include the datum at
        start, but exclude the datum (if any) at end."""

        if start >= end:
            raise ValueError("start (%d) must be less than end (%d)" %
                             (start, end))
        if start < 0:
            raise ValueError("start (%d) must be >= 0" % start)
        if end > self.npoints():
            raise ValueError("end (%d) must be <= npoints (%d)" %
                             (end, self.npoints()))
        return self.data[:, start:end]

    def data_slice_xnorm(self, start, end):
        """Return a dataslice of the data with a normalized x-range.

        Return the dataslice of the data between start and end, which are
        indices into the data array. The returned dataslice will include the
        datum at start, but exclude the datum (if any) at end. The x-values
        of the returned dataslice will be linearly transformed so as to run
        from 0 to 1."""

        if end - start <= 1:
            raise ValueError("(end - start) must be >1.")
        dataslice = self.data_slice(start, end).copy()
        xmin = dataslice[0][0]
        xmax = dataslice[0][-1]
        xlen = xmax - xmin
        dataslice[0] = (dataslice[0] - xmin) / xlen
        return dataslice

    @staticmethod
    def combine_series(specifiers, resolution):
        """Create a composite series using sections of existing series.

        Create a composite series from selected sections of other
        series, according to a supplied tuple of specifiers.
        The series are resampled to a constant interval specified by
        the resolution parameter. Each
        specifier is itself a tuple with the following structure:

        ("single", X_START, SERIES) or
        ("transition", X_START, SERIES1, SERIES2) or
        ("end", X_POSITION)

        A typical tuple of specifiers might be

        (
        ("single", 10, series1),
        ("transition", 50, series1, series2),
        ("single", 90, series2),
        ("end", 200)
        )

        "single" and "transition" both create a sequence of data points
        from X_START to the X_START or X_POSITION of the following specifier.

        "single" takes values from a single series.

        "transition" imposes a smooth transition between the two series
        throughout the specified segment, weighting them linearly according
        to the x position within the transition segment.

        "end" specifies an end-point for the previous specifier.

        :param resolution:
        :param specifiers:
        :return:
        """

        xs, ys = [], []

        for i in range(len(specifiers)):
            specifier = specifiers[i]
            specifier_type = specifier[0]
            x_start = specifier[1]
            if specifier_type == "single":
                x_end = specifiers[i + 1][1]
                series = specifier[2]
                mapping = series.mapping()
                for x in range(x_start, x_end, resolution):
                    xs.append(x)
                    ys.append(mapping(x))
            elif specifier_type == "transition":
                x_end = specifiers[i + 1][1]
                series1 = specifier[2]
                series2 = specifier[3]
                mapping1 = series1.mapping()
                mapping2 = series2.mapping()
                x_total_distance = x_end - x_start
                for x in range(x_start, x_end, resolution):
                    xs.append(x)
                    x_distance = x - x_start
                    scale = float(x_distance) / float(x_total_distance)
                    y1 = mapping1(x)
                    y2 = mapping2(x)
                    ys.append(y2 * scale + y1 * (1-scale))

            elif specifier_type == "end":
                pass
            else:
                raise ValueError("Unknown specifier type {}".format(
                                               specifier_type))

        return Series(np.array([xs, ys]))

    def linear_position_transform(self, tie_points):
        """Linearly transform the positions of this series.

        In practice, this method can be used to transform a depth series
        to a time series. The position transformation is defined by a
        supplied list of time points mapping a position in the old system
        (e.g. depth) to a position in the new system (e.g. time). The values
        remain the same, but the positions are mapped linearly from the old
        system to the new system.

        :param tie_points: a list of (p1, p2) pairs, where p1 is a position
               in this series’ system and p2 is a position in the system
               of the series to be produced
        :return: a new series with the same values as this one, but with
                the positions linearly mapped to the new system
        """

        xs_old = self.data[0]
        tie_points_flipped = zip(*tie_points)
        transform_xs = tie_points_flipped[0]
        transform_ys = tie_points_flipped[1]

        f = interp1d(transform_xs, transform_ys, fill_value="extrapolate")
        # xs_new = np.interp(xs_old, transform_xs, transform_ys)
        xs_new = f(xs_old)

        return self.copy(data=np.array([xs_new, self.data[1]]))

    def wrap_values(self, maximum=180, period=360):
        """Wrap values into a periodic range.

        This method is intended for declinations and other periodic values.
        For each value in the series, it will subtract enough multiples of
        the period to bring the value under the maximum (in effect, a modulo
        function with an offset). The method returns a new series.

        :param maximum: the maximum value to allow in the new series
        :param period: the period of the values
        :return: a new series with the values wrapped into the specified period
        """
        def wrapper(value):
            while value > maximum:
                value -= period
            return value

        v_wrapper = np.vectorize(wrapper)
        return self.copy(data=np.array([self.data[0], v_wrapper(self.data[1])]))
