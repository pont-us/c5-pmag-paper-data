# -*- coding: utf-8 -*-
# cython: profile=True

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

import numpy as np
from series import Series
import random
import fractions
import pylab
import matplotlib
import math

class Bseries(object):
    """Block-structed series.

    This class wraps one or more Series and provides methods which allow them
    to be processed as equal-sized blocks rather than individual
    data-points."""

    def __init__(self, series, nblocks):
        if isinstance(series, Series):
            self.series = (series,)
        else:
            self.series = series
        self.nblocks = nblocks
        lengths = [s.npoints() for s in self.series]
        if lengths.count(lengths[0]) != len(lengths):
            raise ValueError("All series must be of equal length " + str(lengths))
        N = self.series[0].npoints()
        self.block_starts = map(int, np.linspace(0, N, nblocks,
                                                 endpoint=False))

    def get_block_start(self, block):
        """Return the index of the first point in a block.

        If an index is given which is equal to the number of blocks,
        n+1 will be returned, where n is the last index in the last
        block. This may be thought of as the first index in a
        'virtual' block immediately following the last actual
        block, and is useful in determining the end of the last
        block."""
        if block<0:
            raise IndexError("Negative block index given (%d)" % block)
        if block<self.nblocks: return self.block_starts[block]
        if block==self.nblocks: return self.series[0].npoints()
        raise IndexError("Block index too high (%d > %d)" % (block, self.nblocks))

    def get_xrange(self, start, end):
        """Return series indices corresponding to given blocks.

        start is the index of the first block in the range. end is one
        greater than the index of the last block. e.g. get_xrange(3, 7) will
        return series indices encompassing blocks 3, 4, 5, and 6. Values are
        returned as a 2-tuple."""

        return (self.get_block_start(start),
                self.get_block_start(end))

    def get_xrange_vals(self, start, end):
        """Return series values corresponding to given blocks."""
        i0, i1 = self.get_xrange(start, end)
        return [(s.data[0][i0], s.data[0][i1-1]) for s in self.series]

    def get_slice_xnorm(self, start, end):
        """Return block of x and y data corresponding to given blocks.

        start is the index of the first block in the range.
        end is one greater than the index of the last block.
        x-values of returned data will be scaled to [0,1]."""
        s, e = self.get_xrange(start, end)
        try:
            return [series.data_slice_xnorm(s, e) for series in self.series]
        except RuntimeWarning:
            print start, end, s, e
            assert(False)

    @classmethod
    def compare(cls, bser0, bser1, range0, range1):
        """Return a similarity score for two series ranges.

        Calculate a similarity metric between the specified blocks of the two
        supplied series. Ranges are scaled so that their endpoints match. A
        score of zero indicates identical ranges. The higher the score, the
        greater the difference."""

        start0, end0 = range0
        start1, end1 = range1
        data0 = bser0.get_slice_xnorm(start0, end0)
        data1 = bser1.get_slice_xnorm(start1, end1)

        def difference(sample_xs, sample_ys, ref_xs, ref_ys):
            interp_ys = np.interp(sample_xs, ref_xs, ref_ys)
            return np.sum(np.abs(sample_ys - interp_ys))

        total = 0

        for i in range(len(data0)):
            xs0, ys0 = data0[i][0], data0[i][1]
            xs1, ys1 = data1[i][0], data1[i][1]
            total += (difference(xs0, ys0, xs1, ys1) +
                     difference(xs1, ys1, xs0, ys0)) ** 2
           
        return total

class Bcomparator(object):
    """A caching comparator for a pair of Bseries"""

    def __init__(self, series1, series2):
        self.series1 = series1
        self.series2 = series2
        self.ccache = {}

    def compare(self, range1, range2):
        key = (range1, range2)
        if key not in self.ccache:
            self.ccache[key] = (
                Bseries.compare(self.series1, self.series2,
                                range1, range2))
        return self.ccache[key]

    def compare2(self, a, b, c, d):
        return self.compare((a, b), (c, d))

    def dump(self, fh):
        for k in self.ccache.keys():
            fh.write('%d\t%d\t%d\t%d\t%f\n' %
                     (k[0][0], k[0][1], k[1][0], k[1][1], self.ccache[k]))

    def plot_pylab(self, xlabel = None, ylabel = None, title = None):
        max_score = max(self.ccache.values())
        scale = 1./2000.
        cNorm  = matplotlib.colors.\
            Normalize(vmin=0, vmax=math.log(max_score*scale+1))
        cMap = pylab.get_cmap('gist_rainbow')
        cmapper = matplotlib.cm.ScalarMappable(norm = cNorm, cmap = cMap)

        pylab.subplots_adjust(left=0.06, bottom=0.06, right=0.97, top=0.95)
        if xlabel != None: pylab.xlabel(xlabel)
        if ylabel != None: pylab.ylabel(ylabel)
        if title != None: pylab.title(title)

        pylab.axes().set_axis_bgcolor('black')
        for ((x1, x2), (y1, y2)), score in self.ccache.items():
            fettled_score = math.log(score*scale+1)
            pylab.plot([x1, x2], [y1, y2],
                       color = cmapper.to_rgba(fettled_score),
                       zorder = -score)

class Bwarp(object):
    """A block-level mapping between two Bseries."""

    def __init__(self, series0, series1,
                 comparator = None, runs = None,
                 name = 'Unknown',
                 max_rate = 4,
                 rc_penalty = 0,
                 rnd = random.Random()):
        self.series = (series0, series1)
        self.name = name
        self.rc_penalty = rc_penalty
        self.rnd = rnd
        if comparator:
            self.comp = comparator
        else:
            self.comp = Bcomparator(series0, series1)
        if self.series[0].nblocks != self.series[1].nblocks:
            raise ValueError("Differing series lengths (%d, %d)" %
                             (self.series[0].nblocks,
                              self.series[1].nblocks))
        self.nblocks = self.series[0].nblocks
        if runs:
            if len(runs) != 2:
                raise ValueError('runs must be of length 2')
            if len(runs[0]) != len(runs[1]):
                raise ValueError('run-sequences must be of equal length')
            blocks0 = sum(runs[0])
            if blocks0 != sum(runs[1]):
                raise ValueError('run-sequences must contain an ' +
                                 'equal number of blocks')
            if blocks0 != series0.nblocks:
                raise ValueError('Mismatch between series blocks and ' +
                                 'warp blocks (%d / %d).' %
                                 (series0.nblocks, blocks0))
            self.runs = runs
        else:
            self.runs = ([1] * self.nblocks, [1] * self.nblocks)
        self.min_rate = 1
        self.max_rate = max_rate

    def _choose_slice(self):
        rs = self.runs
        max_len = len(rs[0]) / 8
        slice_len = self.rnd.randint(1, max_len)
        start = self.rnd.randint(0, len(rs[0]) - slice_len)
        return start, slice_len

    def _add_blocks_slice(self, runs, start, slice_len):
        added = 0
        for i in xrange(slice_len):
            target = start + i
            if runs[target] < self.max_rate:
                nblocks = self.rnd.randint(1, self.max_rate - runs[target])
                added += nblocks
                runs[target] += nblocks
            else:
                nblocks = self.rnd.randint(1, self.max_rate)
                added += nblocks
                total = runs[target] + nblocks
                half_ish = total // 2
                runs[target] = half_ish
                runs.insert(target, total - half_ish)
                start += 1
        return added, start

    def _remove_blocks_slice(self, runs, slice_len, start = None):
        if start==None: start = self.rnd.randint(0, len(runs) - slice_len)
        for i in xrange(slice_len):
            target = start + i
            if runs[target] > self.min_rate:
                runs[target] -= 1
            else:
                runs.pop(target)
                start -= 1
        return slice_len

    def _add_run(self, s):
        runs = self.runs[s]
        # Find all runs large enough to split,
        big_runs = [i for i in range(0, len(runs)) if runs[i] > self.min_rate]
        # pick one at random,
        target = self.rnd.choice(big_runs)
        # remove half its blocks,
        half_ish = runs[target] // 2
        runs[target] = runs[target] - half_ish
        # and insert a new adjacent run containing them.
        runs.insert(target, half_ish)

    def _remove_run(self, s):
        runs = self.runs[s]
        # Find all runs small enough to remove,
        small_runs = [i for i in range(0, len(runs))
                      if runs[i] < self.max_rate // 2]
        # Check that we have enough
        if len(small_runs) < 2:
            #for i in 0, 1:
            #    print self.runs[i],
            #    print '*' if i==s else ''
            #print('Not enough small runs.')
            return False
        # pick two at random,
        target1, target2 = self.rnd.sample(small_runs, 2)
        # transfer all blocks from run 1 to run 2,
        runs[target2] += runs[target1]
        # and remove run 1.
        runs.pop(target1)
        return True
        
    def clean_runs(self):
        """Put the runs into a canonical form.

        The same warping function can have more than one representation if
        corresponding runs have common divisors. This function reduces all
        run-pairs to their lowest terms; that is, it splits any run-pairs
        with common factors. Thus, the warp [2,1] : [2,1] becomes [1,1,1] :
        [1,1,1], and the warp [4,1] : [2,1] becomes [2,2,1] : [1,1,1].
        """
        new0 = [None] * self.nblocks # pre-dimension for speed
        new1 = [None] * self.nblocks
        run = 0
        for i in xrange(0, len(self.runs[0])):
            len0 = self.runs[0][i]
            len1 = self.runs[1][i]
            if len0==1 or len1==1: # handle the common case efficiently
                new0[run] = len0
                new1[run] = len1
                run += 1
            else:
                gcd = fractions.gcd(len0, len1)
                if gcd > 1:
                    for j in xrange(gcd):
                        new0[run + j] = len0 / gcd
                        new1[run + j] = len1 / gcd
                    run += gcd
                else:
                    new0[run] = len0
                    new1[run] = len1
                    run += 1
        self.runs = (new0[:run], new1[:run])

    def _rnd_near(self, target, semirange, maximum):
        result = target + self.rnd.randint(-semirange, semirange)
        if result < 0: result = 0
        if result > maximum: result = maximum
        return result

    def _change_rates_one(self, s, start, slice_len):
        runs_s = self.runs[s]
        num_added, add_point = self._add_blocks_slice(runs_s, start, slice_len)
        window = 10 # len(runs_s) // 6
        remove_point = self._rnd_near(add_point, window,
                                       len(runs_s) - num_added)
        self._remove_blocks_slice(runs_s, num_added, remove_point)

    def change_rates_both(self):
        start, slice_len = self._choose_slice()
        #for s in 0, 1:
        for s in (self.rnd.randint(0, 1),):
            self._change_rates_one(s, start, slice_len)
        s = 0
        t = 1 - s
        runs_s = self.runs[s]
        while len(runs_s) < len(self.runs[t]):
            success = self._remove_run(t)
            if not success:
                self._add_run(s)
        while len(runs_s) > len(self.runs[t]): self._add_run(t)
        if self.rnd.random() < 0.001: self.clean_runs()
        #self.printself()

    def slice_swap(self):
        rs = self.runs
        max_len = self.nblocks // 6
        slice_len = self.rnd.randint(1, max_len)
        a = self.rnd.randint(0, len(rs[0]) - slice_len)
        b = self._rnd_near(a, max_len, len(rs[0]) - slice_len)
        for i in xrange(slice_len):
            for j in 0, 1:
                rs[j][a+i], rs[j][b+i] = rs[j][b+i], rs[j][a+i]

    def slice_reverse(self):
        rs = self.runs
        max_len = self.nblocks // self.max_rate
        slice_len = self.rnd.randint(1, max_len)
        a = self.rnd.randint(0, len(rs[0]) - slice_len)
        for j in 0, 1:
            b = rs[j][a:a+slice_len]
            b.reverse()
            rs[j][a:a+slice_len] = b

    def make_variant(self):
        copy = Bwarp(self.series[0], self.series[1],
                     self.comp, (self.runs[0][:], self.runs[1][:]),
                     self.name, self.max_rate, self.rc_penalty,
                     rnd = self.rnd)
        r = self.rnd.random()
        if r < 0.2: copy.slice_reverse()
        #elif r < 0.2: copy.slice_swap()
        else:
            copy.change_rates_both()
        return copy

    def rate_change_score(self):
        """Return a rate change score for this warp.
 
        The more the rate changes, the higher the score."""
       
        total = 0
        prev = None
        for i in xrange(len(self.runs[0])):
            a, b = self.runs[0][i], self.runs[1][i]
            # a *= self.max_rate
            rate = 5 * math.log(self.max_rate * (float(a) / float(b)))
            # print rate
            if prev != None:
                diff = abs(rate - prev) ** 2
                total += diff
            prev = rate
        return total

    def score(self):
        """Calculate the goodness-of-match score for this warp."""
        # Profiling has shown that this method is a hot-spot.
        # There's not much to optimize, however.
        if hasattr(self, '_score'): return self._score
        pos0, pos1 = 0, 0
        total = 0
        run0, run1 = self.runs[0], self.runs[1]
        comp = self.comp.compare
        for i in xrange(len(run0)):
            new0, new1 = pos0 + run0[i], pos1 + run1[i]
            total += comp((pos0, new0), (pos1, new1))
            pos0, pos1 = new0, new1
        if self.rc_penalty != 0:
            total += self.rate_change_score() * self.rc_penalty
        self._score = total
        return total

    @classmethod
    def _to_string(cls, seq):
        table = ('.', ':', 'I', '#', 'M', '6', '7', '8', '9', 'A', 'B', 'C')
        symbols = [table[x-1] for x in seq]
        return ''.join(symbols)

    def to_strings(self):
        return (self._to_string(self.runs[0]),
                self._to_string(self.runs[1]))

    @classmethod
    def from_string(cls, seq):
        table = dict(zip(' .:I#M6789', range(10)))
        return [table[x] for x in seq]

    def printself(self, with_score = False):
        for s in self.to_strings():
            print s
        if with_score: print self.score()

    def apply(self, series, subseries = 0):
        """Warp one of the series using the other as a template.

        :param series: the index of the ``Bseries`` to warp, 0 or 1
        :type series: ``int``
        :param subseries: for multi-series ('tandem') ``Bseries``, the index
        of the actual ``Series`` within the ``Bseries``
        :return: the series, warped to fit the other series
        :rtype: ``Series``
        :raise ValueError: when ``series`` is not 0 or 1
        """

        if series not in (0, 1):
            raise ValueError("series must be 0 or 1.")

        warpee = series
        target = 1 - series
        warpee_bs = self.series[warpee]
        target_bs = self.series[target]

        # Make a copy of the warpee's data. We'll keep the
        # y values but go through it overwriting the x values
        # with their warped counterparts.
        new_data = self.series[warpee].series[subseries].data.copy()
        
        pos_w, pos_t = 0, 0 # warpee, target
        for run_n in range(len(self.runs[0])):
            len_w = self.runs[warpee][run_n]
            len_t = self.runs[target][run_n]

            # get the endpoints of the warpee and target runs
            w0, w1 = warpee_bs.get_xrange(pos_w, pos_w + len_w)
            t0, t1 = target_bs.get_xrange(pos_t, pos_t + len_t)
            wx0 = warpee_bs.series[subseries].data[0][w0]
            wx1 = warpee_bs.series[subseries].data[0][w1-1]
            tx0 = target_bs.series[subseries].data[0][t0]
            tx1 = target_bs.series[subseries].data[0][t1-1]

            # calculate scaling between warpee-x and target-x
            scale = (tx1 - tx0) / (wx1 - wx0)

            # apply the mapping to the warpee data within the run
            w0 = warpee_bs.get_block_start(pos_w)
            w1 = warpee_bs.get_block_start(pos_w + len_w)
            for i in xrange(w0, w1):
                new_data[0][i] = (new_data[0][i] - wx0) * scale + tx0

            # update block positions
            pos_w += len_w
            pos_t += len_t
            
        return Series(new_data, warpee_bs.series[subseries].name + "-wp")

    def plot_pylab(self, scale = 1, colour = 'white', invert = False):
        xs, ys = self.get_coords(scale, invert)
        pylab.plot(xs, ys, color = colour, zorder = -1e100, lw=5)
        pylab.plot(xs, ys, color = colour, zorder = 1e100, lw=1)

    def get_coords(self, scale = 1., invert = False):
        xs = [0]
        ys = [0]
        x0, y0 = 0, 0
        for i in xrange(len(self.runs[0])):
            x1 = x0 + self.runs[0][i]
            y1 = y0 + self.runs[1][i]
            xs.append(x1 * scale)
            ys.append(y1 * scale)
            x0, y0, = x1, y1
        if invert: return ys, xs
        else: return xs, ys

    def get_rates(self, scale = 1., invert = False, subseries = 0,
                  xoffset = 0, yoffset = 0):
        xs, ys = [], []
        r, s = self.get_run(0), self.get_run(1)
        if invert: r, s = s, r
        block_n = 0
        for run_n in range(len(r)):
            i0, i1 = block_n, block_n + r[run_n]
            x0, x1 = self.series[0].get_xrange_vals(i0, i1)[subseries]
            rate = float(s[run_n]) / float(r[run_n])
            xs.extend([x0 + xoffset, x1 + xoffset])
            ys.extend([rate * scale + yoffset,
                       rate * scale + yoffset])
            block_n += r[run_n]
        return xs, ys

    def get_rates_as_series(self, **args):
        xs, ys = self.get_rates(**args)
        return Series(np.array([xs, ys]))

    def get_run(self, run):
        return self.runs[run]

    def scale_up(self):
        def repeat_gen(iterable):
            for x in iterable:
                yield x
                yield x

        def double_up(xs):
            return list(repeat_gen(xs))

        nblocks = self.series[0].nblocks
        doubled_series_0 = Bseries(self.series[0].series, nblocks * 2)
        doubled_series_1 = Bseries(self.series[1].series, nblocks * 2)

        copy = Bwarp(doubled_series_0, doubled_series_1,
                     None,
                     (double_up(self.runs[0][:]),
                      double_up(self.runs[1][:])),
                     self.name, self.max_rate, self.rc_penalty)
        return copy
