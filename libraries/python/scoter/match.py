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

import os
import subprocess
from series import Series

def _fix_spaces(string):
    return string.replace(" ", "_")

class MatchSeriesConf(object):
    """Configuration for a series in a run of the match program.
    Note that more than one series may be specified (as a tuple)."""
    def __init__(self, series, intervals = 200, gapfile = None,
                 begin = None, end = None):
        if isinstance(series, Series):
            self.series = (series,)
        else:
            # assume a tuple or other iterable
            self.series = series
        if begin != None: self.begin = begin
        else: self.begin = max([s.positions()[0] for s in self.series])
        if end != None: self.end = end
        else: self.end = min([s.positions()[-1] for s in self.series])
        self.intervals = intervals
        self.gapfile = gapfile

    def writep(self, fh, num, name, value):
        fh.write('%-16s%s\n' % ('%s%d' % (name, num), str(value)))

    def write(self, fh, num):
        self.writep(fh, num, 'series',
                    ' '.join(map(lambda x: _fix_spaces(x.name), self.series)))
        self.writep(fh, num, 'begin', self.begin)
        self.writep(fh, num, 'end', self.end)
        self.writep(fh, num, 'numintervals', self.intervals)
        if self.gapfile != None:
            self.writep(fh, num, 'gapfile', self.gapfile)
    
class MatchConf(object):
    """Configuration for a run of the match program."""

    def __init__(self, series1, series2, params = {}, tie_points = None):
        self.params = dict(nomatch = 1e9, speedpenalty = 2,
                           targetspeed = '1:1', speedchange = 10,
                           tiepenalty = 6000, gappenalty = 100,
                           speeds = ('1:3,2:5,1:2,3:5,2:3,3:4,4:5,1:1,' +
                                     '5:4,4:3,3:2,5:3,2:1,5:2,3:1'))
        self.name = 'match'
        self.params.update(params)
        self.series1 = series1
        self.series2 = series2
        self.matchfile = self.name+'.match'
        self.logfile = self.name+'.log'
        self.tiefile = self.name+'.tie'
        self.tie_points = tie_points

    def writep(self, fh, name, value):
        if value != None:
            fh.write('%-16s%s\n' % (name, str(value)))

    def write_to(self, fh):
        self.series1.write(fh, 1)
        self.series2.write(fh, 2)
        for key, value in self.params.iteritems():
            self.writep(fh, key, value)
        self.writep(fh, 'matchfile', self.matchfile)
        self.writep(fh, 'logfile', self.logfile)
        if self.tie_points:
            self.writep(fh, 'tiefile', self.tiefile)

    def write_ties(self, path):
        if self.tie_points:
            with open(path, 'w') as fh:
                for a, b in self.tie_points:
                    if (self.series1.series[0].contains(a) and
                        self.series2.series[0].contains(b)):
                        fh.write(" {0:16.7e} {1:16.7e}\n".format(a, b))

    def run_match(self, match_path, dir_path, dummy_run = False):
        """Run the match program with this configuration, in the specified
        directory. If the directory does not exist it will be created.
        
        Args:
            match_path: path to the match binary
            dir_path: directory for match input and output data
            dummy_run: if True, match will not actually be run, but the directory
                will be created and populated with input data files
        
        Returns:
            a MatchResult object representing the results
       
        """
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        name = self.name
        conf_path = os.path.join(dir_path, name+'.conf')
        self.write_ties(os.path.join(dir_path, name+'.tie'))
        with open(conf_path, 'w') as fh:
            self.write_to(fh)
        for ss in [self.series1.series, self.series2.series]:
            for s in ss:
                filename = _fix_spaces(os.path.join(dir_path, s.name))
                s.write(filename)
        if not dummy_run:
            p = subprocess.Popen([match_path, name + '.conf'], cwd = dir_path,
                                 stdout = subprocess.PIPE, stderr = subprocess.PIPE)
            output = p.communicate() # run Match and wait for it to terminate
            result = MatchResult(self, dir_path)
            result.stdout, result.stderr = output
            result.return_code = p.returncode
            result.error = (p.returncode != 0)
            return result
        else:
            result = MatchResult(self, dir_path)
            result.error = False
            return result
        
class MatchResult(object):
    """The results of a run of the Match program"""

    def __init__(self, match_conf, dir_path):
        self.series1 = []
        for s1 in match_conf.series1.series:
            fn1 = os.path.join(dir_path, s1.name + '.new')
            if os.path.isfile(fn1):
                self.series1.append(Series.read(fn1, name=s1.name+'-tuned',
                                                col1 = 1, col2 = 2))
        match_file = os.path.join(dir_path, match_conf.matchfile)
        if os.path.isfile(match_file):
            self.match = Series.read(match_file,
                                     name = os.path.basename(dir_path)+'-rel',
                                     col1 = 1, col2 = 3)
