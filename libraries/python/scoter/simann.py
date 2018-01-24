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

import random
import math
import logging

logger = logging.getLogger(__name__)

class AdaptiveSchedule(object):
    """Lowers the temperature after a set number of accepted changes.
    """

    def __init__(self, start_temp, end_temp,
                 changes_per_tstep, steps_per_tstep,
                 rate = 0.99):
        self.changes_per_tstep = changes_per_tstep
        self.steps_per_tstep = steps_per_tstep
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.rate = rate
        self.reset()

    def _reset_tstep(self):
        self.step_t = 0
        self.changes_tstep = 0

    def reset(self):
        self.step = 0
        self.temp = self.start_temp
        self._reset_tstep()

    def advance(self, accepted):
        self.step += 1
        self.step_t += 1
        if accepted: self.changes_tstep += 1
        if (self.changes_tstep >= self.changes_per_tstep or
            self.step_t > self.steps_per_tstep):
            self.temp = self.temp * self.rate
            self._reset_tstep()

    def finished(self):
        return self.temp <= self.end_temp

class SimpleSchedule(object):

    def __init__(self, start_temp, end_temp, rate):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.rate = rate
        self.reset()

    def reset(self):
        self.step = 0
        self.temp = self.start_temp

    def advance(self, accepted):
        self.temp = self.temp * self.rate
        self.step += 1

    def finished(self):
        return self.temp <= self.end_temp

class Annealer(object):

    def __init__(self, start, rnd = random.Random()):
        self.start = start # initial solution
        self.scores = []
        self.rnd = rnd

    def run(self, schedule,
            restarts = 0, callback = None):
        soln_restart = self.start
        for restart_n in xrange(restarts + 1):
            schedule.reset()
            soln_current = soln_restart
            step = 0
            while not schedule.finished():
                soln_new = soln_current.make_variant()
                if soln_new.score() < soln_restart.score():
                    soln_restart = soln_new
                diff = soln_new.score() - soln_current.score()
                accepted = (diff < 0 or
                            math.exp( -diff / schedule.temp ) >
                            self.rnd.random())
                if accepted: soln_current = soln_new
                logger.info("Restart %3d Step %9d Temp %12d" %
                            (restart_n, step, schedule.temp))
                # soln_current.printself()
                if callback:
                    abort = callback(soln_current, soln_new, schedule)
                    if abort: return False # terminated prematurely
                schedule.advance(accepted)
                self.scores.append(soln_current.score())
                step += 1
        self.soln_best = soln_current
        return True # terminated normally

    def output_scores(self, filename):
        with open(filename, 'w') as fh:
            for score in self.scores:
                fh.write('%f\n' % score)
