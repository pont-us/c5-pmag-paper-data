#!/usr/bin/env python
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

import os.path
import sys
from series import Series
from simann import Annealer, AdaptiveSchedule
from block import Bwarp, Bseries
from plot import WarpPlotter
from match import MatchConf, MatchSeriesConf
import logging
import math
import random
import shutil
import tempfile
from collections import namedtuple
import ConfigParser
import argparse

SCOTER_VERSION = "0.00"

def _find_executable_noext(leafname):
    """Helper function for find_executable."""
    def is_exe(supplied_path):
        return os.path.isfile(supplied_path) and os.access(supplied_path, os.X_OK)

    supplied_path = os.path.split(leafname)[0]
    if supplied_path:
        if is_exe(leafname):
            return leafname
    else:
        logger.info("Looking for %s on %s" % (leafname, os.environ["PATH"]))
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, leafname)
            if is_exe(exe_file):
                return exe_file

    return None

def find_executable(leafname):
    """Look for an executable on the current system path.
    
    For the name "foo.bar", this function will first try to find an
    executable with exactly that name; if this is not found, it will
    look for "foo.bar.exe"; if this is not found either, it will look
    for "foo".
    
    Args:
        leafname: the name of the executable file (extension optional)
    Returns:
        the full path of the executable, or None if it cannot be found
    """
    path = _find_executable_noext(leafname)
    if (path == None):
        path = _find_executable_noext(leafname+".exe")
    if (path == None):
        path = _find_executable_noext(os.path.splitext(leafname)[0])
    logger.info("Resolved executable: %s -> %s" % (leafname, path))
    return path


ScoterConfigBase = namedtuple("ScoterConfigBase", """
config_type      scoter_version

interp_active    interp_type         interp_npoints       detrend
normalize        weight_d18o         weight_rpi           max_rate
make_pdf         live_display        precalc

sa_active
sa_intervals     temp_init           temp_final           cooling
max_changes      max_steps           rc_penalty           random_seed

match_active
match_intervals  match_nomatch       match_speed_p        match_tie_p
match_target_speed                   match_speedchange_p
match_gap_p      match_rates         match_path

target_d18o_file record_d18o_file    target_rpi_file      record_rpi_file
target_start     target_end          record_start         record_end
output_dir       debug
""")

class ScoterConfig(ScoterConfigBase):
    """A configuration for Scoter.
    
    This class encapsulates all the information required for a Scoter
    run.    
    """
    
    def __new__(cls,
                  config_type = "plain_scoter",
                  scoter_version = SCOTER_VERSION,
                  interp_active = True,
                  interp_type = "linear",
                  interp_npoints = -2, # -2 min, -1 max, 0 undef, >0 actual #points
                  detrend = "linear",
                  normalize = True,
                  weight_d18o = 1.0,
                  weight_rpi = 1.0,
                  max_rate = 4,
                  make_pdf = False,
                  live_display = False,
                  precalc = False,
                  sa_active = True,
                  sa_intervals = 64,
                  temp_init = 1.0e3,
                  temp_final = 1.0,
                  cooling = 0.95,
                  max_changes = 5,
                  max_steps = 200,
                  rc_penalty = 0.,
                  random_seed = 42,
                  match_active = True,
                  match_intervals = 64,
                  match_nomatch = 1e12,
                  match_speed_p = 0.0,
                  match_tie_p = 100,
                  match_target_speed = "1:1",
                  match_speedchange_p = 1.0,
                  match_gap_p = 1,
                  match_rates = "1:4,1:3,1:2,2:3,3:4,1:1,4:3,3:2,2:1,3:1,4:1",
                  match_path = "", # empty string => look for match on current path
                  target_d18o_file = "",
                  record_d18o_file = "",
                  target_rpi_file = "",
                  record_rpi_file = "",
                  target_start = -1,
                  target_end = -1,
                  record_start = -1,
                  record_end = -1,
                  output_dir = "",
                  debug = ""
                  ):

        return super(ScoterConfig, cls).__new__\
            (cls, config_type, scoter_version,
             interp_active, interp_type, interp_npoints, detrend,
             normalize, weight_d18o, weight_rpi, max_rate,
             make_pdf, live_display, precalc,
             sa_active, sa_intervals, temp_init, temp_final,
             cooling, max_changes, max_steps,
             rc_penalty, random_seed,
             match_active, match_intervals, match_nomatch, match_speed_p,
             match_tie_p, match_target_speed, match_speedchange_p,
             match_gap_p, match_rates, match_path,
             target_d18o_file, record_d18o_file,
             target_rpi_file, record_rpi_file,
             target_start, target_end, record_start, record_end, output_dir,
             debug)
    
    def write_to_file(self, filename):
        """Write this configuration to a ConfigParser file.
        """
        parser = ConfigParser.RawConfigParser()
        cfgdict = self._asdict()
        for key, value in cfgdict.items():
            parser.set("DEFAULT", key, value)
        with open(filename, "wb") as fh:
            parser.write(fh)
    
    @classmethod
    def read_from_file(cls, filename):
        """Create a new ScoterConfig from a ConfigParser file.
        
        Args:
            filename: full path to ConfigParser file containing a
                Scoter configuration
        
        Returns:
            a ScoterConfig initialized from the supplied file            
        """
        default_config = ScoterConfig()
        default_dict = default_config._asdict()
        cp = ConfigParser.RawConfigParser(default_dict)
        cp.read(filename)
        s = "DEFAULT"
        return ScoterConfig(
            config_type = cp.get(s, "config_type"),
            scoter_version = cp.get(s, "scoter_version"),
            interp_type = cp.get(s, "interp_type"),
            interp_npoints = cp.getint(s, "interp_npoints"),
            detrend = cp.get(s, "detrend"),
            normalize = cp.getboolean(s, "normalize"),
            weight_d18o = cp.getfloat(s, "weight_d18o"),
            weight_rpi = cp.getfloat(s, "weight_rpi"),
            max_rate = cp.getint(s, "max_rate"),
            make_pdf = cp.getboolean(s, "make_pdf"),
            live_display = cp.getboolean(s, "live_display"),
            precalc = cp.getboolean(s, "precalc"),
            sa_active = cp.getboolean(s, "sa_active"),
            sa_intervals = cp.getint(s, "sa_intervals"),
            temp_init = cp.getfloat(s, "temp_init"),
            temp_final = cp.getfloat(s, "temp_final"),
            cooling = cp.getfloat(s, "cooling"),
            max_changes = cp.getint(s, "max_changes"),
            max_steps = cp.getint(s, "max_steps"),
            rc_penalty = cp.getfloat(s, "rc_penalty"),
            random_seed = cp.getint(s, "random_seed"),
            match_active = cp.getboolean(s, "match_active"),
            match_intervals = cp.getint(s, "match_intervals"),
            match_nomatch = cp.getfloat(s, "match_nomatch"),
            match_speed_p = cp.getfloat(s, "match_speed_p"),
            match_tie_p = cp.getfloat(s, "match_tie_p"),
            match_target_speed = cp.get(s, "match_target_speed"),
            match_speedchange_p = cp.getfloat(s, "match_speedchange_p"),
            match_gap_p = cp.getfloat(s, "match_gap_p"),
            match_rates = cp.get(s, "match_rates"),
            match_path = cp.get(s, "match_path"),
            target_d18o_file = cp.get(s, "target_d18o_file"),
            record_d18o_file = cp.get(s, "record_d18o_file"),
            target_rpi_file = cp.get(s, "target_rpi_file"),
            record_rpi_file = cp.get(s, "record_rpi_file"),
            target_start = cp.getfloat(s, "target_start"),
            target_end = cp.getfloat(s, "target_end"),
            record_start = cp.getfloat(s, "record_start"),
            record_end = cp.getfloat(s, "record_end"),
            output_dir = cp.get(s, "output_dir"),
            debug = cp.get(s, "debug")
            )

class Scoter(object):
    """Scoter correlates geological records with reference curves.
    
    This is the central class of the Scoter package, and provides high-level
    methods to control the correlation process: reading data series,
    preprocessing them, and running the actual correlation. It provides a simple
    API suitable for use from the command line, from the companion GUI ScoterGui,
    or from other Python code importing the scoter module. 
    """
    
    def __init__(self):
        self.parent_dir = os.path.dirname(os.path.realpath(__file__))
        self.default_match_path = find_executable("match")
        self.match_dir = None
        self.log_file = None
        self.aligned_sa = None
        self._init_data_structures()
    
    def _rel_path(self, filename):
        """Resolve a filename relative to the parent directory of this script."""
        return os.path.join(self.parent_dir, filename)

    def _init_data_structures(self):
        """Initialize data structures."""
        self.series = [[None, None],[None, None]]
        self.filenames = [["", ""],["", ""]]
    
    def read_data(self, role, parameter, filename, base_dir = None):
        """Read a data series.
        
        Read a data series (record or target curve) into Scoter.
        
        Args:
            role: 0 for record, 1 for target
            parameter: 0 for d18O, 1 for RPI
            filename: path to data file
                If a filename of "" is supplied, read_data will ignore
                it and return with no error.
            base_dir: base directory used to resolve filename if it
                is a relative path
        """
        assert(0 <= role <= 1)
        assert(0 <= parameter <= 1)
        param_name = ("d18o", "rpi")[parameter]
        if os.path.isabs(filename):
            full_path = filename
        else:
            if base_dir == None: return
            full_path = os.path.join(base_dir, filename)
        if full_path != "" and os.path.isfile(full_path):
            logger.debug("Reading file: %d %d %s" % (role, parameter, full_path))
            self.filenames[role][parameter] = full_path
            self.series[role][parameter] = Series.read(full_path, parameter=param_name)
    
    def has_series(self, role, parameter):
        return (self.series[role][parameter] != None)
    
    def clear_data(self, role, parameter):
        """Clear a data series.
        
        Remove a previously read data series (record or target curve) from Scoter.
        
        Args:
            role: 0 for record, 1 for target
            parameter: 0 for d18O, 1 for RPI
        """
        assert(0 <= role <= 1)
        assert(0 <= parameter <= 1)
        self.series[role][parameter] = None
        self.filenames[role][parameter] = ""

    def read_data_using_config(self, config, base_dir = None):
        """Read data files specified in the supplied configuration.
        
        Data file paths in the configuration can be relative or absolute.
        If they are relative, they will be resolved relative to the 
        supplied base directory.
        
        Args:
            config: a ScoterConfig object
            base_dir: base directory for non-absolute filenames in configuration
        """
        self.read_data(0, 0, config.record_d18o_file, base_dir)
        self.read_data(0, 1, config.record_rpi_file, base_dir)
        self.read_data(1, 0, config.target_d18o_file, base_dir)
        self.read_data(1, 1, config.target_rpi_file, base_dir)
        
    def preprocess(self, config):
        """Preprocess data sets in preparation for correlation.
        
        Args:
            config: a ScoterConfig object
        """
        
        # make sure we actually have enough data to work with
        assert((self.series[0][0] != None and self.series[1][0] != None) or
               (self.series[0][1] != None and self.series[1][1] != None))
        
        # Each series is a tuple of parallel records of different parameters
        series_picked = [[], []]
        for parameter in (0, 1):
            # Do we have this parameter as both record and target?
            if self.series[0][parameter] != None and self.series[1][parameter] != None:
                # If so, store for matching
                for role in (0, 1):
                    series_picked[role].append(self.series[role][parameter])

        # series_picked will now be something like
        # [[record_d18O, record_RPI] , [target_d18O, target_RPI]]
        # or for a non-tandem match something like
        # [[record_d18O] , [target_d18O]]
        
        self.n_record_types = len(series_picked[0])

        series_picked_flat = series_picked[0] + series_picked[1]
        series_npointss = [s.npoints() for s in series_picked_flat]
        interp_npoints = None
        if config.interp_active:
            if config.interp_npoints == -2: # use minimum
                interp_npoints = min(series_npointss)
            elif config.interp_npoints == -1: # use maximum
                interp_npoints = min(series_npointss)
            elif config.interp_npoints > 0:
                assert(hasattr(config, "interp_npoints"))
                interp_npoints = config.interp_npoints
            else:
                raise Exception("Illegal interp_npoints value: %d" % interp_npoints)
        
        record_start = config.record_start if config.record_start > -1 else 0
        target_start = config.target_start if config.target_start > -1 else 0
        
        record_end = min([s.end() for s in series_picked[0]] +
                            ([config.record_end] if config.record_end > -1 else []))
        target_end = min([s.end() for s in series_picked[1]] +
                            ([config.target_end] if config.target_end > -1 else []))
        
        logger.debug(str(record_end))
        
        series_truncated = [map(lambda s: s.clip((record_start, record_end)), series_picked[0]),
                            map(lambda s: s.clip((target_start, target_end)), series_picked[1])]
        
        def preproc(series):
            result = series
            if config.detrend == "submean":
                result = result.subtract_mean()
            elif config.detrend == "linear":
                result = result.detrend()
            if config.interp_active:
                result = result.interpolate(interp_npoints, config.interp_type)
                logger.debug("Interpolating to "+str(interp_npoints))
            if config.normalize:
                if series.parameter == "d18o":
                    result = result.scale_std_to(config.weight_d18o)
                    logger.debug("Scaling to: %f" % config.weight_d18o)
                elif series.parameter == "rpi":
                    result = result.scale_std_to(config.weight_rpi)
                    logger.debug("Scaling to: %f" % config.weight_rpi)
                else:
                    raise Exception("Unknown parameter type: %s" % series.parameter)
            return result

        self.series_preprocessed = [map(preproc, series_truncated[0]),
                                    map(preproc, series_truncated[1])]
        
        # Rename series to reflect their role in the correlation. Apart from anything
        # else, this ensures that there won't be any name clashes if they are written
        # to files for the use of the Match program.
        for series in self.series_preprocessed[0]:
            series.name = "record_" + series.parameter + ".data"
        for series in self.series_preprocessed[1]:
            series.name = "target_" + series.parameter + ".data"

    def correlate_sa(self, known_line, config, callback_obj):
        """Perform a correlation using simulated annealing.
        
        Args:
            known_line: known correlation curve (for display when testing) (optional)
            config: a ScoterConfig object
            callback_obj: callback object to monitor progress (optional)
                This object must provide the methods:
                    simann_callback_update(self, soln_current, soln_new, percentage)
                    simann_callback_finished(self, status)
                    simann_check_abort(self)
        
        Returns:
            "completed" if simulated annealing was completed successfully;
            "aborted" if it was aborted by the user.
        """
        #if config.multiscale > -1:
        #    return solve_sa_multiscale(series0, series1, config.sa_intervals, known_line, config)
        
        random_generator = random.Random(config.random_seed)
        n_record_types = len(self.series_preprocessed[0])
        
        starting_warp = Bwarp(Bseries(self.series_preprocessed[0], config.sa_intervals),
                              Bseries(self.series_preprocessed[1], config.sa_intervals),
                              max_rate = config.max_rate,
                              rc_penalty = config.rc_penalty,
                              rnd = random_generator)
        
        starting_warp.max_rate = config.max_rate
        
        # Set up warp plotter if needed
        plotter = None
        if config.live_display:
            plotter = WarpPlotter(config.sa_intervals, known_line, 100,
                                  pdf_file = 'dsaframes-1.pdf' if config.make_pdf else None)
        
        ltemp_max = math.log(config.temp_init)
        ltemp_min = math.log(config.temp_final)
        debug_file = None
        if config.debug:
            debug_file = open(os.path.expanduser("~/scoter-sa.txt"), "w")
        
        def callback(soln_current, soln_new, schedule):
            if plotter != None:
                plotter.replot(soln_current, soln_new, schedule.step)
            if debug_file != None:
                debug_file.write(str(soln_current.score()) + "\n")
            if callback_obj != None:
                pc = (ltemp_max - math.log(schedule.temp)) / (ltemp_max - ltemp_min) * 100
                callback_obj.simann_callback_update(soln_current, soln_new, pc)
                return callback_obj.simann_check_abort()
        
        # Create and run the simulated annealer.

        schedule = AdaptiveSchedule(config.temp_init, config.temp_final,
                                    config.max_changes, config.max_steps, rate = config.cooling)

        finished_ok = True
        if config.precalc:
            bwarp_annealed = starting_warp
        else:
            annealer = Annealer(starting_warp, random_generator)
            finished_ok = annealer.run(schedule, restarts = 0, callback = callback)
            if not finished_ok:
                callback_obj.simann_callback_finished("aborted")
                return "aborted"
            bwarp_annealed = annealer.soln_best
        
        if debug_file:
            debug_file.close()
        logger.debug("SA final score: %.3g" % bwarp_annealed.score())
        
        # Apply the annealed antiwarp to the warped data
        if plotter: plotter.finish()
        bwarp_annealed.name = 'Sim. Ann.'
        for s in bwarp_annealed.to_strings(): logger.debug(s)
        self.aligned_sa = []
        self.aligned_sa.append(bwarp_annealed.apply(0, 0))
        if (n_record_types == 2):
            self.aligned_sa.append(bwarp_annealed.apply(0, 1))
    
        self.warp_sa = bwarp_annealed
        if callback_obj:
            callback_obj.simann_callback_finished("completed")
        return "completed"
    
    def correlate_match(self, config, remove_files = False):
        """Perform a correlation using the external match program.
        
        Args:
            config: a ScoterConfig object
        
        Returns:
            a MatchResult object representing the results
        """
        
        dir_path = tempfile.mkdtemp("", "scoter", None)
        
        match_params = dict(
        nomatch = config.match_nomatch,
        speedpenalty = config.match_speed_p,
        targetspeed = config.match_target_speed,
        speedchange = config.match_speedchange_p,
        tiepenalty = config.match_tie_p,
        gappenalty = config.match_gap_p,
        speeds = config.match_rates
        )

        match_conf =  MatchConf(MatchSeriesConf(self.series_preprocessed[0],
                                                intervals = config.match_intervals),
                                MatchSeriesConf(self.series_preprocessed[1],
                                                intervals = config.match_intervals),
                                match_params)
        match_path = self.default_match_path if config.match_path == "" else config.match_path
        logger.debug("Match path: %s", match_path)
        if match_path == None or match_path == "":
            logger.error("No match path set!")
            return None
        match_result = match_conf.run_match(match_path, dir_path, False)
        if not match_result.error:
            self.aligned_match = match_result.series1
        if remove_files:
            shutil.rmtree(dir_path, ignore_errors = True)
        else:
            self.match_dir = dir_path
        return match_result
    
    def save_results(self, directory = None):
        """Save the results of correlation to the specified directory
        """
        
        if directory == None:
            path = self.output_dir
        else:
            assert(os.path.isabs(directory))
            if not os.path.exists(directory):
                os.makedirs(directory)
            assert(os.path.isdir(directory))
            path = directory
            
        logger.debug("Saving to: %s" % path)
        
        # Copy match directory
        if self.match_dir:
            match_dest_path = os.path.join(path, "match")
            # Remove any existing match directory to save new results
            if os.path.exists(match_dest_path):
                logger.info("Deleting old match results folder.")
                shutil.rmtree(match_dest_path)
            shutil.copytree(self.match_dir, match_dest_path)
        
        # Save dewarped data
        if self.aligned_sa:
            for i in range(len(self.aligned_sa)):
                self.aligned_sa[i].write(os.path.join(path, "data-simann-%d" % i))
            # Save warp (i.e. sedimentation rate or depth/age ties)
            warp = self.warp_sa
            scale = (warp.series[1].series[0].end() /
            warp.series[0].series[0].end())
            rates_sa = warp.get_rates_as_series(scale = scale)
            rates_sa.write(os.path.join(path, "rate-simann"))
    
    def finalize(self):
        if self.match_dir:
            shutil.rmtree(self.match_dir)
        
        # It's important to remove the handler, in case perform_complete_correlation
        # is being run from the GUI: in this case, the GUI will keep using the 
        # logger afterwards, but there's no guarantee that the file will remain
        # accessible.
        self.file_log_handler.close()
        logger.removeHandler(self.file_log_handler)
    
    def add_file_log_handler(self):
        self.file_log_handler = logging.FileHandler(os.path.join(self.output_dir,
                                                                 "scoter.log"))
        self.file_log_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s "
                                      "%(message)s")
        self.file_log_handler.setFormatter(formatter)
        logger.addHandler(self.file_log_handler)
    
    def perform_complete_correlation(self, config_file):
        """Reads, correlates, and saves data.
        
        This is the master function for non-interactive operation. All parameters
        are read from the supplied configuration file.
        """

        config = ScoterConfig.read_from_file(config_file)
        
        # We need to locate (and possibly create) the output directory before doing
        # anything else, because we need somewhere for the log file to go.
        output_dir = config.output_dir
        logger.debug("Output dir: ‘%s’" % output_dir)
        if output_dir == "":
            # If no output directory is explicitly set, use the parent directory
            # of the configuration file.
            output_dir = os.path.dirname(config_file)
        if not os.path.isabs(output_dir):
            # If the specified output directory is a relative path, resolve it
            # relative to the parent directory of the configuration file.
            output_dir = os.path.join(os.path.dirname(config_file), output_dir)
        if not os.path.isdir(output_dir):
            # Create the output directory if it does not already exist.
            os.makedirs(output_dir)
        self.output_dir = output_dir
        
        self.add_file_log_handler()
        logger.setLevel(logging.DEBUG)
        logger.debug("Scoter version %s starting." % SCOTER_VERSION)
        
        if config.scoter_version != SCOTER_VERSION:
            logger.warn("Configuration has version %s, but Scoter has version %s" %
                        (config.scoter_version, SCOTER_VERSION))
        
        logger.debug("Reading data.")
        self.read_data_using_config(config, os.path.dirname(config_file))
        self.preprocess(config)
        if config.sa_active:
            logger.debug("Starting SA correlation.")
            self.correlate_sa(None, config, None)
            logger.debug("Finished SA correlation.")
        if config.match_active:
            logger.debug("Starting Match correlation.")
            self.correlate_match(config)
            logger.debug("Finished Match correlation.")
        self.save_results()
        logger.debug("Correlation(s) complete.")
        self.finalize()

def main():
    # TODO configure logging to stdout
    parser = argparse.ArgumentParser(description="Correlate geological records.")
    parser.add_argument("configuration", metavar="filename", type=str, nargs="?",
                   help="a Scoter configuration file")
    parser.add_argument("--version", action="store_true",
                   help="display Scoter's version number")    
    parser.add_argument("--write-config", metavar="filename", type=str,
                   help="write a configuration template to the supplied filename")
    parser.add_argument("--overwrite", action="store_true",
                   help="when writing a configuration template, overwrite any existing file")
    parser.add_argument("--log-level", metavar="level", type=str,
                   help="logging level (non-negative integer or CRITICAL/ERROR/WARNING/INFO/DEBUG/NOTSET)",
                   default="INFO")
    args = parser.parse_args()
        
    if args.version:
        print(SCOTER_VERSION)
        sys.exit(0)
    
    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(logging.Formatter("%(levelname)-8s: %(message)s",
                                                  None))
    stderr_handler.setLevel(args.log_level)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(stderr_handler)
    
    logger.info("Scoter starting.")
        
    if args.write_config:
        ok_to_write = False
        if os.path.isfile(args.write_config):
            logger.info("File '%s' exists." % args.write_config)
            if args.overwrite:
                logger.info("Overwriting as requested.")
                ok_to_write = True
            else:
                logger.info("Use --overwrite option to overwrite existing file")
        elif os.path.isdir(args.write_config):
            logger.error("'%s' is a directory; not writing configuration." % args.write_config)
        else:
            ok_to_write = True
            
        if ok_to_write:
            config = ScoterConfig()
            config.write_to_file(args.write_config)
            
    else:   # not args.write_config
        if args.configuration:
            scoter = Scoter()
            scoter.perform_complete_correlation(args.configuration)
        else:
            logger.warning("No configuration file specified.")
    
    logger.info("Scoter run finished.")

logger = logging.getLogger(__name__)
if __name__ == "__main__":
    main()

