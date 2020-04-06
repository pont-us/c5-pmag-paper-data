#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""Function and classes for reading and handling PSV and RPI data.

The Location class represents a position (latitude and longitude) on
the Earth's surface.

The Dataset class holds a collection of PSV and/or RPI data for some
timespan and location.

The main function provided by curveread is read_dataset, which
fetches a PSV and/or RPI dataset specified by name and returns it as a Dataset
object, optionally relocating any PSV data to a provided Location.
"""

import hashlib
import inspect
import logging
import numpy as np
import os
import pexpect
import re
import scipy.stats.mstats
import shutil
import subprocess
import tarfile
import urllib
import zipfile
from math import degrees, radians, sin, cos, tan, asin, atan, atan2, pi, sqrt

from scoter.series import Series

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(os.path.normpath(BASE_DIR))
PRESENT_YEAR = 1950  # standardize on BP ages
logger = logging.getLogger(__name__)


def dtan(angle):
    return tan(radians(angle))


def datan(x):
    return degrees(atan(x))


def md5_on_file(filename):
    md5 = hashlib.md5()
    with open(filename, "rb") as fh:
        for chunk in iter(lambda: fh.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


class Location(object):
    
    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude

    def lat_rads(self):
        return radians(self.latitude)

    def long_rads(self):
        return radians(self.longitude)

    @classmethod
    def from_radians(cls, latitude, longitude):
        return cls(degrees(latitude), degrees(longitude))

    @staticmethod
    def _dms_to_deg(dms):
        assert(len(dms) < 4)
        deg = dms[0]
        if len(dms) > 1:
            deg += dms[1] / 60.
        if len(dms) == 3:
            deg += dms[2] / 3600.
        return deg

    @classmethod
    def from_dms(cls, lat_dms, long_dms):
        return cls(cls._dms_to_deg(lat_dms),
                   cls._dms_to_deg(long_dms))

    def calculate_vgp_rad(self, dec, inc):
        """dec and inc in radians"""
        inc_m = inc
        dec_m = dec
        lambda_s = self.lat_rads()
        phi_s = self.long_rads()
        p = atan2(2, tan(inc_m))
        lambda_p = asin(sin(lambda_s)*cos(p) +
                        cos(lambda_s)*sin(p)*cos(dec_m))
        beta = asin((sin(p)*sin(dec_m)) / (cos(lambda_p)))
        phi_p = (((phi_s + beta) if
                  (cos(p) >= sin(lambda_s) * sin(lambda_p)) else
                  (phi_s + pi - beta))
                 + (2 * pi)) % (2 * pi)
        return Location.from_radians(lambda_p, phi_p)

    def calculate_vgp_deg(self, dec, inc):
        return self.calculate_vgp_rad(radians(dec), radians(inc))

    def relocate_by_cvp(self, dest_loc, decs, incs):
        """Transform directions measured at this location to hypothetical
        directions which would have been measured at dest_loc, using the
        "conversion via pole" method (see doi:
        10.1111/j.1365-246X.1990.tb04594.x )
    
        Returns a tuple consisting of a sequence of transformed
        declinations and a sequence of transformed inclinations.
    
        Declinations and inclinations are in degrees.

        """
        
        assert(len(decs) == len(incs))
    
        new_decs = np.empty_like(decs)
        new_incs = np.empty_like(incs)
    
        lambda_r = dest_loc.lat_rads()
        phi_r = dest_loc.long_rads()
    
        for i in range(len(decs)):
            dec = decs[i]
            inc = incs[i]
            vgp = self.calculate_vgp_deg(dec, inc)
            lambda_p = vgp.lat_rads()
            phi_p = vgp.long_rads()
            beta = phi_r - phi_p + pi
    
            c = atan(((sin(lambda_p) * sin(lambda_r)
                      + cos(lambda_p)*cos(lambda_r)*cos(phi_p-phi_r))**-2.
                      - 1.)**0.5)
            inc_rad = atan(2. / tan(c))
            dec_rad = asin(sin(beta) * cos(lambda_p) / sin(c))
    
            new_decs[i] = degrees(dec_rad)
            new_incs[i] = degrees(inc_rad)
    
        return new_decs, new_incs

    def distance_km(self, other):
        phi1 = self.lat_rads()
        phi2 = other.lat_rads()
        dlambda = abs(self.long_rads() - other.long_rads())
        part1 = (cos(phi2) * sin(dlambda))**2
        part2 = (cos(phi1)*sin(phi2) - sin(phi1)*cos(phi2)*cos(dlambda))**2
        part3 = sin(phi1)*sin(phi2) + cos(phi1)*cos(phi2)*cos(dlambda)
        dsigma = atan2(sqrt(part1+part2), part3)
        earth_radius_km = 6371
        return earth_radius_km * dsigma


class Dataset(object):
    
    def __init__(self, name, xs, decs=None, incs=None, rpis=None,
                 named_data=None, data_to_normalize=None):
        self.data_to_normalize = [] \
            if data_to_normalize is None else data_to_normalize
        self.extras = {} \
            if named_data is None else named_data
        self.name = name
        self.xs = xs
        self.decs = decs
        self.incs = incs
        self.rpis_raw = rpis
        self.rpis = None

    def normalize(self):
        for key in self.extras:
            values = np.array(self.extras[key])
            if key in self.data_to_normalize:
                self.extras[key] = Dataset.zscores_ignoring_outliers(values, 2)
            else:
                self.extras[key] = values

        self.rpis = None if self.rpis_raw is None else \
            Dataset.zscores_ignoring_outliers(self.rpis_raw, 2)

    def truncate(self, max_x):
        max_index = np.argmax(self.xs > max_x)
        if max_index == 0:
            return
        for attribute in "xs", "decs", "incs", "rpis_raw", "rpis":
            value = getattr(self, attribute)
            if value is not None:
                setattr(self, attribute, value[:max_index])
        for key in self.extras:
            self.extras[key] = self.extras[key][:max_index]

    @staticmethod
    def zscores(values):
        values_copy = values.copy()
        values_copy[~np.isnan(values)] = \
            scipy.stats.mstats.zscore(values[~np.isnan(values)])
        return values_copy

    @staticmethod
    def zscores_ignoring_outliers(data, n_stdevs=2):
        data_without_outliers = \
            data[abs(data - np.mean(data)) < n_stdevs * np.std(data)]
        mean = np.mean(data_without_outliers)
        stdev = np.std(data_without_outliers)
        def zscorify(x): return (x - mean) / stdev
        data_copy = data.copy()
        data_copy[~np.isnan(data)] = \
            zscorify(data[~np.isnan(data)])
        return data_copy

    def flip(self):
        self.xs = self.xs[::-1]
        if self.decs is not None:
            self.decs = self.decs[::-1]
        if self.incs is not None:
            self.incs = self.incs[::-1]
        if self.rpis_raw is not None:
            self.rpis_raw = self.rpis_raw[::-1]
        if self.rpis is not None:
            self.rpis = self.rpis[::-1]
        for key in self.extras:
            self.extras[key] = self.extras[key][::-1]

    def inc_series(self, name):
        return Series(np.array([self.xs, self.incs]), name=name)

    def dec_series(self, name):
        return Series(np.array([self.xs, self.decs]), name=name)

    def rpi_series(self, name):
        return Series(np.array([self.xs, self.rpis]), name=name)

    def get_series(self, key, name):
        return Series(np.array([self.xs, self.extras[key]]), name=name)

    def has_named_data(self, key):
        return key in self.extras

    def get_named_data(self, key):
        return self.extras[key]

    def calculate_mad3_limits(self):
        if "mad3" in self.extras:
            if self.incs is not None:
                self.extras["inc-min"] = self.incs - self.extras["mad3"]
                self.extras["inc-max"] = self.incs + self.extras["mad3"]
            if self.decs is not None:
                self.extras["dec-min"] = self.decs - self.extras["mad3"]
                self.extras["dec-max"] = self.decs + self.extras["mad3"]


def ensure_dir_exists(dirname):
    try: 
        os.makedirs(dirname)
    except OSError:
        if not os.path.isdir(dirname):
            raise


def abs_path(path):
    return os.path.join(BASE_DIR, path)


def refdata_path(*components):
    return os.path.join(PARENT_DIR, "ref-data", *components)


def data_path(*components):
    return os.path.join(PARENT_DIR, "data", *components)


def relocate_inclination(src_lat, dest_lat, incs):
    """Transform inclinations measured at latitude src_lat
    to hypothetical inclinations which would have been measured
    at latitude dest_lat assuming a perfect GAD field.

    Returns a sequence of transformed inclinations.

    All latitudes and inclinations in degrees.
    """

    offset = (datan(2 * dtan(dest_lat)) -
              datan(2 * dtan(src_lat)))

    return incs + offset


def read_cals(model_name, location):

    url_prefix = ("http://www.gfz-potsdam.de/fileadmin/gfz/"
                  "sec23/data/Models/CALSxK/")

    model_table = {  # ZIP file, Fortran file, header lines, md5, proper name
        "cals10k1b":  ("CALS10k_1b", "CALS10kfield", 2,
                       "e10c79c9243b90c551d988468043a57b",
                       "CALS10k.1b"),
        "cals3k4":    ("cals3k-4",   "fieldpred3k",  2,
                       "ad515e8a3fbc9c5a9615ae214b72b2df",
                       "CALS3k.4"),
        "cals10k2":   ("CALS10k2",   "CALS10kfield", 1,
                       "91aed1c4fd765a7bda3bfadbb622c7ab",
                       "CALS10k.2"),
        "cals7k2":    ("cals7k2",    "CALS7Kseries", 4,
                       "63e09f9c8e0f85cfdae563ca0198960b",
                       "CALS7k.2"),
    }

    zip_name, prog_name, header_lines, md5_expected, proper_name = \
        model_table[model_name]

    lat_str = str(location.latitude)
    long_str = str(location.longitude)
    
    url = url_prefix + zip_name + ".zip"  # URL of zip file to download
    # model_dir: parent directory for model code and output data
    model_dir = refdata_path(model_name)
    # zip_path: local path of downloaded ZIP file
    zip_path = os.path.join(model_dir, zip_name + ".zip")
    results_file = ("output-%s-%s" %
                    (lat_str.replace(".", "_"),
                     long_str.replace(".", "_")))

    # Generate the output file, if required.
    if not os.path.isfile(os.path.join(model_dir, results_file)):
        # No existing results file -- generate it from scratch.
        # Actually some make-type process (perhaps using scons) 
        # would be more appropriate here, but it's not worth 
        # implementing just for this script.

        # Check if we already have the zip of the software.
        if not os.path.isfile(zip_path):
            # If not, download it.
            ensure_dir_exists(model_dir)
            response = urllib.urlopen(url)
            with open(zip_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

        # Checksum the zip file to make sure it's what we expected
        # -- e.g. we might have got a 404 document instead, or the
        # file on the server may have been changed.
        md5_calculated = hashlib.md5(open(zip_path, "rb").read()).hexdigest()
        if md5_expected != md5_calculated:
            raise RuntimeError(
                "Incorrect checksum for %s : expected %s, got %s" %
                (zip_path, md5_expected, md5_calculated))

        # Unpack the zip file.
        logger.debug("Unzipping " + zip_path)
        with zipfile.ZipFile(zip_path) as zip_file:
            zip_file.extractall(model_dir)

        # Compile the program.
        p = subprocess.Popen(["gfortran", "-o", prog_name, prog_name+".f"],
                             cwd=model_dir)
        p.wait()

        # Run the program, using pexpect to supply the parameters
        # since it only accepts interactive input.
        child = pexpect.spawnu(os.path.join(model_dir, prog_name),
                               cwd=model_dir,
                               timeout=2)

        # CALS silently truncates the output filename to 15 characters,
        # so we give a short generic name here, then rename it
        # afterwards using the "proper" name with a lat/long suffix.

        temp_results_file = "output"
        prompts_responses = \
            [(u"to store the results:\r\n", temp_results_file),  # CALS7K.2
             (u"latitude in decimal degrees:\r\n", lat_str),  # CALS7K.2
             (u"longitude in decimal degrees:\r\n", long_str),  # CALS7K.2
             (u"file name:\r\n", temp_results_file),  # 3k.4?
             (u"(S negative):\r\n", lat_str),  # 3k.4?
             (u"(W negative):\r\n", long_str),  # 3k.4?
             (u"negative for South:\r\n", lat_str),  # 10k.2
             (u">180 for W):\r\n", long_str),  # 10k.2
             (u"CALS3k.4b\r\n", "3")  # model selection (3k.4)
             ]

        prompts = [x[0] for x in prompts_responses]
        responses = [x[1] for x in prompts_responses]

        while True:
            try:
                index = child.expect_exact(prompts)
                logger.debug("Prompt: " + prompts[index])
                logger.debug("Response: " + responses[index])
                child.sendline(responses[index])
            except pexpect.EOF:
                logger.debug("Received EOF")
                break
            
        child.close()

        # Rename the results file to something that includes the
        # longitude and latitude. This lets us cache results for
        # different locations.
        os.rename(os.path.join(model_dir, temp_results_file),
                  os.path.join(model_dir, results_file))

    # Read the output file.
    # Fields (whitespace separated):
    # Calendar-Year D(deg) I(deg) F(microT) deltaD deltaI deltaF
    # Skip first two lines.
    ages, decs, incs, pints = [], [], [], []
    with open(os.path.join(model_dir, results_file), "r") as infile:
        lines = infile.readlines()
        for line in lines[header_lines:]:
            parts = line.strip().split()
            cal_yr = int(parts[0])
            dec, inc, pint = map(float, parts[1:4])
            ages.append(PRESENT_YEAR - cal_yr)
            decs.append(dec)
            incs.append(inc)
            pints.append(pint)
    
    d = Dataset(proper_name, np.array(ages), np.array(decs),
                np.array(incs), np.array(pints))
    d.flip()
    return d


def read_augusta(curve_name, location):
    # Lat: 37°12.69′N; Long: 15°15.19′E
    # Mean inclination: 47.88
    # Mean inc. for CALS10.2: 56.11 (for 152-4174 yr bp)
    # Estimated shallowing: 8.23 degrees
    # Ages are in years BP.

    augusta_location = Location.from_dms((37, 12.69), (15, 15.19))
    augusta_file = refdata_path(
        "augusta-bay",
        "MS06-MS06SW all_published x Pont.csv")
    data_raw = np.genfromtxt(
        augusta_file, delimiter=",",
        skip_header=1)
    relevant_data = data_raw[:, [3, 5, 7, 12]]  # age, dec, inc, int

    # Discard any row that contains a NaN. In addition to completely
    # data-free rows, this removes the nine rows which contain direction
    # data but no RPI data. Doing a tandem fit with varying data
    # availability between parameters is tricky, so those nine
    # inclination values would be impractical to use anyway.

    valid_rows = ~np.isnan(relevant_data[:]).any(axis=1)
    valid_data = relevant_data[valid_rows]
    data = valid_data.swapaxes(0, 1)
    data[2] += 8.23
    if location is not None:
        decs, incs = augusta_location.relocate_by_cvp(
            location, data[1], data[2])
        data[1] = decs
        data[2] = incs
    d = Dataset("Augusta", data[0], data[1], data[2], data[3])
    return d


def read_shadif14k(model_name, location):
    model_dir = refdata_path(model_name)
    lat_str = "%.6f" % location.latitude
    long_str = "%.6f" % location.longitude
    location_suffix = (lat_str + "_" + long_str).replace(".", "p")
    results_filename = "output_" + location_suffix + ".dat"
    results_path = os.path.join(model_dir, results_filename)

    if not os.path.isfile(results_path):
        
        eval_expr = "shadif14k_prediction({}, {}, 1)".format(lat_str, long_str)
        p = subprocess.Popen(["octave", "--eval", eval_expr],
                             cwd=model_dir)
        p.wait()
        # The matlab code writes to the hard-coded filename
        # 'psvc_sha.dif.14k.dat', which we rename after running
        # the model.
        os.rename(os.path.join(model_dir, "psvc_sha.dif.14k.dat"),
                  results_path)

    shadif_data = np.genfromtxt(results_path).swapaxes(0, 1)
    shadif_ybp = PRESENT_YEAR - shadif_data[0]  # column 0 is calendar_year
    # Columns: 0 year, 1 latitude, 2 longitude, 3 dec, 4 dec error, 5 inc,
    # 6 inc error, 7 intensity, 8 intensity error
    d = Dataset("SHA.DIF.14k", shadif_ybp, shadif_data[3],
                shadif_data[5], shadif_data[7])
    d.flip()
    return d


def read_uk(name, location):

    # Location taken from:
    # ftp://ftp.ngdc.noaa.gov/geomag/Paleomag/access/ver3.5/secvrasc/REGION.txt
    uk_location = Location(54.5, -3.5)

    # Ensure the data file exists, and download it if not.
    url = ("ftp://ftp.ngdc.noaa.gov/geomag/Paleomag/"
           "access/ver3.5/secvrasc/REGIONAL%20LAKES.txt")
    md5_expected = "714f1d5b8cabc6bbff208078ba791575"
    data_dir = refdata_path(name)
    ensure_dir_exists(data_dir)
    leafname = "REGIONAL LAKES.txt"
    pathname = os.path.join(data_dir, leafname)
    if not os.path.isfile(pathname):
        response = urllib.urlopen(url)
        with open(pathname, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        md5_calculated = hashlib.md5(open(pathname, "rb").read()).hexdigest()
        if md5_expected != md5_calculated:
            raise RuntimeError("Incorrect checksum for "+leafname)

    # Read data from the file. 

    # Table definition, taken from Access database.

    # # Key Name    Type     Description
    #
    # 1 Y REGION    Text     Region name linked to REGION table
    # 2 Y RESNO     Long Int Unique number for each region [1]
    # 3 N C14AGES   Long Int C14 age for this mean
    # 4 N CALAGES   Long Int Calibrated C14 age for this mean
    # 5 N DEC       Single   Declination
    # 6 N INC       Single   Inclination (range: -90 to 90)
    # 7 N INTENSITY Single   Intensity (optional)
    #
    # [1] This is poorly phrased in the original description.
    # The number is unique *within* each region, which is why
    # it is combined with the region to form the primary key.
    
    # There is no intensity data for the UK curve, so we
    # ignore column 7.
    data_raw = np.genfromtxt(
        pathname,
        delimiter=";",
        dtype="a32, int32, int32, int32, float64, float64",
        usecols=np.arange(6))
    selector = np.vectorize(lambda(x): x[0] == '"Great Britain 1"')
    valid_rows = selector(data_raw)
    data = data_raw[valid_rows]
    decs, incs = data["f4"], data["f5"]

    # Relocate data if required
    if location is not None:
        decs, incs = uk_location.relocate_by_cvp(location, decs, incs)

    return Dataset("UK Master", data["f3"], decs, incs, None)


def read_bulg1998():
    # digitized from Kovacheva et al. (1998).
    # doi: 10.1023/A:1006502313519
    # N.B. Low-resolution and possibly inaccurate digitization.
    # For most purposes, the Kovacheva et al. (2014) dataset should in
    # any case supercede it.
    filename = refdata_path("bulgaria1998", "incs.csv")
    data = np.genfromtxt(filename, delimiter=",",
                         skip_header=1).swapaxes(0, 1)
    return Dataset("Bulgaria", data[0], None, data[1], None)


def read_bulg2014():
    # Digitized from Kovacheva et al. (2014)
    # doi:10.1016/j.pepi.2014.07.002
    # NB: it's not clear what location these data have been reduced to.
    filename = refdata_path("bulgaria2014", "intensity.csv")
    data = np.genfromtxt(filename, delimiter=",",
                         skip_header=1).swapaxes(0, 1)
    return Dataset("Bulgaria", PRESENT_YEAR - data[0][::-1], None,
                   None, data[1][::-1])


def read_salerno_incs():
    # Digitized from Iorio et al. (2014) fig. 9, core C1201
    # doi:10.1016/j.gloplacha.2013.11.005

    # Inclination, declination, and RPI records don't share the same age
    # co-ordinates, which is why we have to put them in different Dataset
    # objects.

    filename_incs = refdata_path("salerno", "inc.tsv")
    data_incs = np.genfromtxt(filename_incs, delimiter="\t",
                              skip_header=1).swapaxes(0, 1)
    return Dataset("Salerno", data_incs[0]*1000, None, data_incs[1], None)


def read_salerno_decs():
    # Digitized from Iorio et al. (2014) fig. 9, core C1201
    # doi:10.1016/j.gloplacha.2013.11.005
    # Inclination, declination, and RPI records don't share the same age
    # co-ordinates, which is why we have to put them in different Dataset
    # objects.

    filename_decs = refdata_path("salerno", "dec.tsv")
    data_decs = np.genfromtxt(filename_decs, delimiter="\t",
                              skip_header=1).swapaxes(0, 1)
    return Dataset("Salerno", data_decs[0]*1000, data_decs[1], None, None)


def read_salerno_rpis():
    # Digitized from Iorio et al. (2014) fig. 10, core C1201
    # doi:10.1016/j.gloplacha.2013.11.005
    # Inclination, declination, and RPI records don't share the same age
    # co-ordinates, which is why we have to put them in different Dataset
    # objects.

    filename_rpis = refdata_path("salerno", "rpi.tsv")
    data_rpis = np.genfromtxt(filename_rpis, delimiter="\t",
                              skip_header=1).swapaxes(0, 1)
    return Dataset("Salerno", data_rpis[0]*1000, None, None, data_rpis[1])


def read_piso():
    # Channell et al. (2009)
    # doi:10.1016/j.epsl.2009.03.012
    filename = refdata_path("piso", "mmc2.txt")
    data = np.genfromtxt(filename, delimiter="\t",
                         skip_header=1).swapaxes(0, 1)
    return Dataset("PISO-1500", data[0]*1000, None, None, data[1])


def read_u1308():
    filename = refdata_path("u1308", "313-U1308_rpi.tab")
    data = np.genfromtxt(filename, delimiter="\t",
                         skip_header=16).swapaxes(0, 1)
    return Dataset("U1308", data[0][0:200]*1000, None, None, data[1][0:200]*10)


def read_igrf12(model_name, location):
    # https://www.ngdc.noaa.gov/IAGA/vmod/geomag70_linux.tar.gz
    url_prefix = "https://www.ngdc.noaa.gov/IAGA/vmod/"
    archive_name = "geomag70_linux.tar.gz"
    binary_name = "geomag70.exe"
    model_dir = refdata_path(model_name)
    ensure_dir_exists(model_dir)
    archive_path = os.path.join(model_dir, archive_name)
    if not os.path.isfile(archive_path):
        response = urllib.urlopen(url_prefix + archive_name)
        with open(archive_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
    if not os.path.isfile(os.path.join(model_dir, binary_name)):
        archive = tarfile.open(archive_path)
        archive.extractall(model_dir)

    # The archive contains a precompiled binary, so we just run that
    # rather than recompiling the C source. This assumes that we're running on
    # Linux, of course.

    lat_str = str(location.latitude)
    long_str = str(location.longitude)
    results_filename = os.path.join(
        model_dir,
        "output_%sn_%se" % (lat_str.replace(".", "p"),
                            long_str.replace(".", "p")))
    model_subdir = os.path.join(model_dir, "geomag70_linux")

    with open(results_filename, "w") as fh:
        p = subprocess.Popen(
            [os.path.join(model_subdir, binary_name),
             "IGRF12.COF", "1900.00-1990.00-1.00", "D",
             "K0.00", lat_str, long_str],
            stdout=fh, cwd=model_subdir)
        p.wait()

    with open(results_filename, "r") as fh:
        lines = fh.readlines()

    def extract_data(dataline):
        fields = re.split(" +", re.sub("[dm]", "", dataline).strip())
        age = PRESENT_YEAR - int(float(fields[0]))
        dec = float(fields[1]) + float(fields[2]) / 60
        inc = float(fields[3]) + float(fields[4]) / 60
        f = float(fields[9])
        return age, dec, inc, f

    data_rows = np.array([extract_data(line) for line in lines[15:105]])
    data = np.fliplr(data_rows.swapaxes(0, 1))
    return Dataset(model_name, data[0], data[1], data[2], data[3])


def read_pfm9k_source(curve_name, location):
    """Read a dataset from the pfm9k data sources file.

    Note: this function doesn't run the pfm9k model itself! It reads a
    curve from the data archives on which pfm9k is based.

    Reference: Nilsson et al. (2014), doi:10.1093/gji/ggu120 .

    :param curve_name: name of curve (three-letter code as used in
    data file)
    :param location: location to which to relocate the curve
    :return: the requested curve
    """

    data_root = refdata_path("pfm9k_source")
    archive_path = os.path.join(data_root, "pfm9k.zip")
    pfm9k1_data_path = os.path.join(data_root, "pfm9k1")
    ensure_dir_exists(data_root)
    if not os.path.isdir(pfm9k1_data_path):
        if not os.path.isfile(archive_path):
            response = urllib.urlopen(
                "http://earthref.org/ERDA/download:1951/")
            with open(archive_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        with zipfile.ZipFile(archive_path) as zip_file:
            files_to_extract = \
                filter(lambda s: s.startswith("pfm9k1/"), zip_file.namelist())
            zip_file.extractall(path=data_root, members=files_to_extract)

    with open(os.path.join(pfm9k1_data_path, "pfm9k.1_DAT"), "r") as fh:
        lines = fh.readlines()

    ages, decs, incs = [], [], []
    source_location = None
        
    for line in lines[1:]:
        parts = line.split("\t")
        record_code = parts[2]
        if record_code == curve_name.upper():
            if location is not None:
                if source_location is None:
                    latitude = float(parts[3])
                    longitude = float(parts[4])
                    source_location = Location(latitude, longitude)
            year = int(parts[0])
            declination = float(parts[5])
            inclination = float(parts[7])
            if inclination != -999 and declination != -999:
                ages.append(PRESENT_YEAR - year)
                decs.append(declination)
                incs.append(inclination)

    if location is not None:
        decs, incs = source_location.relocate_by_cvp(location, decs, incs)

    dataset_names = dict(ty1="ET91-18", ty2="ET95-4")
    dataset_name =\
        dataset_names[curve_name] if curve_name in dataset_names else curve_name
    
    d = Dataset(dataset_name, ages, decs, incs)
    d.flip()
    return d


def read_geomagia_italy_psv(name, location):
    data_root = os.path.join(refdata_path("geomagia-italy"))
    with open(os.path.join(data_root, "archeo013.csv"), "r") as fh:
        lines = fh.readlines()

    ages, decs, incs = [], [], []
    for line in lines[2:]:
        parts = line.split(",")
        year = int(parts[0])
        dec_at_site = float(parts[14])
        inc_at_site = float(parts[15])
        if dec_at_site == -999 or inc_at_site == -999:
            continue
        latitude = float(parts[23])
        longitude = float(parts[24])
        site_loc = Location(latitude, longitude)
        decs_reloc, incs_reloc = \
            site_loc.relocate_by_cvp(location, [dec_at_site], [inc_at_site])
        ages.append(PRESENT_YEAR - year)
        decs.append(decs_reloc)
        incs.append(incs_reloc)

    d = Dataset("Italy", ages, decs, incs)
    d.flip()
    return d


def read_geomagia_italy_rpi(name, location):
    data_root = os.path.join(refdata_path("geomagia-italy"))
    with open(os.path.join(data_root, "archeo013.csv"), "r") as fh:
        lines = fh.readlines()

    ages, rpis = [], []
    for line in lines[2:]:
        parts = line.split(",")
        year = int(parts[0])
        rpi = float(parts[7])
        if rpi != -999:
            ages.append(PRESENT_YEAR - year)
            rpis.append(rpi)

    d = Dataset("Italy", ages, None, None, np.array(rpis))
    d.flip()
    return d


def read_rio_martino(name, location):
    rm_location = Location(44 + 42 / 60, 7 + 9 / 60)
    data = np.genfromtxt(refdata_path("rio-martino", name + ".csv"),
                         delimiter=",",
                         skip_header=1)
    age_kyr, decs_rm, incs_rm = data[:, 1], data[:, 2], data[:, 3]
    decs, incs = rm_location.relocate_by_cvp(location, decs_rm, incs_rm)
    return Dataset(name.upper(), age_kyr * 1000, decs, incs, None)


def read_taranto_mp49_psv(name, location):
    mp49_location = Location(39 + 50.07 / 60, 17 + 48.06 / 60)
    data = np.genfromtxt(refdata_path("taranto-mp49", "psv.csv"),
                         delimiter=",",
                         skip_header=0)
    year, site_decs, site_incs = data[:, 0], data[:, 1], data[:, 2]
    decs, incs = mp49_location.relocate_by_cvp(location, site_decs, site_incs)
    d = Dataset("MP49", PRESENT_YEAR - year, decs, incs, None)
    d.flip()
    return d


def read_taranto_mp49_rpi(name, location):
    data = np.genfromtxt(refdata_path("taranto-mp49", "vadm.csv"),
                         delimiter=",",
                         skip_header=0)
    year, vadm = data[:, 0], data[:, 1]
    d = Dataset("MP49", PRESENT_YEAR - year, None, None, vadm)
    d.flip()
    return d


def read_w_europe(name, location):

    # Location (Paris) specified in the papers
    paris = Location(48.9, 2.3)

    data_dir = refdata_path(name)

    data1 = np.genfromtxt(os.path.join(data_dir, "950BCE-50BCE.txt"),
                          delimiter=" ",
                          skip_header=3)
    years1, decs1, incs1 = -data1[:, 0], data1[:, 5], data1[:, 4]

    data2 = np.genfromtxt(os.path.join(data_dir, "100BCE-1800CE.txt"),
                          delimiter=" ",
                          skip_header=3)
    data2_skip = 3  # skip rows which overlap with data1
    years2, decs2, incs2 =\
        data2[data2_skip:, 0], data2[data2_skip:, 5], data2[data2_skip:, 4]

    years, decs, incs =\
        np.append(years1, years2),\
        np.append(decs1, decs2),\
        np.append(incs1, incs2)

    # Relocate data if required
    if location is not None:
        decs, incs = paris.relocate_by_cvp(location, decs, incs)

    return Dataset("W. Europe", PRESENT_YEAR - years, decs, incs, None)


def read_c5():
    # Read declination, inclination, and MAD3s
    psv_file = data_path("processed", "psv-from-script.csv")
    psv_data = np.genfromtxt(psv_file,
                             delimiter=",",
                             skip_header=1).swapaxes(0, 1)
    logger.info("C5 mean inc. %f" % np.mean(psv_data[6][20:]))
    depths_inc = set(psv_data[1])

    # Read ARM-normalized RPIs
    rpi_arm_file = data_path("processed", "rpi-arm-optimized.csv")
    rpi_arm_data = np.genfromtxt(rpi_arm_file,
                                 delimiter=",",
                                 skip_header=1).swapaxes(0, 1)
    depths_rpi_arm = set(rpi_arm_data[0])

    # Read IRM-normalized RPIs
    rpi_irm_file = data_path("processed", "rpi-irm-optimized.csv")
    rpi_irm_data = np.genfromtxt(rpi_irm_file,
                                 delimiter=",",
                                 skip_header=1).swapaxes(0, 1)
    depths_rpi_irm = set(rpi_irm_data[0])

    # Read MS-normalized RPIs
    rpi_ms_file = data_path("processed", "rpi-ms.csv")
    rpi_ms_data = np.genfromtxt(rpi_ms_file,
                                delimiter=",",
                                skip_header=1).swapaxes(0, 1)
    depths_rpi_ms = set(rpi_ms_data[0])

    depths = depths_inc.intersection(depths_rpi_arm, depths_rpi_ms,
                                     depths_rpi_irm)

    # 1 depth 6 inc 8 mad3
    decs, incs, mad3s = [], [], []
    for i in range(len(depths_inc)):
        if psv_data[1][i] in depths:
            incs.append(psv_data[6][i])
            decs.append(psv_data[5][i])
            mad3s.append(psv_data[8][i])

    rpis_arm_ratio, rpis_arm_slope, rpis_arm_rsquared = [], [], []
    # 0 depth 10 mean ratio 11 slope
    for i in range(len(depths_rpi_arm)):
        if rpi_arm_data[0][i] in depths:
            rpis_arm_ratio.append(rpi_arm_data[10][i])
            rpis_arm_slope.append(rpi_arm_data[11][i])
            rpis_arm_rsquared.append(rpi_arm_data[13][i])

    rpis_irm_ratio, rpis_irm_slope, rpis_irm_rsquared = [], [], []
    # 0 depth 10 mean ratio 11 slope
    for i in range(len(depths_rpi_irm)):
        if rpi_irm_data[0][i] in depths:
            rpis_irm_ratio.append(rpi_irm_data[10][i])
            rpis_irm_slope.append(rpi_irm_data[11][i])
            rpis_irm_rsquared.append(rpi_irm_data[13][i])

    rpis_ms = []
    for i in range(len(depths_rpi_ms)):
        if rpi_ms_data[0][i] in depths:
            rpis_ms.append(rpi_ms_data[1][i])

    dataset = Dataset("C5 core", np.array(sorted(list(depths))),
                      np.array(decs), np.array(incs), np.array(rpis_ms),
                      named_data={"mad3": np.array(mad3s),
                                  "rpi-arm-ratio": rpis_arm_ratio,
                                  "rpi-arm-slope": rpis_arm_slope,
                                  "rpi-arm-rsquared": rpis_arm_rsquared,
                                  "rpi-irm-ratio": rpis_irm_ratio,
                                  "rpi-irm-slope": rpis_irm_slope,
                                  "rpi-irm-rsquared": rpis_irm_rsquared,
                                  "rpi-ms": rpis_ms
                                  },
                      data_to_normalize=["rpi-arm-ratio", "rpi-arm-slope",
                                         "rpi-irm-ratio", "rpi-irm-slope",
                                         "rpi-ms"])

    # NB median RPI created seperately
    dataset.calculate_mad3_limits()
    return dataset


def create_c5_median_rpi(dataset):

    # We have to add the median RPI data after normalizing the RPI
    # values, because we want the mean of the *normalized* values, and
    # normalization is done in the constructor.

    rpis_median = []
    for i in range(len(dataset.xs)):
        all_estimates = \
            [dataset.get_named_data(estimate)[i] for estimate in
             ("rpi-arm-ratio",
              "rpi-arm-slope",
              "rpi-irm-ratio",
              "rpi-irm-slope",
              "rpi-ms")]
        median = np.median(all_estimates)
        rpis_median.append(median)
    dataset.extras["rpi-median"] = np.array(rpis_median)
    dataset.rpis = dataset.extras["rpi-median"]


def read_c5_rpi_ms():
    """Read the RPI record of the C5 core as calculated by normalization
    to magnetic susceptibility.
    """
    rpi_file = data_path("processed", "rpi-ms.csv")
    rpi_data = np.genfromtxt(rpi_file,
                             delimiter=",",
                             skip_header=0).swapaxes(0, 1)
    return Dataset("C5 core RPI MS", rpi_data[0],
                   None, None, rpi_data[2], None)


dispatch_table = {
    "cals10k1b": read_cals,
    "cals3k4": read_cals,
    "cals10k2": read_cals,
    "cals7k2": read_cals,
    "sha-dif-14k": read_shadif14k,
    "uk": read_uk,
    "bulg1998": read_bulg1998,
    "bulg2014": read_bulg2014,
    "salerno_incs": read_salerno_incs,
    "salerno_decs": read_salerno_decs,
    "salerno_rpis": read_salerno_rpis,
    "piso": read_piso,
    "u1308": read_u1308,
    "c5": read_c5,
    "c5_rpi_ms": read_c5_rpi_ms,
    "augusta": read_augusta,
    "igrf_12": read_igrf12,
    "ty1": read_pfm9k_source,
    "ty2": read_pfm9k_source,
    "w-europe": read_w_europe,
    "geomagia-italy-psv": read_geomagia_italy_psv,
    "geomagia-italy-rpi": read_geomagia_italy_rpi,
    "rmd1": read_rio_martino,
    "rmd8": read_rio_martino,
    "taranto-mp49-psv": read_taranto_mp49_psv,
    "taranto-mp49-rpi": read_taranto_mp49_rpi,
}


def read_dataset(key, location, max_x=None):
    fn = dispatch_table[key]
    nargs = len(inspect.getargspec(fn).args)
    if nargs == 0:
        dataset = fn()
    elif nargs == 2:
        dataset = fn(key, location)
    else:
        raise ValueError("Invalid number of parameters for " + fn.func_name)
    if max_x is not None:
        dataset.truncate(max_x)
    dataset.normalize()
    if key == "c5":
        create_c5_median_rpi(dataset)
    return dataset
