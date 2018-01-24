#!/usr/bin/env jython

"""
Calculate CRM directions from NRM data and RPI estimates
from NRM and ARM data. This script uses PuffinPlot's API, so it
must be run using Jython rather than the more common CPython
interpreter, and it can't be integrated with any script that use
CPython libraries such as numpy.
"""

import sys
sys.path.append("../libraries/java/PuffinPlot.jar")

from net.talvi.puffinplot.data import Suite
from net.talvi.puffinplot.data import Correction
from net.talvi.puffinplot.data import SensorLengths
from net.talvi.puffinplot.data.file import TwoGeeLoader
from net.talvi.puffinplot.data import FileType
from net.talvi.puffinplot.data import ArasonLevi
from net.talvi.puffinplot.data import RpiDataset
from java.util import HashMap
from java.io import File
from tglc import File as TwoGeeFile
import re


def assemble(directory, output_file, section_list, ms_only=False):
    """Assemble core sections into an entire core.

    section_list is a list of 4-tuples. Each 4-tuple has the
    following form:

    (name [str], section_length [int], empty_bottom [int], empty_top [int])

    "name" is a string label for the section, used to find the data
    file. section_length is the length of the actual core section,
    which may be shorter than the length of magnetometer track measured.
    empty_bottom and empty_top specify the lengths of the empty track
    sections below and above the core. These three integers should add
    up to the total length of measured data in the corresponding file.
    (If they don't, this function will throw an assertion error.)

    The ordering of the sections in section_list determines the
    order of section assembly. Sections are ordered from top to
    bottom, so the top of the first section in the list will be at 0 cm
    depth.

    A note on the empty_bottom and empty_top parameters: these are
    used to remove measured data that doesn't come from the core itself
    and doesn't correspond to any sampled depth. Such measurements are
    taken when the core section itself is shorter than the measured
    track length. For instance, if the 2G software is told to measure
    110cm of data, and a 100cm core is placed at a 5cm offset from
    the start of the track, the first and last 5cm will be "empty"
    measurements and can be removed from the data using empty_bottom
    and empty_top parameters.

    The sections removed using empty_bottom and empty_top are really
    removed from the depth stack. In contrast, data erased to avoid edge
    effects in the top/bottom few cm still leaves a gap with an
    associated depth: it's a missing measurement at a valid depth,
    whereas empty_bottom / empty_top removes measurements which don't
    even have a valid depth.

    There is a slight fencepost complication in the way depths are
    added up. On, say, a 100 cm core, measurements are taken at each
    point from 0 cm to 100 cm. The number of measurements is thus
    one more than the number of centimetres in the core. The first
    and last measurements don't have a unique depth: the bottom-most
    measurement of a core is at the same depth as the topmost measurement
    from the core below it. Thus, if we do an "edge-effect"  removal
    of four measurements from the adjacent ends of two cores, we're actually
    only blanking out measurements through 2 * 4 - 1 = 7 cm of depth.

    This isn't currently a general-purpose function: some things are
    hard-coded.

    Args:
        directory: the directory containing the core section files
        output_file: name for the the assembled file to be written
        section_list: list of section codes and truncation specifiers
        ms_only: True iff the files contain only magnetic susceptibility data

    """

    section_names = [section[0] for section in section_list]
    section_dict = {section[0]: (section[1], section[2], section[3])
                    for section in section_list}

    sections = []
    top = 0
    for section in section_names:
        f = TwoGeeFile()
        f.read(directory + "/" + ("C5%s.DAT" % section))
        edge_thickness = 4  # amount to chop off for edge effects
        section_length, empty_bottom, empty_top = \
            section_dict[section]
        f.truncate(empty_bottom,
                   empty_top)  # chop off the empty tray
        thickness = f.get_thickness()  # thickness of the actual mud
        assert thickness == section_length, \
            "specified length %d does not equal actual length %d" % \
            (section_length, thickness)
        f.truncate(edge_thickness, edge_thickness)

        def depth(x):
            return top + thickness - x*100 + empty_bottom

        f.change_depth(depth)
        if ms_only:
            f.chop_fields("Depth\tMS corr\n")
        sections.append(f)
        top += thickness
    
    composite = TwoGeeFile()
    composite.concatenate(sections)
    composite.sort()
    composite.write(output_file)
    print


def read_suite(filename, intensity_correction):
    """Read a suite from the combined DAT file,
    truncate it to the section of interest, and correct
    the intensities."""
    
    suite = Suite("C5")
    input_file = File(filename)
    sensor_lengths = SensorLengths.fromStrings("4.09", "4.16", "6.67")
    
    suite.readFiles([input_file], sensor_lengths, TwoGeeLoader.Protocol.NORMAL,
                    False, FileType.TWOGEE, None, HashMap())
    
    suite.removeSamplesOutsideDepthRange(20, 440)
    
    # Correct intensity: for the NRM measurements (but not the ARM), the
    # u-channel cross-section was 3.8 cm^2, but it wasn't written to the
    # DAT file so PuffinPlot defaults to 4 cm^2. We apply a constant
    # scale factor to each moment measurement to correct this.
    
    if intensity_correction is not None:
        for sample in suite.getSamples():
            for datum in sample.getData():
                datum.setMoment(datum.getMoment().times(intensity_correction))

    return suite


def read_row(line):
    parts = re.split(" +", line.strip())
    age = 2010 - int(parts[0])
    return (age, float(parts[1]), float(parts[2]), float(parts[3]))


def read_model_inclinations():
    """Read inclinations from model data.
    """

    with open("../ref-data/cals10k2/output-40_9735980556-13_7840316667") as fh:
        lines = fh.readlines()

    rows = [read_row(line) for line in lines[1:]]

    my_rows = [row for row in rows if row[0] > 100 and row[0] < 4500]
    cols = zip(*my_rows)
    return cols[2]


def print_aralev(inclinations):
    """Print Arason-Levi statistics for supplied inclinations.
    """

    al = ArasonLevi.calculate(inclinations)
    am = ArasonLevi.ArithMean.calculate(inclinations)
    print "%8s %8s %8s %8s" % \
        ("inc", "kappa", "t63", "a95")
    print "%8.1f %8.1f %8.1f %8.1f Arason-Levi" % \
        (al.getMeanInc(), al.getKappa(), al.getT63(), al.getA95())
    print "%8.1f %8.1f %8.1f %8.1f arithmetic mean" % \
        (am.getMeanInc(), am.getKappa(), am.getT63(), am.getA95())


def optimize_rpi_points(nrm_suite, arm_suite):
    """
    Find the "best" RPI treatment steps by brute force search among
    reasonable first and last treatment levels. We take the step
    selection which produces the largest minimum r-value -- i.e.
    that which makes the worst data point as good as possible.
    """
    mins = [0, 0.005, 0.01, 0.015, 0.02]
    maxs = [0.06, 0.08, 0.1]

    maxminr = -1
    params = None
    bestrpi = None
    for m0 in mins:
        for m1 in maxs:
            rpidata = RpiDataset.calculateWithArm(nrm_suite, arm_suite, m0, m1)
            rpis = rpidata.getRpis()
            rs = [rpi.getR() for rpi in rpis]
            meanr = sum(rs) / float(len(rs))
            minr = min(rs)
            # print "%g\t%g\t%.4f\t%.4f" % (m0, m1, meanr, minr)
            if minr > maxminr:
                params = m0, m1
                bestrpi = rpidata
                maxminr = minr

    bestrpi.writeToFile("../data/processed/rpi-arm-optimized.csv")
    print "RPI parameters:", params, maxminr


def main():

    # Assemble core sections into composite files.
    assemble("../data/nrm", "../data/nrm/c5-nrm-all.dat",
             [("H",  52, 5, 8),
              ("G", 100, 5, 5),
              ("F", 100, 5, 5),
              ("E", 100, 5, 5),
              ("D", 100, 5, 5),
              ("C",  52, 4, 4),
              ("B", 100, 5, 5),
              ("A",  99, 6, 5),
              ])
    assemble("../data/arm", "../data/arm/c5-arm-all.dat",
             [("H",  52, 5, 8),
              ("G", 100, 5, 5),
              ("F", 100, 5, 5),
              ("E", 100, 5, 5),
              ("D", 100, 5, 5),
              ("C",  52, 5, 4),
              ("B", 100, 5, 5),
              ("A",  99, 6, 5),
              ])
    assemble("../data/ms/2g-sections", "../data/ms/c5-lab-ms-all.dat",
             [("H",  52, 0, 3),
              ("G", 100, 0, 0),
              ("F", 100, 0, 0),
              ("E", 100, 0, 0),
              ("D", 100, 0, 0),
              ("C",  52, 2, 1),
              ("B", 100, 0, 0),
              ("A",  99, 1, 0),
              ],
             True)

    # Read the composite files into PuffinPlot suites.
    nrm_suite = read_suite("../data/nrm/c5-nrm-all.dat", 4.0/3.8)
    arm_suite = read_suite("../data/arm/c5-arm-all.dat", None)
    ms_suite = read_suite("../data/ms/c5-lab-ms-all.dat", None)
    nrm_samples = nrm_suite.getSamples()
    
    # Perform a PCA calculation for each sample.
    for sample in nrm_samples:
        sample.selectByTreatmentLevelRange(0.014999, 0.10001)

        # Exclude the 30 mT from C5D due to a flux jump
        data = sample.getData()
        name = data[0].getDiscreteId()
        if name == "C5D": data[6].setSelected(False)
        
        sample.useSelectionForPca()
        sample.setPcaAnchored(False)
        sample.calculateMdf()
    nrm_suite.doSampleCalculations(Correction.NONE)
    
    # Calculate the mean MAD3
    total_mad3 = sum([sample.getPcaValues().getMad3()
                      for sample in nrm_samples])
    print "Mean MAD3 = %f / %d = %f" % \
        (total_mad3, nrm_suite.getNumSamples(),
         total_mad3/ nrm_suite.getNumSamples())
    
    # Calculate the mean MDF
    total_mdf = sum([sample.getMdf().getDemagLevel()
                      for sample in nrm_samples])
    print "Mean MDF = %f / %d = %f" % \
        (total_mdf, nrm_suite.getNumSamples(),
         total_mdf/ nrm_suite.getNumSamples())
    
    
    # Count the number of MAD3 values under 2
    under2 = 0
    maxmad3 = 0
    for sample in nrm_samples:
        mad3 = sample.getPcaValues().getMad3() 
        if mad3 < 2: under2 += 1
        if mad3 > maxmad3: maxmad3 = mad3
    print "%d samples with MAD3 < 2" % under2
    print "Maximum MAD3: %.2f" % maxmad3
    
    # Calculate Arason-Levi mean inclination
    print "Statistics for C5 core"
    print_aralev([sample.getPcaValues().getDirection().getIncDeg()
                  for sample in nrm_samples])
    print "\nStatistics for model"
    print_aralev(read_model_inclinations())
    
    # Save the results of the PCA calculation.
    output_file = File("../data/processed/incs-from-script.csv")
    nrm_suite.saveCalcsSample(output_file)

    # Save a copy of the NRM suite
    nrm_suite.saveAs(File("../data/processed/nrms-from-script.ppl"))
    
    # Calculate RPI from ARM
    optimize_rpi_points(nrm_suite, arm_suite)

    # Calculate RPI from MS
    rpis_ms = RpiDataset.calculateWithMagSus(nrm_suite, ms_suite)
    rpis_ms.writeToFile("../data/processed/rpi-ms.csv")

    
if __name__ == "__main__":
    main()
