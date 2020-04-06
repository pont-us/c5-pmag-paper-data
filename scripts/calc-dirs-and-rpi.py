#!/usr/bin/env jython

"""
Calculate CRM directions from NRM data and RPI estimates
from NRM and ARM data. This script uses PuffinPlot's API, so it
must be run using Jython rather than the more common CPython
interpreter, and it can't be integrated with any script that use
CPython libraries such as numpy.
"""

from __future__ import print_function
import re
from tglc import assemble_sections
import sys
sys.path.append("../libraries/java/PuffinPlot.jar")
from net.talvi.puffinplot.data import Suite
from net.talvi.puffinplot.data import Correction
from net.talvi.puffinplot.data import SensorLengths
from net.talvi.puffinplot.data.file import TwoGeeLoader
from net.talvi.puffinplot.data import FileType
from net.talvi.puffinplot.data import ArasonLevi
from net.talvi.puffinplot.data import SuiteRpiEstimate
from net.talvi.puffinplot.data import CoreSections
from net.talvi.puffinplot.data.file import TwoGeeLoader
from java.util import HashMap
from java.io import File


def main():
    do_c5_calculations()


def do_c5_calculations():
    assemble_c5_sections()

    # Read the composite files into PuffinPlot suite objects.
    nrm_suite = read_suite("../data/nrm/c5-nrm-all.dat", 4.0 / 3.8, False)
    arm_suite = read_suite("../data/arm/c5-arm-all.dat", None, True)
    irm_suite = read_suite("../data/irm/c5-irm-all.dat", None, True)
    ms_suite = read_suite("../data/ms/c5-lab-ms-all.dat", None, True)

    do_sample_calculations(nrm_suite)

    # Save the results of the sample calculations.
    nrm_suite.saveCalcsSample(File("../data/processed/psv-from-script.csv"))

    print_sample_statistics(nrm_suite.getSamples())

    # Save a copy of the NRM suite
    nrm_suite.saveAs(File("../data/processed/nrms-from-script.ppl"))

    # Calculate RPI from ARM
    optimize_rpi_points(nrm_suite, arm_suite, "rpi-arm-optimized.csv")

    # Calculate RPI from IRM
    optimize_rpi_points(nrm_suite, irm_suite, "rpi-irm-optimized.csv")

    # Calculate RPI from MS
    rpis_ms = SuiteRpiEstimate.calculateWithMagSus(nrm_suite, ms_suite)
    rpis_ms.writeToFile("../data/processed/rpi-ms.csv")


def do_sample_calculations(nrm_suite):
    # Perform a PCA calculation for each sample.
    for sample in nrm_suite.getSamples():
        sample.selectByTreatmentLevelRange(0.014999, 0.10001)

        # Exclude the 30 mT from C5D due to a flux jump
        data = sample.getTreatmentSteps()
        name = data[0].getDiscreteId()
        if name == "C5D":
            data[6].setSelected(False)

        sample.useSelectionForPca()
        sample.setPcaAnchored(False)
        sample.calculateMdf()
    nrm_suite.doSampleCalculations(Correction.NONE)

    # Align declinations and remove edge samples
    
    core_sections = CoreSections.fromSampleListByDiscreteId(
            nrm_suite.getSamples())
    core_sections.alignSections(10, 0, CoreSections.TargetDeclinationType.MEAN)
    nrm_suite.removeSamples(core_sections.getEndSamples(4))
    nrm_suite.removeSamplesOutsideDepthRange(20, 440)    


def print_sample_statistics(nrm_samples):
    # Calculate the mean MAD3
    total_mad3 = sum([sample.getPcaValues().getMad3()
                      for sample in nrm_samples])
    print("Mean MAD3 = %f / %d = %f" %
          (total_mad3, len(nrm_samples),
           total_mad3 / len(nrm_samples)))

    # Calculate the mean MDF
    total_mdf = sum([sample.getMdf().getDemagLevel()
                     for sample in nrm_samples])
    print("Mean MDF = %f / %d = %f" %
          (total_mdf, len(nrm_samples),
           total_mdf / len(nrm_samples)))

    # Count the number of MAD3 values under 2
    under2 = 0
    maxmad3 = 0
    for sample in nrm_samples:
        mad3 = sample.getPcaValues().getMad3()
        if mad3 < 2:
            under2 += 1
        if mad3 > maxmad3:
            maxmad3 = mad3
    print("%d samples with MAD3 < 2" % under2)
    print("Maximum MAD3: %.2f" % maxmad3)

    # Calculate Arason-Levi mean inclination
    print("Statistics for core")
    print_aralev([sample.getPcaValues().getDirection().getIncDeg()
                  for sample in nrm_samples])
    print("\nStatistics for CALS10k.2")
    print_aralev(read_model_inclinations_cals10k2())
    print("\nStatistics for SHA.DIF.14k")
    print_aralev(read_model_inclinations_shadif14k())


def assemble_c5_sections():
    # Assemble core sections into composite files.
    assemble_sections("../data/nrm/C5%s.DAT", "../data/nrm/c5-nrm-all.dat",
                      [("H", 52, 5, 8),
                       ("G", 100, 5, 5),
                       ("F", 100, 5, 5),
                       ("E", 100, 5, 5),
                       ("D", 100, 5, 5),
                       ("C", 52, 4, 4),
                       ("B", 100, 5, 5),
                       ("A", 99, 6, 5),
                       ])
    assemble_sections("../data/arm/C5%s.DAT", "../data/arm/c5-arm-all.dat",
                      [("H", 52, 5, 8),
                       ("G", 100, 5, 5),
                       ("F", 100, 5, 5),
                       ("E", 100, 5, 5),
                       ("D", 100, 5, 5),
                       ("C", 52, 5, 4),
                       ("B", 100, 5, 5),
                       ("A", 99, 6, 5),
                       ])
    assemble_sections("../data/irm/C5%s.DAT", "../data/irm/c5-irm-all.dat",
                      [("H", 52, 0, 0),
                       ("G", 100, 0, 0),
                       ("F", 99, 0, 1, 1),  # omit duplicate 152cm measurement
                       ("E", 100, 0, 0),
                       ("D", 100, 0, 0),
                       ("C", 51, 0, 0, 1),  # 1 cm space at top
                       ("B", 100, 0, 0),
                       ("A", 98, 0, 0, 1),  # 1 cm space at top
                       ])
    assemble_sections("../data/ms/2g-sections/C5%s.DAT",
                      "../data/ms/c5-lab-ms-all.dat",
                      [("H", 52, 0, 3),
                       ("G", 100, 0, 0),
                       ("F", 100, 0, 0),
                       ("E", 100, 0, 0),
                       ("D", 100, 0, 0),
                       ("C", 52, 2, 1),
                       ("B", 100, 0, 0),
                       ("A", 99, 1, 0),
                       ],
                      True)


def read_suite(filename, intensity_correction,
               remove_section_ends_and_truncate):
    """Read a suite from the combined DAT file, truncate it to the section of
    interest, and correct the intensities.

    intensity_correction: if not None, scale the demagnetization step
    moments by this value.

    top_declination: if not None, align the declinations, using the
    given value as the top declination in degrees.
    """
    
    suite = Suite("C5")
    input_file = File(filename)
    sensor_lengths = SensorLengths.fromStrings("4.09", "4.16", "6.67")

    load_options = HashMap()
    load_options.put("protocol", TwoGeeLoader.Protocol.NORMAL)
    load_options.put("sensor_lengths", sensor_lengths)
    load_options.put("read_moment_from", TwoGeeLoader.MomentFields.CARTESIAN)
    
    suite.readFiles([input_file], FileType.TWOGEE, load_options)
    
    # Correct intensity: for the NRM measurements (but not the ARM), the
    # u-channel cross-section was 3.8 cm^2, but it wasn't written to the
    # DAT file so PuffinPlot defaults to 4 cm^2. We apply a constant
    # scale factor to each moment measurement to correct this.
    
    if intensity_correction is not None:
        for sample in suite.getSamples():
            for datum in sample.getTreatmentSteps():
                datum.setMoment(datum.getMoment().times(intensity_correction))

    # Remove the top and bottom few measurements of each section, if
    # requested.
    if remove_section_ends_and_truncate:
        core_sections = CoreSections.fromSampleListByDiscreteId(
            suite.getSamples())
        suite.removeSamples(core_sections.getEndSamples(4))
        suite.removeSamplesOutsideDepthRange(20, 440)
        
    return suite


def _read_cals10k2_inclination_row(line):
    parts = re.split(" +", line.strip())
    age = 2010 - int(parts[0])
    return age, float(parts[1]), float(parts[2]), float(parts[3])


def read_model_inclinations_cals10k2():
    """Read inclinations from model data.
    """

    with open("../ref-data/cals10k2/output-40_9735980556-13_7840316667") as fh:
        lines = fh.readlines()

    rows = [_read_cals10k2_inclination_row(line) for line in lines[1:]]

    my_rows = [row for row in rows if 100 < row[0] < 4500]
    cols = zip(*my_rows)
    return cols[2]


def read_model_inclinations_shadif14k():

    with open("../ref-data/sha-dif-14k/output_40p973598_13p784032.dat") as fh:
        lines = fh.readlines()

    def read_row(line):
        parts = re.split(" +", line.strip())
        age = 1950 - int(float(parts[0]))
        inc = float(parts[5])
        return age, inc

    rows = [read_row(line) for line in lines]
    my_rows = [row for row in rows if 100 < row[0] < 4500]
    cols = zip(*my_rows)
    return cols[1]


def print_aralev(inclinations):
    """Print Arason-Levi statistics for supplied inclinations.
    """

    al = ArasonLevi.calculate(inclinations)
    am = ArasonLevi.ArithMean.calculate(inclinations)
    print("%8s %8s %8s %8s" %
          ("inc", "kappa", "t63", "a95"))
    print("%8.1f %8.1f %8.1f %8.1f Arason-Levi" %
          (al.getMeanInc(), al.getKappa(), al.getT63(), al.getA95()))
    print("%8.1f %8.1f %8.1f %8.1f arithmetic mean" %
          (am.getMeanInc(), am.getKappa(), am.getT63(), am.getA95()))


def optimize_rpi_points(nrm_suite, normalizer_suite, filename):
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
            rpidata = SuiteRpiEstimate.calculateWithStepwiseAF(
                nrm_suite, normalizer_suite, m0, m1)
            rpis = rpidata.getRpis()
            rs = [rpi.getR() for rpi in rpis]
            minr = min(rs)
            if minr > maxminr:
                params = m0, m1
                bestrpi = rpidata
                maxminr = minr

    bestrpi.writeToFile("../data/processed/" + filename)
    rpis = bestrpi.getRpis()
    print("RPI parameters %s :" % filename, params, maxminr)
    print("Mean R^2:", sum([rpi.getrSquared() for rpi in rpis]) / len(rpis))


if __name__ == "__main__":
    main()
