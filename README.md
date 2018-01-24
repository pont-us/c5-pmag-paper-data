This repository contains material associated with the paper ‘A
4,500-year record of paleosecular variation and relative paleointensity
from the Tyrrhenian Sea’, submitted for publication in *Journal of
Geophysical Research – Solid Earth* by Lurcock et al. in 2018.

The repository contains the code and data used to produce the results
described in the paper. The contents are as follows.

Directory | Description
----------|-------------------------------------
data      | Palaeomagnetic data from the C5 core
libraries | software libraries used by processing scripts
ref-data  | Data sets and models used for comparison and tuning
script-output | Output from the data processing scripts
scripts | Scripts to analyse and plot the data

Publicly available third-party data is not included in this repository.
Rather, it is automatically downloaded as required and cached in the
ref-data directory.

To run the scripts in this repository, the following software (all
freely available) must be installed.

* Python (the standard CPython implementation), versions 2.7 and 3,
  including the standard libraries
* Additional Python libraries: numpy, matplotlib, scipy, pexpect
* Jython, version 2.7
* Match ( http://www.lorraine-lisiecki.com/match.html ), version 2.3.1

The analysis scripts also make use of the PuffinPlot program (
https://puffinplot.bitbucket.io/ ). For convenience, a pre-built archive
of the appropriate version is included in the libraries directory; it
was produced from commit a08a863 (2018-01-24) in the PuffinPlot
repository.
