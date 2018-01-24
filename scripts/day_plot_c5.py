#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""Make a Day plot of the C5 data."""

from day_plot import *


def main():
    samples = ["c5a-35-1",
               "c5a-85-f",
               "c5b-85-1", "c5d-35",
               "c5d-85", "c5e-35", "c5h-01"]
    data_dir = "../data/rock-mag/"
    hyst_samples = [data_dir + "hyst/with-slope/" + s + "-hyst-s" for s in samples]
    irm_samples = [data_dir + "irm/micromag/" + s + "-irm" for s in samples]

    fields_in_file = ["Initial slope", "Saturation",
                      "Remanence", "Coercivity", "S*", "Coercivity (remanent)"]
    fields_to_calculate = ["Bcr/Bc", "Mrs/Ms"]
    fields_all = fields_in_file + fields_to_calculate

    sample_dict = read_micromag_files(hyst_samples + irm_samples,
                                      fields_in_file)
    calculate_day_plot_parameters_and_write_to_dict(sample_dict)
    write_fields_and_params_to_stdout(sample_dict, fields_all)
    xs, ys = get_day_plot_parameters_from_dict(sample_dict)
    make_plot("en", xs, ys)


if __name__ == "__main__":
    main()
