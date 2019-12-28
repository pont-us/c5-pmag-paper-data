#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""Make a Day plot of the C5 data."""

from day_plot import *
from plot_settings import set_matplotlib_parameters


def main():
    set_matplotlib_parameters()

    samples = ["c5a-35", "c5a-85", "c5b-30", "c5b-85", "c5b-90", "c5c-10",
               "c5c-50", "c5d-10", "c5d-30", "c5d-35", "c5d-70", "c5d-85",
               "c5e-10", "c5e-35", "c5e-50", "c5e-90", "c5f-10", "c5f-50",
               "c5f-70", "c5f-90", "c5g-40", "c5g-70", "c5g-90", "c5h-01",
               "c5h-10", "c5h-30", "c5h-50"]

    data_dir = "../data/rock-mag/micromag/"
    hyst_samples = [data_dir + s + "-hyst-slope" for s in samples]
    irm_samples = [data_dir + s + "-irm" for s in samples]

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
