import matplotlib


def set_matplotlib_parameters():
    fontname = "NimbusSanL"
    font = {"family": fontname,
            "weight": "normal",
            "size": 10}
    matplotlib.rc("font", **font)
    matplotlib.rcParams["mathtext.fontset"] = "custom"
    matplotlib.rcParams["mathtext.it"] = fontname + ":italic"
    matplotlib.rcParams["mathtext.rm"] = fontname
    matplotlib.rcParams["mathtext.tt"] = fontname
    matplotlib.rcParams["mathtext.bf"] = fontname
    matplotlib.rcParams["mathtext.cal"] = fontname
    matplotlib.rcParams["mathtext.sf"] = fontname
    matplotlib.rcParams["legend.fontsize"] = 7
    matplotlib.rcParams["legend.numpoints"] = 1
    matplotlib.rcParams["legend.labelspacing"] = 0.1


PLOT_WIDTH = 180 / 25.4
