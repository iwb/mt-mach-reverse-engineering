import matplotlib.pyplot as plt

COLOR_DEFAULT_BLUE = "#0065BD"
COLOR_DEFAULT_BLACK = "#000000"
COLOR_ACCENT_ORANGE = "#E37222"
COLOR_GREY_50 = "#808080"

DPI = 300


def set_figure_config():
    _text_font_size_title = 12
    _text_font_size_large = 10
    _text_font_size_small = 8
    plt.rcParams.update(
        {
            "figure.titlesize": _text_font_size_title,
            "axes.titlesize": _text_font_size_large,
            "font.size": _text_font_size_small,
            "axes.labelsize": _text_font_size_large,
            "xtick.labelsize": _text_font_size_small,
            "ytick.labelsize": _text_font_size_small,
            "legend.fontsize": _text_font_size_small,
            "legend.title_fontsize": _text_font_size_small,
        }
    )

    plt.rcParams.update({"lines.linewidth": 1})
    plt.rcParams.update({"lines.markersize": 5})

    plt.rcParams.update({"axes.grid": True})
    plt.rcParams.update({"legend.labelspacing": 0.1})

    plt.rcParams.update({"font.family": "Arial"})

    plt.rcParams["axes.autolimit_mode"] = "round_numbers"
    plt.rcParams["axes.xmargin"] = 0
    plt.rcParams["axes.ymargin"] = 0

    plt.rcParams.update(
        {
            "figure.subplot.left": 0.0,
            "figure.subplot.right": 1.0,
            "figure.subplot.bottom": 0.0,
            "figure.subplot.top": 1.0,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01,
        }
    )
    plt.rcParams.update(
        {
            "axes.prop_cycle": plt.cycler(
                color=[
                    COLOR_DEFAULT_BLUE,
                    COLOR_DEFAULT_BLACK,
                    "#005293",
                    "#003359",
                    "#808080",
                ]
            )
        }
    )

    plt.rcParams.update(
        {
            "figure.figsize": (one_column_width(), one_column_width() / 1.61),
            "figure.dpi": DPI,
        }
    )
    return


def one_column_width() -> float:
    return 8.4 / 2.54


def two_column_width() -> float:
    return 17.3 / 2.54
