import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.legend_handler import HandlerLine2D


def pad_a_like_b(a, b):
    """Pad a to have the same number of dimensions as b."""
    if isinstance(a, float | int):
        return a
    return a.reshape(-1, *([1] * (b.dim() - 1)))


def create_plot_grid(n_classes, figsize=None, ncols=None):
    """Create a grid of subplots for plotting."""
    if ncols is not None:
        ncols = ncols
    else:
        if (n_classes == 2) | (n_classes == 4):
            ncols = 2
        else:
            ncols = 3
    nrows = int(np.ceil(n_classes / ncols))
    figsize = (5 * ncols, 4 * nrows) if figsize is None else figsize
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)

    # Flatten the axes array for easy iteration
    if n_classes == 1:
        ax = [ax]
    if n_classes >= 2:
        ax = np.atleast_2d(ax)

    # Hide any unused subplots
    for i in range(n_classes, len(ax)):
        ax[i].set_visible(False)

    return fig, ax, ncols, nrows


#
# Plotting
#

COLORMAP10 = [
    "#38A6A5FF",
    "#0F8554FF",
    "#73AF48FF",
    "#EDAD08FF",
    "#E17C05FF",
    "#CC503EFF",
    "#94346EFF",
    "#6F4070FF",
    "#994E95FF",
    "#666666FF",
]

COLORMAP12 = [
    "#5F4690FF",
    "#1D6996FF",
    "#38A6A5FF",
    "#0F8554FF",
    "#73AF48FF",
    "#EDAD08FF",
    "#E17C05FF",
    "#CC503EFF",
    "#94346EFF",
    "#6F4070FF",
    "#994E95FF",
    "#666666FF",
]

COLORMAP21 = [
    "#5F4690FF",
    "#1D6996FF",
    "#38A6A5FF",
    "#0F8554FF",
    "#73AF48FF",
    "#EDAD08FF",
    "#E17C05FF",
    "#CC503EFF",
    "#94346EFF",
    "#6F4070FF",
    "#994E95FF",
    "#666666FF",
    "#A02818FF",
    "#D55E00FF",
    "#CC503EFF",
    "#A05195FF",
    "#332288FF",
    "#6699CCFF",
    "#88CCEEFF",
    "#44AA99FF",
    "#117733FF",
]


class ThickerLine2D(HandlerLine2D):
    """Custom handler for thicker lines in legends."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        line = super().create_artists(legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)
        line[0].set_linewidth(3)
        line[0].set_markersize(10)
        return line


def color_picker(color_classes):
    """Select a color based on the number of classes."""
    if len(color_classes) >= 20:

        def generate_distinct_colors(n):
            colors = []
            for i in range(n):
                hue = i / n
                saturation = 0.9  # High saturation for vivid colors
                value = 0.9  # High value for brightness
                color = plt.cm.hsv(hue)[:3]  # Convert HSV to RGB
                colors.append(color)
            return colors

        colors = generate_distinct_colors(len(color_classes))

    elif (len(color_classes) > 10) & (len(color_classes) < 20):
        colors = sns.color_palette("tab20")[: len(color_classes)]
    else:
        colors = sns.color_palette("tab10")[: len(color_classes)]

    return colors
