import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rc

rc('text', usetex=True) # LaTeX formatter

def plot_src(meta, estimate, method, metrics={}, patches=[], title="", cmap='bone_r', filename=""):
    """Plot equivalent source hologram."""

    # extract source geometry from meta
    x_src = np.linspace(-0.5 * meta["lx"], 0.5 * meta["lx"], meta["Nx"])
    y_src = np.linspace(-0.5 * meta["ly"], 0.5 * meta["ly"], meta["Ny"])
    X_src, Y_src = np.meshgrid(x_src, y_src)

    # setup figure
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # set title
    if not title:
        title = r"Source Strength $\mathbf{{q}}$ - " + method
    ax.set_title(r"\textbf{{{0}}}".format(title), fontsize=20)

    # set axis labels
    ax.set_xlabel('x [cm]', fontsize=18)
    ax.set_ylabel('y [cm]', fontsize=18)
    
    # set ticks and ticklabels
    ax.tick_params(axis='both', which='major', labelsize=14)
    xticks = np.linspace(-0.5 * meta["lx"], 0.5 * meta["lx"], 5)
    yticks = np.linspace(-0.5 * meta["ly"], 0.5 * meta["ly"], 5)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # set ticklabels (m to cm)
    xticklabels = xticks * 1e2 
    yticklabels = yticks * 1e2
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    
    # plot source image
    c_src = np.abs(estimate)
    im = ax.scatter(
        x=X_src,
        y=Y_src,
        c=np.abs(estimate),
        cmap=cmap,
        linewidths=0.2,
        edgecolor='k',
    )

    # set colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.get_yaxis().labelpad = 25
    cbar.ax.set_ylabel('Magnitude [ - ]', rotation=270, fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    if c_src.max() < 0.1:
        cbar.formatter.set_powerlimits((0, 0)) # use scientifc notation for small values
        cbar.ax.yaxis.set_offset_position('left') # align scientific notation with cbar
    # add src patches
    for patch in patches:
        ax.add_patch(patch)

    # show metrics in textbox
    if metrics:
        # make space for metrics textbox
        ax.set_ylim([xticks[0] + 0.2 * xticks[0], xticks[-1] + 0.1 * xticks[-1]])
        # align metrics in str and restrict values to one decimal point
        metricstr = ' '.join(
            r'\textbf{{{0}}}: {1:.1f}'.format(metric.upper(), value) 
            for metric, value in metrics.items()
            )
        # create metrics textbox
        bbox = dict(boxstyle='round', facecolor='white', alpha=0.75)
        ax.text(
            x=0.0475, y=0.03, s=metricstr, transform=ax.transAxes, 
            fontsize=11, verticalalignment='baseline', bbox=bbox
            )
    
    # export plot
    if filename:
        fig.savefig(filename, bbox_inches='tight')
        plt.close()

    return fig


def plot_y_against_x(y, x, label, title, ylabel, xlabel, yrange=None, axvline=None, axvlabel="", filename=""):
    """Plot y against x for metric vs. frequency plots of Figure 6."""
    # setup figure
    fig, ax = plt.subplots()
    fig.subplots_adjust(wspace=1, hspace=1)

    # name axes
    plt.title(title, fontsize=20) 
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    
    # adjust hyperparams
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f')) 
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(which="major", alpha=0.5)
    if yrange is not None:
        ax.set_ylim(yrange)
    
    # plot y against x
    ax.plot(x, y, label=label, color="k", linewidth=2, marker="x", markevery=1)
    
    # plot axvline
    if axvline is not None:
        plt.axvline(x=axvline, color="k", linestyle="dashed", linewidth=1, label=axvlabel)

    # plot legend
    leg = plt.legend(loc="upper left", prop={'size': 13})
    leg.get_frame().set_edgecolor('black')

    # export plot
    if filename:
        fig.savefig(filename, bbox_inches='tight')
        plt.close()

    return fig

def plot_metric_evolution(metrics, method, filename=""):
    """Plot metric vs. layer/iterations of Figure 8."""
    # setup figure
    fig, ax = plt.subplots(2,1, sharex=True)
    adjust_aspect_ratio(fig, 1.25)
    fig.subplots_adjust(hspace=0)

    # name axes
    title = fr"Error {method} - Source Strength $\mathbf{{q}}$"
    ax[0].set_title(title, fontsize=20) 
    ax[1].set_xlabel(r"Layer $\mathbf{{q^t}}$", fontsize=18)
    ax[0].set_ylabel(r"[\%]", fontsize=18)
    ax[1].set_ylabel(r"[$\mathrm{dB}$]", fontsize=18)
    fig.supylabel(r"Error / Similarity", fontsize=18, x=0.05)

    # yticks
    # metrics in %
    ax[0].set_yticks(np.arange(0, 110, 20))
    ax[0].set_yticks(np.arange(0, 110, 10), minor=True)
    ax[0].set_ylim([0, 100])
    ax[0].tick_params(axis='both', which='major', labelsize=14)

    # metrics in dB
    ax[1].set_yticks(np.arange(0, 25, 5))
    ax[1].set_yticks(np.arange(0, 25, 10), minor=True)
    ax[1].set_ylim([0, 25])
    ax[1].tick_params(axis='both', which='major', labelsize=14)

    # xticks
    # layers/iterates 
    ax[1].set_xticks(np.arange(0, len(metrics)))
    ax[1].set_xticklabels([fr"$\mathbf{{q}}^{{{t}}}$" for t in range(len(metrics))])
    ax[0].grid(which="major", alpha=0.5)
    ax[1].grid(which="major", alpha=0.5)
    ax[0].set_xlim(0, len(metrics)-1)

    # extract metrics
    metric_keys = ["rmse", "ncc", "ssim", "psnr"]
    metric_specs = {
        "rmse": {"ax": 0, "linestyle": "solid", "color": "black", "marker": "o"},
        "ncc": {"ax": 0, "linestyle": "dashdot", "color": "black", "marker": "s"},
        "ssim": {"ax": 0, "linestyle": "dotted", "color": "black", "marker": "x"},
        "psnr": {"ax": 1, "linestyle": "dashed", "color": "black", "marker": "^"},
        }
    for key in metric_keys:
        metric = [metric[key] for metric in metrics]
        specs = metric_specs[key]
        ax[specs["ax"]].plot(
            metric,
            label=key.upper(), 
            linestyle=specs["linestyle"], 
            color="k",
            linewidth=2,
            marker=specs["marker"],
            fillstyle="none",
            )
    
    # set legend
    leg = fig.legend(
        loc="lower right", 
        prop={'size': 12}, 
        bbox_to_anchor=(0.808, 0.168), 
        ncol=4, 
        columnspacing=0.5
        )
    leg.get_frame().set_edgecolor('black')

    # export plot
    if filename:
        fig.savefig(filename, bbox_inches='tight')
        plt.close()

    return fig

def adjust_aspect_ratio(fig,aspect=1):
    """Adjust subplot aspect ratio."""
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize, ysize)
    xlim = .4 * minsize / xsize
    ylim = .4 * minsize / ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(
        left=.5-xlim,
        right=.5+xlim,
        bottom=.5-ylim,
        top=.5+ylim
        )