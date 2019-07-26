"""Helper functions to make matplotlib a bit nicer"""

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

set2 = [mpl.cm.get_cmap('Set2', 8)(i)[:-1] for i in range(8)]
# [:-1] to remove alpha
almost_black = '#262626'
light_grey = [0.97254902, 0.97254902, 0.97254902]

set2 = [set2[i] for i in [2, 7, 4, 1, 0, 3, 6, 5]]
bgcolor = 'w'

bbox_props = dict(boxstyle="round4", fc="linen", ec=almost_black, lw=1,
                  alpha=.5)


def change_default():
    """Change the default properties of matplotlib graphs

    Importing seaborn also change these, so you might want to call
    that after the import

    """
    # mpl.rc('patch', linewidth=0.75, facecolor='none',
    #           edgecolor=set2[0])
    mpl.rc('axes', edgecolor=almost_black,
           labelcolor=almost_black,
           linewidth=0.5,
           prop_cycle=mpl.cycler('color', set2))
    mpl.rcParams['grid.color'] = bgcolor
    mpl.rcParams['ytick.color'] = almost_black
    mpl.rcParams['xtick.color'] = almost_black
    mpl.rcParams['text.color'] = almost_black
    mpl.rc('figure', facecolor=bgcolor,
           edgecolor=almost_black)


def clean(thing, spines=('top', 'right')):
    """Remove part of the box around the axes

    `thing` can be a axes instance or a figure."""
    if isinstance(thing, mpl.figure.Figure):
        axes = thing.axes
    elif isinstance(thing, mpl.axes.Axes):
        axes = [thing]
    else:
        axes = thing

    kept = {'top': 1,
            'bottom': 2,
            'right': 1,
            'left': 2}
    xpos = ['none', 'top', 'bottom', 'both']
    ypos = ['none', 'right', 'left', 'both']
    for ax in axes:
        [ax.spines[sp].set_visible(True) for sp in kept.keys()]
        for sp in spines:
            ax.spines[sp].set_visible(False)
            kept[sp] = 0
        ax.xaxis.set_ticks_position(xpos[kept['top'] + kept['bottom']])
        ax.yaxis.set_ticks_position(ypos[kept['right'] + kept['left']])
        ax.set_facecolor(bgcolor)


def scat_hist(fig, xdata, ydata, xbins, ybins, hist_kwargs={}, **kwargs):
    """"""
    # Add axes at position left, bottom, width, height
    # Position is given as a fraction of figure size
    left, width = 0.1, 0.6  # Scatter plot position
    bottom, height = 0.1, 0.6  # Scatter plot position
    bottom_h = left_h = left + width + 0.07  # Histograms position

    # Define rect positions
    scatter_rect = [left, bottom, width, height]
    histrectx = [left, bottom_h, width, 0.2]
    histrecty = [left_h, bottom, 0.2, height]

    # Create the different axes
    axscatter = fig.add_axes(scatter_rect)
    axhistx = fig.add_axes(histrectx)
    axhisty = fig.add_axes(histrecty)

    # Plot
    axscatter.spines['top'].set_visible(False)
    axscatter.spines['right'].set_visible(False)
    axscatter.yaxis.set_ticks_position('none')
    axscatter.xaxis.set_ticks_position('bottom')

    axscatter.scatter(xdata, ydata, **kwargs)

    hist_light(axhistx, xdata, bins=xbins, **hist_kwargs)
    hist_light(axhisty, ydata, bins=ybins, orientation='horizontal', **hist_kwargs)
    # axhistx.hist(xData, bins=xBins)
    # axhisty.hist(yData, bins=yBins,orientation='horizontal')

    axhistx.set_xlim(axscatter.get_xlim())
    axhisty.set_ylim(axscatter.get_ylim())
    axhistx.set_xticklabels([])
    axhisty.set_yticklabels([])
    return axscatter, axhistx, axhisty


def hist_light(ax, data, **kwargs):
    """ Function for plotting a simplified histogram """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if 'orientation' in kwargs:
        if kwargs['orientation'] == "horizontal":
            ax.grid(axis='x', color='white', linestyle='-', which='major')
    else:
        ax.grid(axis='y', color='white', linestyle='-', which='major')

    foo = ax.hist(data, **kwargs)
    return foo


def paired_plot(fig, datas, bins=None):
    """It should plot all pairwise scatter plots out of datas (a 2D
    array) but I broke it a while ago and never needed it since then

    """
    n = len(datas)
    if n < 2:
        raise ValueError('Need at least 2 lines of data')
    if bins is None:
        bins = [max(len(d) / 10, 10) for d in datas]
    elif isinstance(bins, int):
        bins = [bins] * n
    hist_axes = [fig.add_subplot(n, n, i * (n + 1) + 1) for i in range(n)]
    for line, d in enumerate(datas):
        h = hist_axes[line]
        hist_light(h, d, bins=bins[line])
        if line != n - 1:
            h.set_xticklabels('')
        for col in range(line):
            ax = fig.add_subplot(n, n, line * n + col + 1)
            ax.scatter(datas[col], d)
            clean(ax)
            if col != 0:
                ax.set_yticklabels('')
            if line != n - 1:
                ax.set_xticklabels('')
            else:
                ax.set_xticklabels([l.get_text() for l in ax.get_xticklabels()],
                                   rotation=20)


def barplot(ax, data, labels=None, ylabel=None, repeated_measure=False,
            mean_sd=True, dotcolor=None, add_dots=True):
    """Barplotise one 2-D array with nprop lines and N column in ax"""

    nprop = len(data)
    if labels is None:
        labels = ['prop %i' % i for i in range(nprop)]
    elif len(labels) != nprop:
        raise ValueError('need %s labels' % nprop)

    dots = []
    if add_dots:
        if dotcolor is None:
            dotcolor = ['k' for i in range(len(data))]
        x = np.random.rand(max([i.size for i in data])) / 2. - 0.25
        for ind, line in enumerate(data):
            dots.append(ax.scatter(np.zeros_like(line) + ind + x[:line.size], line, marker='o',
                                   alpha=.5, color=dotcolor))
        if repeated_measure:
            col = [[data[col][line] for col in range(nprop)] for line in range(len(data[0]))]
            lines = [ax.plot(np.arange(len(c)) + x[i], c, lw=.3, marker='',
                             c='grey', alpha=.5) for i, c in enumerate(col)]
    if mean_sd:
        erbar = ax.errorbar(np.arange(nprop), [d.mean() for d in data],
                            [d.std() / np.sqrt(d.size) for d in data],
                            ls='', ms=9, mew=2, marker='_', c=set2[2],
                            ecolor=set2[0], alpha=1, elinewidth=10, capsize=0)

    ax.yaxis.grid(True, linestyle='-', which='major',
                  color='lightgrey', alpha=0.5)
    ax.set_axisbelow(True)
    bp1 = ax.boxplot(list(data), notch=0, sym='+', vert=1,
                     whis=1.5, positions=range(nprop))
    plt.setp(bp1['boxes'], color=almost_black)
    plt.setp(bp1['whiskers'], color=almost_black,
             linestyle='-')
    plt.setp(bp1['caps'], ls='')
    plt.setp(bp1['fliers'], color=set2[0], marker='+')
    plt.setp(bp1['medians'], color=set2[0])

    maxi = max([d.max() for d in data])
    mini = min([d.min() for d in data])
    ptp = max([d.ptp() for d in data])

    top = maxi + ptp * 0.1
    bottom = mini - ptp * 0.1
    ax.set_ylim(bottom, top)
    ax.set_xlim(-.5, nprop - 0.5)
    ax.set_xticks(np.arange(nprop))
    ax.set_xticklabels(labels, rotation=45)

    pos = np.arange(nprop)
    upperlabels = [str(np.round(np.median(d), 2)) for d in data]
    for tick, label in zip(range(nprop), ax.get_xticklabels()):
        ax.text(pos[tick], 0.95, upperlabels[tick],
                horizontalalignment='center', size='small',
                color='k', transform=ax.get_xaxis_transform())
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    # left = ax.get_xlim()[0]
    # ax.text(left + abs(left)*0.05, top-(top*0.05), 'median:',
    #         horizontalalignment='left', size='small', color = 'k')
    if not mean_sd:
        return dots, bp1
    return dots, erbar, bp1


def draw_polar_polygon(axis, Xs, Ys, units='deg',
                       color='blue', label='', alpha_patch=0.1,
                       scatter_kwargs={}, edge_kwargs={}, area_kwargs = {}):
    """From: http://www.science-emergence.com/Matplotlib/MatplotlibGallery/RadarChartMatplotlibRougier/"""
    # -----------------------------------------------------------------------------
    # Copyright (C) 2011  Nicolas P. Rougier
    #
    # All rights reserved.
    #
    # Redistribution and use in source and binary forms, with or without
    # modification, are permitted provided that the following conditions are met:
    #
    # * Redistributions of source code must retain the above copyright notice, this
    #   list of conditions and the following disclaimer.
    #
    # * Redistributions in binary form must reproduce the above copyright notice,
    #   this list of conditions and the following disclaimer in the documentation
    #   and/or other materials provided with the distribution.
    #
    # * Neither the name of the glumpy Development Team nor the names of its
    #   contributors may be used to endorse or promote products derived from this
    #   software without specific prior written permission.
    #
    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    # ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    # LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    # CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    # SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    # INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    # CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    # ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    # POSSIBILITY OF SUCH DAMAGE.
    #
    # -----------------------------------------------------------------------------
    import matplotlib.path as path
    import matplotlib.patches as patches

    # convert if needed
    if units.lower() == 'deg':
        if all(Xs < 63):
            print('I\'m expecting degrees you know?')
            print('I will proceed but it looks like you gave me radians')
        Xs = np.deg2rad(Xs)
    elif units.lower() != 'rad':
        raise IOError('Unknown units')
    else:
        if any(Xs > 6.3):  # corresponds to 360 degrees
            print('I\'m expecting radians you know?')
            print('I will proceed but it looks like you gave me degrees')

    # Draw polygon representing values
    points = [(x, y) for x, y in zip(Xs, Ys)]
    points.append(points[0])
    points = np.array(points)
    codes = [path.Path.MOVETO, ] + \
            [path.Path.LINETO, ] * (len(Ys) - 1) + \
            [path.Path.CLOSEPOLY]

    _path = path.Path(points, codes)
    default = dict(fill=True, color=color, linewidth=0,
                   alpha=alpha_patch, label=label)
    default.update(area_kwargs)
    _patch = patches.PathPatch(_path, **default)
    axis.add_patch(_patch)

    default = dict(fill=False, linewidth=2, color=color)
    default.update(edge_kwargs)
    _patch = patches.PathPatch(_path, **default)
    axis.add_patch(_patch)


    # Draw circles at value points
    default = dict(linewidth=2, s=50, color=color, edgecolor='black', zorder=10)
    default.update(scatter_kwargs)
    axis.scatter(points[:, 0], points[:, 1], **default)
