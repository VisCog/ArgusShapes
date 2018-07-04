import numpy as np
import scipy.stats as spst


def scatter_correlation(xvals, yvals, ax, xticks=[], yticks=[], marker=None,
                        color=None, textloc='top right'):
    """Scatter plots some data points and fits a regression curve to them"""
    xvals = np.asarray(xvals)
    yvals = np.asarray(yvals)
    assert np.all(xvals.shape == yvals.shape)
    # Scatter plot the data:
    ax.scatter(xvals, yvals, marker=marker, s=50,
               c=color, edgecolors='white', alpha=0.5)

    # Set axis properties:
    if len(xticks) > 0:
        x_range = np.max(xticks) - np.min(xticks)
        xlim = (np.min(xticks) - 0.1 * x_range, np.max(xticks) + 0.1 * x_range)
        ax.set_xticks(xticks)
        ax.set_xlim(*xlim)
    if len(yticks) > 0:
        y_range = np.max(yticks) - np.min(yticks)
        ylim = (np.min(yticks) - 0.1 * y_range, np.max(yticks) + 0.1 * y_range)
        ax.set_yticks(yticks)
        ax.set_ylim(*ylim)

    # Need at least two data points to fit the regression curve:
    if len(xvals) < 2:
        return

    # Fit the regression curve:
    slope, intercept, rval, pval, _ = spst.linregress(xvals, yvals)
    fit = lambda x: slope * x + intercept
    ax.plot([np.min(xvals), np.max(xvals)], [
            fit(np.min(xvals)), fit(np.max(xvals))], 'k--')

    # Annotate with fitting results:
    va, ha = textloc.split(' ')
    assert ha == 'left' or ha == 'right'
    assert va == 'top' or va == 'bottom'
    a = ax.axis()
    xt = np.max(xticks) if len(xticks) > 0 else a[1]
    yt = np.min(yticks) if len(yticks) > 0 else (a[3] if va == 'top' else a[2])
    if pval >= 0.001:
        ax.text(xt, yt,
                "$N$=%d\n$r$=%.3f\n$p$=%.3f" % (len(yvals), rval, pval),
                va='top', ha='right')
    else:
        ax.text(xt, yt,
                "$N$=%d\n$r$=%.3f\n$p$=%.2e" % (len(yvals), rval, pval),
                va='top', ha='right')
