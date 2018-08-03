import numpy as np
import scipy.stats as spst
import skimage.io as skio
import skimage.transform as skit

import pulse2percept.implants as p2pi
import pulse2percept.retina as p2pr

import os.path as osp
import pkg_resources
data_path = pkg_resources.resource_filename('argus_shapes', 'data/')

from . import imgproc


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


def plot_phosphenes_on_array(ax, subject, Xymu, subjectdata):
    """Plots phosphenes centered over the corresponding electrodes"""
    img_argus1 = skio.imread(osp.join(data_path, 'argus_i.png'))
    img_argus2 = skio.imread(osp.join(data_path, 'argus_ii.png'))
    px_argus1 = np.array([
        [93.12857037,  162.32202802], [138.00952276,  163.7029804],
        [178.74761799,  163.01250421], [227.77142752,  161.63155183],
        [93.12857037,  208.58393279], [143.53333228,  207.8934566],
        [182.89047514,  207.2029804], [227.77142752,  206.51250421],
        [93.12857037,  251.3934566], [137.31904657,  251.3934566],
        [180.81904657,  251.3934566], [227.08095133,  251.3934566],
        [93.81904657,  296.27440898], [140.08095133,  296.27440898],
        [182.89047514,  297.65536136], [227.08095133,  297.65536136]
    ])
    # px_argus1[:, 0] += 70
    # px_argus1[:, 1] -= 70

    px_argus2 = np.array([
        [296.94026284,  140.58506571], [328.48148148,  138.4823178],
        [365.27956989,  140.58506571], [397.87216249,  139.53369176],
        [429.41338112,  138.4823178], [463.05734767,  140.58506571],
        [495.64994026,  139.53369176], [528.24253286,  139.53369176],
        [560.83512545,  139.53369176], [593.42771804,  138.4823178],
        [296.94026284,  173.1776583], [329.53285544,  174.22903226],
        [363.17682198,  173.1776583], [396.82078853,  173.1776583],
        [430.46475508,  173.1776583], [463.05734767,  174.22903226],
        [494.59856631,  173.1776583], [529.29390681,  174.22903226],
        [559.78375149,  175.28040621], [593.42771804,  173.1776583],
        [296.94026284,  206.82162485], [329.53285544,  206.82162485],
        [363.17682198,  205.7702509], [395.76941458,  205.7702509],
        [429.41338112,  205.7702509], [463.05734767,  208.92437276],
        [496.70131422,  207.87299881], [529.29390681,  209.97574671],
        [559.78375149,  208.92437276], [592.37634409,  206.82162485],
        [296.94026284,  240.4655914], [330.58422939,  240.4655914],
        [363.17682198,  240.4655914], [396.82078853,  240.4655914],
        [430.46475508,  240.4655914], [460.95459976,  240.4655914],
        [494.59856631,  242.56833931], [528.24253286,  239.41421744],
        [559.78375149,  240.4655914], [593.42771804,  241.51696535],
        [297.9916368,  274.10955795], [328.48148148,  273.05818399],
        [361.07407407,  274.10955795], [395.76941458,  273.05818399],
        [428.36200717,  274.10955795], [463.05734767,  273.05818399],
        [494.59856631,  275.1609319], [526.13978495,  274.10955795],
        [560.83512545,  274.10955795], [591.32497013,  274.10955795],
        [295.88888889,  306.70215054], [329.53285544,  305.65077658],
        [363.17682198,  305.65077658], [393.66666667,  307.75352449],
        [427.31063321,  307.75352449], [459.90322581,  305.65077658],
        [492.4958184,  308.80489845], [527.1911589,  307.75352449],
        [559.78375149,  307.75352449], [590.27359618,  306.70215054]
    ])

    Xymu = Xymu[Xymu.subject == subject]
    implant_type = subjectdata.loc[subject, 'implant_type']
    argus = implant_type(x_center=subjectdata.loc[subject, 'implant_x'],
                         y_center=subjectdata.loc[subject, 'implant_y'],
                         rot=subjectdata.loc[subject, 'implant_rot'])
    is_argus2 = isinstance(implant_type(), p2pi.ArgusII)
    if is_argus2:
        px_argus = px_argus2
        img_argus = img_argus2
    else:
        px_argus = px_argus1
        img_argus = img_argus1

    padding = 2000
    x_range = (p2pr.ret2dva(np.min([e.x_center for e in argus]) - padding),
               p2pr.ret2dva(np.max([e.x_center for e in argus]) + padding))
    y_range = (p2pr.ret2dva(np.min([e.y_center for e in argus]) - padding),
               p2pr.ret2dva(np.max([e.y_center for e in argus]) + padding))
    # (np.diff(y_range)[0] * 10, np.diff(x_range)[0] * 10)
    out_shape = Xymu.img_shape.unique()[0]
    pts_in = []
    pts_dva = []
    pts_out = []
    for xy, e in zip(px_argus, argus):
        pts_in.append(xy)
        dva = p2pr.ret2dva([e.x_center, e.y_center])
        pts_dva.append(dva)
        xout = (dva[0] - x_range[0]) / \
            (x_range[1] - x_range[0]) * (out_shape[1] - 1)
        yout = (dva[1] - y_range[0]) / \
            (y_range[1] - y_range[0]) * (out_shape[0] - 1)
        pts_out.append([xout, yout])
    argus2dva = skit.estimate_transform('similarity', np.array(pts_in),
                                        np.array(pts_dva))
    dva2out = skit.estimate_transform('similarity', np.array(pts_dva),
                                      np.array(pts_out))
    argus2out = skit.estimate_transform('similarity', np.array(pts_in),
                                        np.array(pts_out))

    # top left, top right, bottom left, bottom right
    pts_draw = [[0, 0], [0, out_shape[1] - 1],
                [out_shape[0] - 1, 0], [out_shape[1] - 1, out_shape[0] - 1]]
    x_range = subjectdata.loc[subject, 'xrange']
    y_range = subjectdata.loc[subject, 'yrange']
    pts_dva = [[x_range[0], y_range[0]], [x_range[0], y_range[1]],
               [x_range[1], y_range[0]], [x_range[1], y_range[1]]]
    draw2dva = skit.estimate_transform(
        'similarity', np.array(pts_draw), np.array(pts_dva))

    all_imgs = np.zeros(out_shape)
    for _, row in Xymu.iterrows():
        e_pos = p2pr.ret2dva((argus[row['electrode']].x_center,
                              argus[row['electrode']].y_center))
        align_center = dva2out(e_pos)[0]
        img_drawing = imgproc.scale_phosphene(
            row['image'], subjectdata.loc[subject, 'scale']
        )
        img_drawing = imgproc.center_phosphene(
            img_drawing, center=align_center[::-1]
        )
        all_imgs += img_drawing
    all_imgs = np.maximum(0, np.minimum(1, 1 - all_imgs))
    ax.imshow(skit.warp(img_argus, argus2out.inverse, cval=1.0,
                        output_shape=out_shape), alpha=0.3)
    # ax.imshow(np.ma.masked_where(all_imgs > 0.5, all_imgs), vmin=0, vmax=1,
    #           cmap='gray')
    ax.imshow(all_imgs, vmin=0, vmax=1, cmap='gray', alpha=0.7)

    fovea = fovea = dva2out([0, 0])[0]
    ax.scatter(fovea[0], fovea[1], s=100, marker='s', c='w', edgecolors='k')
