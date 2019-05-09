from __future__ import absolute_import, division, print_function

from . import imgproc

import numpy as np
import scipy.stats as spst
import skimage.io as skio
import skimage.transform as skit

import pulse2percept.implants as p2pi
import pulse2percept.retina as p2pr
import pulse2percept.utils as p2pu

from matplotlib import patches
import os.path as osp
import pkg_resources
data_path = pkg_resources.resource_filename('argus_shapes', 'data/')


def scatter_correlation(xvals, yvals, ax, xticks=[], yticks=[], marker='o',
                        color='k', textloc='top right'):
    """Scatter plots some data points and fits a regression curve to them

    Parameters
    ----------
    xvals, yvals : list
        x, y coordinates of data points to scatter
    ax : axis
        Matplotlib axis
    xticks, yticks : list
        List of ticks on the x, y axes
    marker : str
        Matplotlib marker style
    color : str
        Matplotlib marker color
    textloc : str
        Location of regression result, top/bottom left/right
    """
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
    def fit(x):
        return slope * x + intercept
    slope, intercept, rval, pval, _ = spst.linregress(xvals, yvals)
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
                va=va, ha=ha)
    else:
        ax.text(xt, yt,
                "$N$=%d\n$r$=%.3f\n$p$=%.2e" % (len(yvals), rval, pval),
                va=va, ha=ha)


def plot_phosphenes_on_array(ax, subject, Xymu, subjectdata, alpha_bg=0.5,
                             thresh_fg=0.95, show_fovea=True):
    """Plots phosphenes centered over the corresponding electrodes

    Parameters
    ----------
    ax : axis
        Matplotlib axis
    subject : str
        Subject ID, must be a valid value for column 'subject' in `Xymu` and
        `subjectdata`.
    Xymu : pd.DataFrame
        DataFrame with required columns 'electrode', 'image'. May contain data
        from more than one subject, in which case a column 'subject' must
        exist. May also have a column 'img_shape' indicating the shape of each
        phosphene image.
    subjectdata : pd.DataFrame
        DataFrame with Subject ID as index. Must have columns 'implant_x',
        'implant_y', 'implant_rot', 'implant_type', and 'eye'. May also have a
        column 'scale' containing a scaling factor applied to phosphene size.
    alpha_bg : float
        Alpha value for the array in the background
    thresh_fg : float
        Grayscale value above which to mask the drawings
    show_fovea : bool
        Whether to indicate the location of the fovea with a square

    """
    for col in ['electrode', 'image']:
        if col not in Xymu.columns:
            raise ValueError('Xymu must contain column "%s".' % col)
    # If subject column not present, choose all entries:
    if 'subject' in Xymu.columns:
        Xymu = Xymu[Xymu.subject == subject]
    for col in ['implant_x', 'implant_y', 'implant_rot', 'implant_type',
                'eye']:
        if col not in subjectdata.columns:
            raise ValueError('subjectdata must contain column "%s".' % col)
    if subject not in subjectdata.index:
        raise ValueError('Subject "%s" not an index in subjectdata.' % subject)
    if 'scale' not in subjectdata.columns:
        print("'scale' not in subjectdata, setting scale=1.0")
        subjectdata['scale'] = 1.0

    eye = subjectdata.loc[subject, 'eye']
    # Schematic of the array:
    img_argus1 = skio.imread(osp.join(data_path, 'argus_i.png'))
    img_argus2 = skio.imread(osp.join(data_path, 'argus_ii.png'))
    # Pixel locations of electrodes (Argus I: A1-4, B1-4, ...; Argus II: A1-10,
    # B1-10, ...) in the above images:
    px_argus1 = np.array([
        [163.12857037, 92.32202802], [208.00952276, 93.7029804],
        [248.74761799, 93.01250421], [297.77142752, 91.63155183],
        [163.12857037, 138.58393279], [213.53333228, 137.8934566],
        [252.89047514, 137.2029804], [297.77142752, 136.51250421],
        [163.12857037, 181.3934566], [207.31904657, 181.3934566],
        [250.81904657, 181.3934566], [297.08095133, 181.3934566],
        [163.81904657, 226.27440898], [210.08095133, 226.27440898],
        [252.89047514, 227.65536136], [297.08095133, 227.65536136]
    ])
    px_argus2 = np.array([
        [296.94026284, 140.58506571], [328.48148148, 138.4823178],
        [365.27956989, 140.58506571], [397.87216249, 139.53369176],
        [429.41338112, 138.4823178], [463.05734767, 140.58506571],
        [495.64994026, 139.53369176], [528.24253286, 139.53369176],
        [560.83512545, 139.53369176], [593.42771804, 138.4823178],
        [296.94026284, 173.1776583], [329.53285544, 174.22903226],
        [363.17682198, 173.1776583], [396.82078853, 173.1776583],
        [430.46475508, 173.1776583], [463.05734767, 174.22903226],
        [494.59856631, 173.1776583], [529.29390681, 174.22903226],
        [559.78375149, 175.28040621], [593.42771804, 173.1776583],
        [296.94026284, 206.82162485], [329.53285544, 206.82162485],
        [363.17682198, 205.7702509], [395.76941458, 205.7702509],
        [429.41338112, 205.7702509], [463.05734767, 208.92437276],
        [496.70131422, 207.87299881], [529.29390681, 209.97574671],
        [559.78375149, 208.92437276], [592.37634409, 206.82162485],
        [296.94026284, 240.4655914], [330.58422939, 240.4655914],
        [363.17682198, 240.4655914], [396.82078853, 240.4655914],
        [430.46475508, 240.4655914], [460.95459976, 240.4655914],
        [494.59856631, 242.56833931], [528.24253286, 239.41421744],
        [559.78375149, 240.4655914], [593.42771804, 241.51696535],
        [297.9916368, 274.10955795], [328.48148148, 273.05818399],
        [361.07407407, 274.10955795], [395.76941458, 273.05818399],
        [428.36200717, 274.10955795], [463.05734767, 273.05818399],
        [494.59856631, 275.1609319], [526.13978495, 274.10955795],
        [560.83512545, 274.10955795], [591.32497013, 274.10955795],
        [295.88888889, 306.70215054], [329.53285544, 305.65077658],
        [363.17682198, 305.65077658], [393.66666667, 307.75352449],
        [427.31063321, 307.75352449], [459.90322581, 305.65077658],
        [492.4958184, 308.80489845], [527.1911589, 307.75352449],
        [559.78375149, 307.75352449], [590.27359618, 306.70215054]
    ])
    # Choose the appropriate image / electrode locations based on implant type:
    implant_type = subjectdata.loc[subject, 'implant_type']
    is_argus2 = isinstance(implant_type(), p2pi.ArgusII)
    if is_argus2:
        px_argus = px_argus2
        img_argus = img_argus2
    else:
        px_argus = px_argus1
        img_argus = img_argus1

    # To simulate an implant in a left eye, flip the image left-right (along
    # with the electrode x-coordinates):
    if eye == 'LE':
        img_argus = np.fliplr(img_argus)
        px_argus[:, 0] = img_argus.shape[1] - px_argus[:, 0] - 1

    # Create an instance of the array using p2p:
    argus = implant_type(x_center=subjectdata.loc[subject, 'implant_x'],
                         y_center=subjectdata.loc[subject, 'implant_y'],
                         rot=subjectdata.loc[subject, 'implant_rot'],
                         eye=eye)

    # Add some padding to the output image so the array is not cut off:
    padding = 2000  # microns
    x_range = (p2pr.ret2dva(np.min([e.x_center for e in argus]) - padding),
               p2pr.ret2dva(np.max([e.x_center for e in argus]) + padding))
    y_range = (p2pr.ret2dva(np.min([e.y_center for e in argus]) - padding),
               p2pr.ret2dva(np.max([e.y_center for e in argus]) + padding))

    # If img_shape column not present, choose shape of first entry:
    if 'img_shape' in Xymu.columns:
        out_shape = Xymu.img_shape.unique()[0]
    else:
        out_shape = Xymu.image.values[0].shape

    # Coordinate transform from degrees of visual angle to output, and from
    # image coordinates to output image:
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
    dva2out = skit.estimate_transform('similarity', np.array(pts_dva),
                                      np.array(pts_out))
    argus2out = skit.estimate_transform('similarity', np.array(pts_in),
                                        np.array(pts_out))

    # Top left, top right, bottom left, bottom right:
    x_range = subjectdata.loc[subject, 'xrange']
    y_range = subjectdata.loc[subject, 'yrange']
    pts_dva = [[x_range[0], y_range[0]], [x_range[0], y_range[1]],
               [x_range[1], y_range[0]], [x_range[1], y_range[1]]]

    # Calculate average drawings, but don't binarize:
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
    all_imgs = 1 - np.minimum(1, np.maximum(0, all_imgs))

    # Draw array schematic with specific alpha level:
    img_arr = skit.warp(img_argus, argus2out.inverse, cval=1.0,
                        output_shape=out_shape)
    img_arr[:, :, 3] = alpha_bg

    # Replace pixels where drawings are dark enough, set alpha=1:
    rr, cc = np.unravel_index(np.where(all_imgs.ravel() < thresh_fg)[0],
                              all_imgs.shape)
    for channel in range(3):
        img_arr[rr, cc, channel] = all_imgs[rr, cc]
    img_arr[rr, cc, 3] = 1

    ax.imshow(img_arr, cmap='gray')

    if show_fovea:
        fovea = fovea = dva2out([0, 0])[0]
        ax.scatter(fovea[0], fovea[1], s=100,
                   marker='s', c='w', edgecolors='k')


def plot_fundus(ax, subject, subjectdata, n_bundles=100, upside_down=False,
                annot_array=True, annot_quadr=True):
    """Plot an implant on top of the axon map

    This function plots an electrode array on top of the axon map, akin to a
    fundus photograph. Implant location should be given via `subjectdata`.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot, optional, default: None
        A Matplotlib axes object. If None given, a new one will be created.
    subject : str
        Subject ID, must be a valid value for column 'subject' in
        `subjectdata`.
    subjectdata : pd.DataFrame
        DataFrame with Subject ID as index. Must have columns 'implant_x',
        'implant_y', 'implant_rot', 'implant_type', 'eye', 'loc_od_x',
        'loc_od_y'.
    n_bundles : int, optional, default: 100
        Number of nerve fiber bundles to plot.
    upside_down : bool, optional, default: False
        Flag whether to plot the retina upside-down, such that the upper
        half of the plot corresponds to the upper visual field. In general,
        inferior retina == upper visual field (and superior == lower).
    annot_array : bool, optional, default: True
        Flag whether to label electrodes and the tack.
    annot_quadr : bool, optional, default: True
        Flag whether to annotate the four retinal quadrants
        (inferior/superior x temporal/nasal).

    """
    for col in ['implant_x', 'implant_y', 'implant_rot', 'implant_type',
                'eye', 'loc_od_x', 'loc_od_y']:
        if col not in subjectdata.columns:
            raise ValueError('subjectdata must contain column "%s".' % col)
    if subject not in subjectdata.index:
        raise ValueError('Subject "%s" not an index in subjectdata.' % subject)
    if n_bundles < 1:
        raise ValueError('Number of nerve fiber bundles must be >= 1.')

    # Choose the appropriate image / electrode locations based on implant type:
    implant_type = subjectdata.loc[subject, 'implant_type']
    implant = implant_type(x_center=subjectdata.loc[subject, 'implant_x'],
                           y_center=subjectdata.loc[subject, 'implant_y'],
                           rot=subjectdata.loc[subject, 'implant_rot'],
                           eye=subjectdata.loc[subject, 'eye'])
    loc_od = tuple(subjectdata.loc[subject, ['loc_od_x', 'loc_od_y']])

    phi_range = (-180.0, 180.0)
    n_rho = 801
    rho_range = (2.0, 45.0)

    # Make sure x-coord of optic disc has the correct sign for LE/RE:
    if (implant.eye == 'RE' and loc_od[0] <= 0 or
            implant.eye == 'LE' and loc_od[0] > 0):
        logstr = ("For eye==%s, expected opposite sign of x-coordinate of "
                  "the optic disc; changing %.2f to %.2f" % (implant.eye,
                                                             loc_od[0],
                                                             -loc_od[0]))
        print(logstr)
        loc_od = (-loc_od[0], loc_od[1])
    if ax is None:
        # No axes object given: create
        fig, ax = plt.subplots(1, figsize=(10, 8))
    else:
        fig = ax.figure

    # Matplotlib<2 compatibility
    if hasattr(ax, 'set_facecolor'):
        ax.set_facecolor('black')
    elif hasattr(ax, 'set_axis_bgcolor'):
        ax.set_axis_bgcolor('black')

    # Draw axon pathways:
    phi = np.linspace(*phi_range, num=n_bundles)
    func_kwargs = {'n_rho': n_rho, 'loc_od': loc_od,
                   'rho_range': rho_range, 'eye': implant.eye}
    axon_bundles = p2pu.parfor(p2pr.jansonius2009, phi,
                               func_kwargs=func_kwargs)
    for bundle in axon_bundles:
        ax.plot(p2pr.dva2ret(bundle[:, 0]), p2pr.dva2ret(bundle[:, 1]),
                c=(0.5, 1.0, 0.5))

    # Plot all electrodes and label them (optional):
    for e in implant.electrodes:
        if annot_array:
            ax.text(e.x_center + 100, e.y_center + 50, e.name,
                    color='white', size='x-large')
        ax.plot(e.x_center, e.y_center, 'ow', markersize=np.sqrt(e.radius))

    # Plot the location of the array's tack and annotate it (optional):
    if implant.tack:
        tx, ty = implant.tack
        ax.plot(tx, ty, 'ow')
        if annot_array:
            if upside_down:
                offset = 100
            else:
                offset = -100
            ax.text(tx, ty + offset, 'tack',
                    horizontalalignment='center',
                    verticalalignment='top',
                    color='white', size='large')

    # Show circular optic disc:
    ax.add_patch(patches.Circle(p2pr.dva2ret(loc_od), radius=900, alpha=1,
                                color='black', zorder=10))

    xmin, xmax, ymin, ymax = p2pr.dva2ret([-20, 20, -15, 15])
    ax.set_aspect('equal')
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel('x (microns)')
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('y (microns)')
    eyestr = {'LE': 'left', 'RE': 'right'}
    ax.set_title('%s in %s eye' % (implant, eyestr[implant.eye]))
    ax.grid('off')

    # Annotate the four retinal quadrants near the corners of the plot:
    # superior/inferior x temporal/nasal
    if annot_quadr:
        if upside_down:
            topbottom = ['bottom', 'top']
        else:
            topbottom = ['top', 'bottom']
        if implant.eye == 'RE':
            temporalnasal = ['temporal', 'nasal']
        else:
            temporalnasal = ['nasal', 'temporal']
        for yy, valign, si in zip([ymax, ymin], topbottom,
                                  ['superior', 'inferior']):
            for xx, halign, tn in zip([xmin, xmax], ['left', 'right'],
                                      temporalnasal):
                ax.text(xx, yy, si + ' ' + tn,
                        color='black', fontsize=14,
                        horizontalalignment=halign,
                        verticalalignment=valign,
                        backgroundcolor=(1, 1, 1, 0.8))

    # Need to flip y axis to have upper half == upper visual field
    if upside_down:
        ax.invert_yaxis()

    return fig, ax


def plot_box(vals1, vals2, ax, is_signif=None):
    ax.boxplot([vals1, vals2], widths=0.7)
    x1, x2 = 1, 2
    y, h, col = np.maximum(np.max(vals1), np.max(vals2)) + 10, 5, 'k'
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1, c=col)

    if is_signif is not None:
        if is_signif:
            txt = '*'
        else:
            txt = 'n.s.'
        ax.text((x1 + x2) * .5, y + h, txt, ha='center', va='bottom',
                color=col, fontsize=14)
