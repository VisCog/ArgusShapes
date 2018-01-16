import os
import numpy as np
import pandas as pd
import skimage.io as skio
import skimage.filters as skif
import skimage.transform as skit
import skimage.measure as skim


def get_thresholded_image(img, thresh='min', res_shape=None, verbose=True):
    if res_shape is not None:
        img = skit.resize(img, res_shape, mode='reflect')
    if thresh == 'min':
        try:
            thresh = skif.threshold_minimum(img)
        except RuntimeError:
            if verbose:
                print('Runtime error with minimum threshold')
            thresh = (img.max() - img.min()) // 2
    else:
        assert isinstance(thresh, (int, float))
    img_th = img > thresh
    return img_th.astype(np.uint8) * 255


def get_avg_image(X, subject, electrode, amp=None, align_center=None):
    idx = np.logical_and(X['subject'] == subject, X['electrode'] == electrode)
    if amp is not None:
        idx_amp = np.isclose(amp, X['amp'])
        assert np.any(idx_amp)
        idx = np.logical_and(idx, idx_amp)

    avg_img = None
    for _, row in X[idx].iterrows():
        img = skio.imread(os.path.join(row['folder'], row['filename']),
                          as_grey=True)
        if align_center is None:
            # Choose center of image
            img_shape = img.shape[:2]
            align_center = [img_shape[1] // 2, img_shape[0] // 2]

        transl = [align_center[0] - row['centroid'][1],
                  align_center[1] - row['centroid'][0]]
        trafo = skit.EuclideanTransform(translation=transl)
        if avg_img is None:
            avg_img = skit.warp(img, trafo.inverse)
        else:
            avg_img += skit.warp(img, trafo.inverse)
    return avg_img


def get_region_props(img, thresh='min', res_shape=None, return_all=False):
    img = get_thresholded_image(img, thresh=thresh, res_shape=res_shape)
    if img is None:
        return None

    regions = skim.regionprops(img)
    if len(regions) == 0:
        #print('No regions: min=%f max=%f' % (img.min(), img.max()))
        return None
    elif len(regions) == 1:
        return regions[0]
    else:
        if return_all:
            return regions
        else:
            # Multiple props, choose largest
            areas = np.array([r.area for r in regions])
            idx = np.argmax(areas)
            return regions[idx]


def center_phosphene(img_in):
    """Centers a phosphene in an image"""
    # Subtract center of mass from image center
    m = skim.moments(img_in, order=1)
    transl = (img_in.shape[1] // 2 - m[1, 0] / m[0, 0],
              img_in.shape[0] // 2 - m[0, 1] / m[0, 0])
    tf_shift = skit.SimilarityTransform(translation=transl)
    return skit.warp(img_in, tf_shift.inverse)


def scale_phosphene(img, scale):
    """Scales phosphene with a scaling factor"""
    # Shift the phosphene to (0, 0), scale, shift back to the image center
    shift_y, shift_x = np.array(img.shape[:2]) / 2.0
    tf_shift = skit.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_scale = skit.SimilarityTransform(scale=scale)
    tf_shift_inv = skit.SimilarityTransform(translation=[shift_x, shift_y])
    return skit.warp(img, (tf_shift + (tf_scale + tf_shift_inv)).inverse)


def dice_coeff(img0, img1):
    """Compute dice coefficient"""
    img0 = img0 > 0
    img1 = img1 > 0
    return 2 * np.sum(img0 * img1) / (np.sum(img0) + np.sum(img1))


def dice_loss(images, n_angles=73, return_raw=False):
    """Calculate loss function"""
    (_, y_true_row), (_, y_pred_row) = images
    assert isinstance(y_true_row, pd.core.series.Series)
    assert isinstance(y_pred_row, pd.core.series.Series)

    img_true = y_true_row['image']
    img_pred = y_pred_row['image']
    assert isinstance(img_true, np.ndarray)
    assert isinstance(img_pred, np.ndarray)
    if not np.allclose(img_true.shape, img_pred.shape):
        print('img_true:', img_true.shape)
        print('img_pred:', img_pred.shape)
        assert False

    img_true = center_phosphene(img_true)
    img_pred = center_phosphene(img_pred)

    # Scale phosphene in `img_pred` to area of phosphene in `img_truth`
    area_true = skim.moments(img_true, order=0)[0, 0]
    area_pred = skim.moments(img_pred, order=0)[0, 0]
    img_pred = scale_phosphene(img_pred, area_true / area_pred)
    
    # Area loss: Make symmetric around 1, so that a scaling factor of 0.5 and
    # 2 both have the same loss. Bound the error in [0, 10] first, then scale
    # to [0, 1]
    loss_scale = np.maximum(area_true / area_pred, area_pred / area_true) - 1
    loss_scale = np.minimum(10, loss_scale) / 10.0

    # Rotation loss: Rotate the phosphene so that the dice coefficient is
    # maximized. If multiple angles give the same dice coefficient, choose
    # the smallest angle. Scale the loss to [0, 1]
    angles = np.linspace(-180, 180, n_angles)
    dice = [dice_coeff(img_true, skit.rotate(img_pred, r)) for r in angles]
    loss_rot = np.abs(angles[np.isclose(dice, np.max(dice))]).min() / 180.0

    # Dice loss: Turn the dice coefficient into a loss in [0, 1]
    loss_dice = 1 - np.max(dice)

    # Now all terms are in [0, 1], so loss is in [0, 3]
    loss = loss_scale + loss_rot + loss_dice
    if return_raw:
        return loss, loss_scale, loss_rot, loss_dice
    else:
        return loss
