import numpy as np
import pandas as pd
import six
import os.path

import skimage
import skimage.io as skio
import skimage.filters as skif
import skimage.transform as skit
import skimage.measure as skim

from . import fast_imgproc as fi


def get_thresholded_image(img, thresh=0.5, out_shape=None, verbose=True):
    """Thresholds an image

    Parameters
    ----------
    image : ndarray
        Input image.
    thresh : int, float, or string
        Threshold value
    out_shape : tuple, list, or ndarray
        Size of the generated output image (rows, cols[, ...][, dim]). If 'dim'
        is not provided, the number of channels is preserved. In case the
        number of input channels does not equal the number of output channels
        a n-dimensional interpolation is applied.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("'img' must be a np.ndarray.")

    # Rescale the image if `out_shape` is given
    if out_shape is not None:
        if not isinstance(out_shape, (tuple, list, np.ndarray)):
            raise TypeError("'out_shape' must be a tuple, list, or np.ndarray")
        img = skit.resize(img, out_shape, mode='reflect')

    if not isinstance(thresh, (int, float, six.string_types)):
        raise TypeError(("'thresh' must be an int, float, a string, or None; "
                         "not %s." % type(thresh)))

    # Find the numerical threshold to apply, either using a known scikit-image
    # method, or by applying the provided int, float directly:
    if isinstance(thresh, six.string_types):
        methods = {'otsu': skif.threshold_otsu,
                   'li': skif.threshold_li,
                   'min': skif.threshold_minimum,
                   'mean': skif.threshold_mean}
        if thresh not in methods:
            raise ValueError(("Unknown thresholding method '%s'. Choose from: "
                              "'%s'." % (thresh, "', '".join(methods.keys()))))
        try:
            th = methods[thresh](img)
        except RuntimeError:
            # If fancy thresholding fails, drop back to simple method
            th = (img.max() - img.min()) / 2.0
            if verbose:
                print(("Runtime error with %s, choose thresh=%f "
                       "instead." % (methods[thresh], th)))
    else:
        # Directly apply the provided int, float
        th = thresh

    # Apply threshold and convert image to [0, 1]
    return skimage.img_as_float(img > th)


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


def get_region_props(img, thresh=0.5, out_shape=None, return_all=False):
    if thresh is not None:
        img = get_thresholded_image(img, thresh=thresh, out_shape=out_shape)
    if img is None:
        return None

    regions = skim.regionprops(img.astype(np.int32))
    if len(regions) == 0:
        # print('No regions: min=%f max=%f' % (img.min(), img.max()))
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


def center_phosphene(img, center=None):
    """Centers a phosphene in an image"""
    # Subtract center of mass from image center
    m = skim.moments(img, order=1)

    # No area found:
    if np.isclose(m[0, 0], 0):
        return img

    if center is None:
        center = (img.shape[0] // 2, img.shape[1] // 2)

    # Valid image: shift the image by -centroid, +image center
    transl = (center[1] - m[1, 0] / m[0, 0],
              center[0] - m[0, 1] / m[0, 0])
    tf_shift = skit.SimilarityTransform(translation=transl)
    return skit.warp(img, tf_shift.inverse)


def scale_phosphene(img, scale):
    """Scales phosphene with a scaling factor"""
    m = skim.moments(img, order=1)

    # No area found:
    if np.isclose(m[0, 0], 0):
        return img

    # Shift the phosphene to (0, 0):
    transl = np.array([-m[1, 0] / m[0, 0], -m[0, 1] / m[0, 0]])
    tf_shift = skit.SimilarityTransform(translation=transl)
    # Scale the phosphene:
    tf_scale = skit.SimilarityTransform(scale=scale)
    # Shift the phosphene back to where it was:
    tf_shift_inv = skit.SimilarityTransform(translation=-transl)
    return skit.warp(img, (tf_shift + (tf_scale + tf_shift_inv)).inverse)


def rotate_phosphene(img, rot_deg):
    """Rotates phosphene by an angle counter-clock-wise (deg) """
    m = skim.moments(img, order=1)

    # No area found:
    if np.isclose(m[0, 0], 0):
        return img

    # Shift the phosphene to (0, 0):
    transl = np.array([-m[1, 0] / m[0, 0], -m[0, 1] / m[0, 0]])
    tf_shift = skit.SimilarityTransform(translation=transl)
    # Rotate the phosphene:
    tf_rot = skit.SimilarityTransform(rotation=np.deg2rad(-rot_deg))
    # Shift the phosphene back to where it was:
    tf_shift_inv = skit.SimilarityTransform(translation=-transl)
    return skit.warp(img, (tf_shift + (tf_rot + tf_shift_inv)).inverse)


def dice_coeff(image0, image1):
    """Computes dice coefficient

    Parameters
    ----------
    image0: np.ndarray
        First image
    image1: np.ndarray
        Second image

    Notes
    -----
    Two empty images give dice coefficient 0.
    """
    if not isinstance(image0, np.ndarray):
        raise TypeError("'image0' must be of type np.ndarray.")
    if not isinstance(image1, np.ndarray):
        raise TypeError("'image1' must be of type np.ndarray.")
    if not np.all(image0.shape == image1.shape):
        raise ValueError(("'image0' and 'image1' must have the same shape "
                          "(%s) vs. (%s)" % (", ".join(image0.shape),
                                             ", ".join(image1.shape))))
    return fi.fast_dice_coeff(image0, image1)


def srd_loss(images, n_angles=37, w_scale=33, w_rot=34, w_dice=33,
             return_raw=False):
    """Calculates new loss function"""
    # Unpack `images` into two images, which can be a bit of pain: Most common
    # used case is when zip(y_true.iterrows(), y_pred.iterrows()) is passed
    type_msg = ("'images' must be a tuple of either two images or two rows in "
                "a pandas DataFrame with an 'image' column.")
    if not isinstance(images, (list, tuple)):
        raise TypeError(type_msg)
    imgs = []
    for item in images[:2]:
        if isinstance(item, np.ndarray):
            # `item` is an image, make sure it's of dtype double, otherwise
            # the moment function is off
            imgs.append(skimage.img_as_float(item))
        elif isinstance(item, (list, tuple)):
            # `item` is probably a row of a pandas DataFrame
            _, row = item
            if not isinstance(row, pd.core.series.Series):
                raise TypeError(type_msg)
            if (not hasattr(row, 'image') or hasattr(row, 'image') and
                    not isinstance(row['image'], np.ndarray)):
                raise TypeError(type_msg)
            # Make sure image is saved with dtype double, otherwise the moment
            # function is off
            imgs.append(skimage.img_as_float(row['image']))
        else:
            raise TypeError(type_msg)

    img_true, img_pred = imgs
    if not np.allclose(img_true.shape, img_pred.shape):
        raise ValueError(("Both images must have the same shape, img_true=(%s)"
                          " img_pred=(%s)" % (", ".join(img_true.shape),
                                              ", ".join(img_pred.shape))))
    if img_true.dtype != img_pred.dtype:
        raise ValueError(("Both images must have the same dtype, img_true=%s "
                          "img_pred=%s" % (img_true.dtype, img_pred.dtype)))

    # Center the phosphenes in the image:
    img_true = center_phosphene(img_true)
    img_pred = center_phosphene(img_pred)

    # Scale phosphene in `img_pred` to area of phosphene in `img_truth`
    area_true = skim.moments(img_true, order=0)[0, 0]
    area_pred = skim.moments(img_pred, order=0)[0, 0]
    if np.isclose(area_true, 0) or np.isclose(area_pred, 0):
        # If one of the images is empty, the following analysis is not
        # meaningful, and we simply return the max loss:
        return w_scale + w_rot + w_dice

    img_scale = np.sqrt(area_true / area_pred)
    img_pred = scale_phosphene(img_pred, img_scale)

    # Area loss: Make symmetric around 1, so that a scaling factor of 0.5 and
    # 2 both have the same loss. Bound the error in [0, 10] first, then scale
    # to [0, 1]
    max_scale = 10.0
    loss_scale = np.maximum(img_scale, 1.0 / img_scale) - 1
    loss_scale = np.minimum(max_scale, loss_scale) / max_scale

    # Rotation loss: Rotate the phosphene so that the dice coefficient is
    # maximized (using bi-cubic interpolation):
    max_rot = 180.0
    angles = np.linspace(-180, 180, n_angles)
    dice = [dice_coeff(img_true, rotate_phosphene(img_pred, r))
            for r in angles]
    # If multiple angles give the same dice coefficient, choose the smallest
    # angle. Scale the loss to [0, 1]:
    img_angle = angles[np.isclose(dice, np.max(dice))]
    loss_rot = np.abs(img_angle).min() / max_rot

    # Dice loss: Turn the dice coefficient into a loss in [0, 1]
    loss_dice = 1 - np.max(dice)

    # Now all terms are in [0, 1], combine with weights (by default, loss is
    # in [0, 100]):
    loss = w_scale * loss_scale + w_rot * loss_rot + w_dice * loss_dice
    if return_raw:
        loss_terms = {'loss_scale': loss_scale, 'loss_rot': loss_rot,
                      'loss_dice': loss_dice}
        params = {'scale': img_scale, 'angle': img_angle, 'dice': np.max(dice)}
        return loss, loss_terms, params
    else:
        return loss
