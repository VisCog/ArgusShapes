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


def calc_shape_descriptors(img, thresh=0.5):
    """Calculates shape descriptors of a grayscale phosphene image"""
    props = get_region_props(img, thresh=thresh)

    if props is None:
        # If no region is found, use dummy values:
        area = 0
        orientation = 0
        eccentricity = 0
        compactness = 1
        centroid = (img.shape[0] // 2, img.shape[1] // 2)
    else:
        # Region is found, calculate calculate descriptors:
        area = props.area
        centroid = props.centroid
        # Avoid division by zero when calculating compactness:
        if np.isclose(props.perimeter, 0):
            # Undefined: Assume tiny circle
            compactness = 1
        else:
            # The most compact shape is a circle, it has compactness 1/(4*pi).
            # All other shapes have smaller values. We therefore multiply with
            # 4*pi to confine the metric to [0, 1], where 1=most compact:
            compactness = 4 * np.pi * area / props.perimeter ** 2
            compactness = np.minimum(1, np.maximum(0, compactness))
        # Calculate orientation and eccentricity:
        if np.isclose(compactness, 1):
            # For circles, orientation is not defined, and eccentricity in
            # skimage is buggy, so manually set to 0:
            orientation = 0
            eccentricity = 0
        else:
            orientation = props.orientation
            eccentricity = props.eccentricity

    descriptors = {'x_center': centroid[1],
                   'y_center': centroid[0],
                   'area': area,
                   'orientation': orientation,
                   'eccentricity': eccentricity,
                   'compactness': compactness}
    return descriptors


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
