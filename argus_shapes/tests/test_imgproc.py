from __future__ import absolute_import, division, print_function
import numpy as np

import pytest
import numpy.testing as npt

import skimage.measure as skim
import skimage.draw as skid

from .. import imgproc
from .. import utils


def test_get_thresholded_image():
    in_shape = (8, 10)
    img = np.random.rand(np.prod(in_shape)).reshape(in_shape)

    # `img` must be a numpy array
    for invalid_img in [12, [1, 2], [[0]]]:
        with pytest.raises(TypeError):
            imgproc.get_thresholded_image(invalid_img)

    # `thresh` must be an int, float, or a string
    for invalid_thresh in [{'a': 0}, [0, 1], None]:
        with pytest.raises(TypeError):
            imgproc.get_thresholded_image(img, thresh=invalid_thresh)
    # If `thresh` is a string, must be a known method
    for invalid_thresh in ['unknown', 'max']:
        with pytest.raises(ValueError):
            imgproc.get_thresholded_image(img, thresh=invalid_thresh)

    # In general, the method should return a binarized image, except when
    # `thresh` is out of range:
    for th, out, exp_uniq in zip([0.1, 'min', -1, 2],
                                 [in_shape, (16, 10), (20, 21), None],
                                 [[0, 1], [0, 1], 1, 0]):
        th_img = imgproc.get_thresholded_image(img, thresh=th, out_shape=out)
        npt.assert_equal(th_img.shape, in_shape if out is None else out)
        npt.assert_equal(np.unique(th_img.ravel()), exp_uniq)
        npt.assert_equal((th_img.dtype == np.float32 or
                          th_img.dtype == np.float64), True)


def test_get_region_props():
    shape = (24, 48)
    img = np.zeros(shape)
    img[10:15, 20:25] = 1

    for return_all in [True, False]:
        regions = imgproc.get_region_props(img, return_all=return_all)
        npt.assert_equal(regions.area, img.sum())
        npt.assert_almost_equal(regions.orientation, np.pi / 4)


def test_calc_shape_descriptors():
    img_shape = (200, 400)
    shape_center = (100, 200)

    # Make sure function works when no region is found:
    props = imgproc.calc_shape_descriptors(np.zeros(img_shape))
    npt.assert_almost_equal(props['area'], 0)
    npt.assert_almost_equal(props['orientation'], 0)
    npt.assert_almost_equal(props['eccentricity'], 0)
    npt.assert_almost_equal(props['compactness'], 1)
    npt.assert_almost_equal(props['x_center'], img_shape[1] // 2)
    npt.assert_almost_equal(props['y_center'], img_shape[0] // 2)

    # Make sure circles work:
    for radius in [5, 7, 9, 11]:
        circle = np.zeros(img_shape, dtype=float)
        rr, cc = skid.circle(shape_center[0], shape_center[1], radius,
                             shape=img_shape)
        circle[rr, cc] = 1.0
        props = imgproc.calc_shape_descriptors(circle)
        npt.assert_almost_equal(props['orientation'], 0)
        npt.assert_almost_equal(props['eccentricity'], 0)
        npt.assert_almost_equal(props['compactness'], 1)
        npt.assert_almost_equal(props['x_center'], shape_center[1])
        npt.assert_almost_equal(props['y_center'], shape_center[0])

    # Make sure ellipses work:
    for rot in [-0.2, 0, 0.2]:
        for c_radius, r_radius in [(17, 13), (19, 16), (25, 21)]:
            ellipse = np.zeros(img_shape, dtype=float)
            rr, cc = skid.ellipse(shape_center[0], shape_center[1],
                                  r_radius, c_radius,
                                  rotation=rot, shape=img_shape)
            ellipse[rr, cc] = 1.0
            props = imgproc.calc_shape_descriptors(ellipse)
            angle = utils.angle_diff(props['orientation'], rot)
            npt.assert_almost_equal(angle, 0, decimal=1)
            npt.assert_equal(props['eccentricity'] > 0.5, True)
            npt.assert_equal(props['compactness'] > 0.5, True)
            npt.assert_almost_equal(props['x_center'], shape_center[1])
            npt.assert_almost_equal(props['y_center'], shape_center[0])

    # Thin line:
    line = np.zeros(img_shape, dtype=float)
    rr, cc = skid.line(0, 0, img_shape[0] - 1, img_shape[1] - 1)
    line[rr, cc] = 1.0
    props = imgproc.calc_shape_descriptors(line)
    npt.assert_equal(props['area'], img_shape[1])
    npt.assert_almost_equal(props['eccentricity'], 1, decimal=3)
    npt.assert_almost_equal(props['orientation'], -np.arctan2(img_shape[0],
                                                              img_shape[1]),
                            decimal=3)


def test_center_phosphene():
    bright = 13
    img = np.zeros((5, 5))
    img[0, 0] = bright
    center_img = imgproc.center_phosphene(img)
    npt.assert_equal(np.sum(img), np.sum(center_img))
    npt.assert_equal(center_img[2, 2], bright)
    npt.assert_equal(np.sum(np.delete(center_img.ravel(), 12)), 0)


def test_scale_phosphene():
    img = np.zeros((200, 200), dtype=np.double)
    img[90:110, 90:110] = 1
    img_area = skim.moments(img, order=0)
    for scale in [0.9, 1, 1.5, 2, 4]:
        scaled = imgproc.scale_phosphene(img, scale)
        scaled_area = skim.moments(scaled, order=0)
        npt.assert_almost_equal(scaled_area, img_area * scale ** 2)


def test_rotate_phosphene():
    img = np.zeros((200, 200), dtype=np.double)
    img[90:110, 80:130] = 1
    true_props = imgproc.get_region_props(img)
    for rot in [-17, -10, 0, 5, 10, 45]:
        rotated = imgproc.rotate_phosphene(img, rot)
        props = imgproc.get_region_props(rotated)
        npt.assert_almost_equal(true_props.orientation + np.deg2rad(rot),
                                props.orientation,
                                decimal=2)


def test_dice_coeff():
    for img0 in [1, (3, 4), [1, 2, 3], [[1, 2]]]:
        with pytest.raises(TypeError):
            imgproc.dice_coeff(img0, np.zeros((3, 3)))
    for img1 in [1, (3, 4), [1, 2, 3], [[1, 2]]]:
        with pytest.raises(TypeError):
            imgproc.dice_coeff(np.zeros((3, 3)), img1)
    for img0shape in [(3, 4), (4, 4), (5, 5)]:
        with pytest.raises(TypeError):
            imgproc.dice_coeff(np.zeros(img0shape), np.zeros((3, 3)))
    for img1shape in [(3, 4), (4, 4), (5, 5)]:
        with pytest.raises(TypeError):
            imgproc.dice_coeff(np.zeros((3, 3)), np.zeros(img1shape))

    # No overlap gives dice coefficient 0:
    npt.assert_equal(imgproc.dice_coeff(np.zeros((7, 8)), np.ones((7, 8))), 0)
    npt.assert_equal(imgproc.dice_coeff(np.ones((7, 8)), np.zeros((7, 8))), 0)

    # Two identical images give dice coefficient 1:
    npt.assert_almost_equal(imgproc.dice_coeff(np.ones((7, 8)),
                                               np.ones((7, 8))), 1)
    # However, two empty images are silly to compare, by convention give 0:
    npt.assert_equal(imgproc.dice_coeff(np.zeros((7, 8)), np.zeros((7, 8))), 0)

    # A real, simple example:
    img0 = np.ones((9, 10))
    img1 = np.zeros((9, 10))
    img1[4:6, :] = 1
    npt.assert_almost_equal(imgproc.dice_coeff(img1, img1), 1)
    npt.assert_almost_equal(imgproc.dice_coeff(img0, img1), 40 / 110.0)
    npt.assert_almost_equal(imgproc.dice_coeff(img1, img0), 40 / 110.0)
