from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd

import pytest
import numpy.testing as npt

import skimage.measure as skim

from .. import imgproc


def test_get_thresholded_image():
    in_shape = (8, 10)
    img = np.random.rand(np.prod(in_shape)).reshape(in_shape)

    # `img` must be a numpy array
    for invalid_img in [12, [1, 2], [[0]]]:
        with pytest.raises(TypeError):
            imgproc.get_thresholded_image(invalid_img)

    # `thresh` must be an int, float, or a string
    for invalid_thresh in [{'a': 0}, [0, 1]]:
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
                                 [[0, 255], [0, 255], 255, 0]):
        th_img = imgproc.get_thresholded_image(img, thresh=th, out_shape=out)
        npt.assert_equal(th_img.shape, in_shape if out is None else out)
        npt.assert_equal(np.unique(th_img.ravel()), exp_uniq)
        npt.assert_equal(th_img.dtype, np.uint8)


def test_get_region_props():
    shape = (24, 48)
    img = np.zeros(shape)
    img[10:15, 20:25] = 1

    for return_all in [True, False]:
        regions = imgproc.get_region_props(img, return_all=return_all)
        npt.assert_equal(regions.area, img.sum())
        npt.assert_almost_equal(regions.orientation, np.pi / 4)


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


def test_scale_rot_dice_loss():
    # `images` must be a tuple of images or rows in a pandas DataFrame:
    for images in [0, [1, 2], (1, 2), [[3, 4]], ((0, 1), (1, 2))]:
        with pytest.raises(TypeError):
            imgproc.scale_rot_dice_loss(images)
    # Even if a DataFrame is passed, it needs to have a valid 'image' column
    # with an np.ndarray in it:
    X = pd.DataFrame([[1], [2]], columns=['image'])
    with pytest.raises(TypeError):
        imgproc.scale_rot_dice_loss(([0, X], [1, X]))

    # Two identical images have zero loss, except when they are empty:
    img = np.zeros((200, 200), dtype=np.double)
    npt.assert_almost_equal(imgproc.scale_rot_dice_loss([img, img]), 100)
    img[90:110, 90:110] = 1
    npt.assert_almost_equal(imgproc.scale_rot_dice_loss([img, img]), 0)

    # # Scale term should be symmetric around 1: scaling with 0.5 and 2.0 should
    # # give the same error
    # for scale in [1, 2, 4]:
    #     print(scale)
    #     img_small = imgproc.scale_phosphene(img, 1.0 / scale)
    #     err_small = imgproc.scale_rot_dice_loss(
    #         (img, img_small), w_rot=0, w_dice=0)
    #     img_large = imgproc.scale_phosphene(img, scale)
    #     err_large = imgproc.scale_rot_dice_loss(
    #         (img, img_large), w_rot=0, w_dice=0)
    #     npt.assert_almost_equal(err_small, err_large)
