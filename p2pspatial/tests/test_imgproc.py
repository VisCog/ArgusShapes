from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import pandas as pd
import numpy.testing as npt
from .. import imgproc


def test_get_thresholded_image():
    shape = (34, 88)
    img = np.zeros(shape)
    img[::2, :] = 1.0

    for sh_fact in [0.9, 1, 2.4]:
        th_shape = tuple([int(s * sh_fact) for s in shape])

        for th in [0.05, 0.5, 0.9]:
            print(th, th_shape)
            th_img = imgproc.get_thresholded_image(img, thresh=th,
                                                   res_shape=th_shape,
                                                   verbose=False)
            print(th_img)
            print(img)
            print('')
            npt.assert_equal(th_img.shape, th_shape)
            npt.assert_almost_equal(th_img.min(), 0)
            npt.assert_almost_equal(th_img.max(), 255)
            if th_shape == shape:
                npt.assert_almost_equal(img * 255, th_img)


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


def test_scale_rot_dice_loss():
    pass
