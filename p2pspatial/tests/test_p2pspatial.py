from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import pandas as pd
import numpy.testing as npt
import p2pspatial


def test_get_thresholded_image():
    shape = (34, 88)
    img = np.zeros(shape)
    img[::2, :] = 1.0

    for sh_fact in [0.9, 1, 2.4]:
        th_shape = tuple([int(s * sh_fact) for s in shape])

        for th in [0.05, 0.5, 0.9]:
            print(th, th_shape)
            th_img = p2pspatial.get_thresholded_image(img, thresh=th,
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


def test_region_props():
    shape = (24, 48)
    img = np.zeros(shape)
    img[10:15, 20:25] = 1

    for return_all in [True, False]:
        regions = p2pspatial.get_region_props(img, return_all=return_all)
        npt.assert_equal(regions.area, img.sum())
        npt.assert_almost_equal(regions.orientation, np.pi / 4)


def test_transform_data():
    n_amps = 4
    Xrows = []
    yrows = []
    for _ in range(2):
        for amp in range(n_amps):
            el = 'A%d' % amp
            Xrows.append({'electrode': el, 'amp': amp, 'avgthis': amp})
            yrows.append({'target1': amp, 'target2': amp})
    Xold = pd.DataFrame(Xrows)
    yold = pd.DataFrame(yrows)

    Xnew, ynew = p2pspatial.transform_data(Xold, yold)

    npt.assert_equal(len(np.unique(Xnew['amp'])), n_amps)
    npt.assert_equal(len(np.unique(Xnew['electrode'])), n_amps)
    npt.assert_equal(np.all([c in Xnew.columns for c in Xold.columns]), True)
    npt.assert_equal(np.all([c in ynew.columns for c in yold.columns]), True)
    for _, row in Xnew.iterrows():
        # The mean of 'avgthis' should be 'amp'
        npt.assert_almost_equal(row['avgthis'], row['amp'])
    for _, row in ynew.iterrows():
        npt.assert_almost_equal(row['target1'], row['target2'])
