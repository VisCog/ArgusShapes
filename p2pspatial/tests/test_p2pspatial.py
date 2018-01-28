from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import pandas as pd
import numpy.testing as npt
import p2pspatial


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

    Xnew, ynew = p2pspatial.average_data(Xold, yold)

    npt.assert_equal(len(np.unique(Xnew['amp'])), n_amps)
    npt.assert_equal(len(np.unique(Xnew['electrode'])), n_amps)
    npt.assert_equal(np.all([c in Xnew.columns for c in Xold.columns]), True)
    npt.assert_equal(np.all([c in ynew.columns for c in yold.columns]), True)
    for _, row in Xnew.iterrows():
        # The mean of 'avgthis' should be 'amp'
        npt.assert_almost_equal(row['avgthis'], row['amp'])
    for _, row in ynew.iterrows():
        npt.assert_almost_equal(row['target1'], row['target2'])


def test_ret2dva():
    # Below 15mm eccentricity, relationship is linear with slope 3.731
    npt.assert_equal(p2pspatial.ret2dva(0.0), 0.0)
    for sign in [-1, 1]:
        for exp in [2, 3, 4]:
            ret = sign * 10 ** exp  # mm
            dva = 3.731 * sign * 10 ** (exp - 3)  # dva
            npt.assert_almost_equal(p2pspatial.ret2dva(ret), dva,
                                    decimal=3 - exp)  # adjust precision


def test_dva2ret():
    # Below 50deg eccentricity, relationship is linear with slope 0.268
    npt.assert_equal(p2pspatial.dva2ret(0.0), 0.0)
    for sign in [-1, 1]:
        for exp in [-2, -1, 0]:
            dva = sign * 10 ** exp  # deg
            ret = 0.268 * sign * 10 ** (exp + 3)  # mm
            npt.assert_almost_equal(p2pspatial.dva2ret(dva), ret,
                                    decimal=-exp)  # adjust precision
