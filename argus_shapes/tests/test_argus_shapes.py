from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import pandas as pd
import numpy.testing as npt
import argus_shapes


def test_ret2dva():
    # Below 15mm eccentricity, relationship is linear with slope 3.731
    npt.assert_equal(argus_shapes.ret2dva(0.0), 0.0)
    for sign in [-1, 1]:
        for exp in [2, 3, 4]:
            ret = sign * 10 ** exp  # mm
            dva = 3.731 * sign * 10 ** (exp - 3)  # dva
            npt.assert_almost_equal(argus_shapes.ret2dva(ret), dva,
                                    decimal=3 - exp)  # adjust precision


def test_dva2ret():
    # Below 50deg eccentricity, relationship is linear with slope 0.268
    npt.assert_equal(argus_shapes.dva2ret(0.0), 0.0)
    for sign in [-1, 1]:
        for exp in [-2, -1, 0]:
            dva = sign * 10 ** exp  # deg
            ret = 0.268 * sign * 10 ** (exp + 3)  # mm
            npt.assert_almost_equal(argus_shapes.dva2ret(dva), ret,
                                    decimal=-exp)  # adjust precision
