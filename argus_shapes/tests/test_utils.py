from __future__ import absolute_import, division, print_function
import numpy as np
import numpy.testing as npt
from .. import utils


def test_ret2dva():
    # Below 15mm eccentricity, relationship is linear with slope 3.731
    npt.assert_equal(utils.ret2dva(0.0), 0.0)
    for sign in [-1, 1]:
        for exp in [2, 3, 4]:
            ret = sign * 10 ** exp  # mm
            dva = 3.731 * sign * 10 ** (exp - 3)  # dva
            npt.assert_almost_equal(utils.ret2dva(ret), dva,
                                    decimal=3 - exp)  # adjust precision


def test_dva2ret():
    # Below 50deg eccentricity, relationship is linear with slope 0.268
    npt.assert_equal(utils.dva2ret(0.0), 0.0)
    for sign in [-1, 1]:
        for exp in [-2, -1, 0]:
            dva = sign * 10 ** exp  # deg
            ret = 0.268 * sign * 10 ** (exp + 3)  # mm
            npt.assert_almost_equal(utils.dva2ret(dva), ret,
                                    decimal=-exp)  # adjust precision


def test_cart2pol():
    npt.assert_almost_equal(utils.cart2pol(0, 0), (0, 0))
    npt.assert_almost_equal(utils.cart2pol(10, 0), (0, 10))
    npt.assert_almost_equal(utils.cart2pol(3, 4), (np.arctan(4 / 3.0), 5))
    npt.assert_almost_equal(utils.cart2pol(4, 3), (np.arctan(3 / 4.0), 5))


def test_pol2cart():
    npt.assert_almost_equal(utils.pol2cart(0, 0), (0, 0))
    npt.assert_almost_equal(utils.pol2cart(0, 10), (10, 0))
    npt.assert_almost_equal(utils.pol2cart(np.arctan(4 / 3.0), 5), (3, 4))
    npt.assert_almost_equal(utils.pol2cart(np.arctan(3 / 4.0), 5), (4, 3))


def test_angle_diff():
    npt.assert_almost_equal(utils.angle_diff(0, 2 * np.pi), 0)
    npt.assert_almost_equal(utils.angle_diff(0, np.pi / 2), np.pi / 2)
    npt.assert_almost_equal(utils.angle_diff(np.pi / 2, 0), -np.pi / 2)
    npt.assert_almost_equal(utils.angle_diff(0, -np.pi / 2), -np.pi / 2)
    npt.assert_almost_equal(utils.angle_diff(-np.pi / 2, np.pi / 2), np.pi)

    for offset1 in [-2 * np.pi, 0, 2 * np.pi]:
        npt.assert_almost_equal(utils.angle_diff(0.1 + offset1, 0.2), 0.1)
    for offset2 in [-2 * np.pi, 0, 2 * np.pi]:
        npt.assert_almost_equal(utils.angle_diff(0.1, 0.2 + offset2), 0.1)

    for offset1 in [-2 * np.pi, 0, 2 * np.pi]:
        npt.assert_almost_equal(utils.angle_diff(0.2 + offset1, 0.1), -0.1)
    for offset2 in [-2 * np.pi, 0, 2 * np.pi]:
        npt.assert_almost_equal(utils.angle_diff(0.2, 0.1 + offset2), -0.1)
