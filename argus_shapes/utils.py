import numpy as np
import scipy.stats as spst


def ret2dva(r_um):
    """Converts retinal distances (um) to visual angles (deg)

    This function converts an eccentricity measurement on the retinal
    surface(in micrometers), measured from the optic axis, into degrees
    of visual angle.
    Source: Eq. A6 in Watson(2014), J Vis 14(7): 15, 1 - 17
    """
    sign = np.sign(r_um)
    r_mm = 1e-3 * np.abs(r_um)
    r_deg = 3.556 * r_mm + 0.05993 * r_mm ** 2 - 0.007358 * r_mm ** 3
    r_deg += 3.027e-4 * r_mm ** 4
    return sign * r_deg


def dva2ret(r_deg):
    """Converts visual angles (deg) into retinal distances (um)

    This function converts a retinal distancefrom the optic axis(um)
    into degrees of visual angle.
    Source: Eq. A5 in Watson(2014), J Vis 14(7): 15, 1 - 17
    """
    sign = np.sign(r_deg)
    r_deg = np.abs(r_deg)
    r_mm = 0.268 * r_deg + 3.427e-4 * r_deg ** 2 - 8.3309e-6 * r_deg ** 3
    r_um = 1e3 * r_mm
    return sign * r_um


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def angle_diff(angle1, angle2):
    """Returns the signed difference between two angles (rad)

    The difference is calculated as angle2 - angle1. The difference will thus
    be positive if angle2 > angle1.

    Parameters
    ----------
    angle1, angle2 : float
        An angle in radians.

    Returns
    -------
    The signed difference angle2 - angle1 in [0, 2*pi).
    """
    # https://stackoverflow.com/questions/1878907/
    #    the-smallest-difference-between-2-angles
    angle1 = np.asarray(angle1)
    angle2 = np.asarray(angle2)
    return np.arctan2(np.sin(angle2 - angle1), np.cos(angle2 - angle1))


def circfve(r_true, r_pred, lo=0, hi=2 * np.pi):
    """Calculates the fraction of variance explained (FVE) for circular data

    Assumes circular data are in the range [lo, hi].
    Uses SciPy's circular stats functions.

    Parameters
    ----------
    r_true : array_like
        Circular data (ground-truth)
    r_pred : array_like
        Circular data (predicted)
    low : float or int, optional
        Low boundary for circular variance range.  Default is 0.
    high : float or int, optional
        High boundary for circular variance range.  Default is ``2*pi``.
    """
    r_true = np.asarray(r_true)
    r_pred = np.asarray(r_pred)
    r_mu_true = spst.circmean(r_true, low=lo, high=hi)
    var_err = spst.circvar(r_true - r_pred, low=lo, high=hi)
    var_tot = spst.circvar(r_true - r_mu_true, low=lo, high=hi)
    if np.isclose(var_err, 0) and np.isclose(var_tot, 0):
        return 1
    if np.isclose(var_tot, 0):
        return 0
    return (1 - var_err / var_tot)
