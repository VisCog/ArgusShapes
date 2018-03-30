import numpy as np
cimport numpy as np
import cython
from libc.math cimport pow as c_pow

cdef inline int signum(float value) nogil:
    return (value > 0) - (value < 0)

cdef inline float float_max(float a, float b) nogil:
    return a if a >= b else b

ctypedef np.intp_t(*metric_ptr)(double[:, :], double, double)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.intp_t argmin_segment(double[:, :] bundles, double x, double y):
    cdef double dist2, min_dist2
    cdef np.intp_t seg, min_seg, n_seg

    min_dist2 = 1e12

    n_seg = bundles.shape[0]
    for seg in range(n_seg):
        dist2 = c_pow(bundles[seg, 0] - x, 2) + c_pow(bundles[seg, 1] - y, 2)
        if dist2 < min_dist2:
            min_dist2 = dist2
            min_seg = seg
    return min_seg


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_finds_closest_axons(double[:, :] bundles, double[:] xret,
                             double[:] yret):
    cdef np.intp_t[:] closest_seg = np.empty(len(xret), dtype=int)
    cdef np.intp_t n_xy, n_seg
    n_xy = len(xret)
    n_seg = bundles.shape[0]
    for pos in range(n_xy):
        closest_seg[pos] = argmin_segment(bundles, xret[pos], yret[pos])
    return closest_seg
