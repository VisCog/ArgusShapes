import numpy as np
cimport numpy as np
import cython
from libc.math cimport pow as c_pow

cdef extern from "math.h":
    cpdef float expf(float x)


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
    return np.asarray(closest_seg)


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_axon_contribution(double[:, :] bundle, double[:] xy, double axlambda):
    cdef np.intp_t p, c, argmin, n_seg
    cdef double dist2
    cdef double[:, :] contrib

    # Find the segment that is closest to the soma `xy`:
    argmin = argmin_segment(bundle, xy[0], xy[1])

    # Add the exact location of the soma:
    bundle[argmin + 1, 0] = xy[0]
    bundle[argmin + 1, 1] = xy[1]

    # For every axon segment, calculate distance from soma by summing up the
    # individual distances between neighboring axon segments
    # (by "walking along the axon"):
    n_seg = argmin + 1
    contrib = np.zeros((n_seg, 3))
    dist2 = 0
    c = 0
    for p in range(argmin, -1, -1):
        dist2 += (c_pow(bundle[p, 0] - bundle[p + 1, 0], 2) +
                  c_pow(bundle[p, 1] - bundle[p + 1, 1], 2))
        contrib[c, 0] = bundle[p, 0]
        contrib[c, 1] = bundle[p, 1]
        contrib[c, 2] = expf(-dist2 / (2.0 * c_pow(axlambda, 2)))
        c += 1
    return np.asarray(contrib)
