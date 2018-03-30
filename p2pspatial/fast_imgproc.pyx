import numpy as np
cimport numpy as np
import cython


def fast_dice_coeff(double[:, ::1] image0, double[:, ::1] image1):
    cdef np.intp_t i, j
    cdef double sum0, sum1, sum01
    sum0 = 0.0
    sum1 = 0.0
    sum01 = 0.0
    for i in range(image0.shape[0]):
        for j in range(image0.shape[1]):
            sum0 += image0[i, j] > 0.5
            sum1 += image1[i, j] > 0.5
            sum01 += (image0[i, j] > 0.5) * (image1[i, j] > 0.5)
    return 2.0 * sum01 / (sum0 + sum1 + 1e-12)
