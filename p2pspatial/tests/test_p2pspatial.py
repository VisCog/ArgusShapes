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
