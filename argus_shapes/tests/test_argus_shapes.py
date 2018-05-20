from __future__ import absolute_import, division, print_function
import os
import numpy as np
import pandas as pd

import numpy.testing as npt
import pytest

import skimage.io as skio

from .. import argus_shapes


def test_load_subjects():
    pass


def test_load_data():
    with pytest.raises(FileNotFoundError):
        argus_shapes.load_data("doesforsurenotexist.csv")

    csvfile = "data.csv"
    imgfile = "test_image.png"
    skio.imsave(imgfile, np.random.rand(10, 10))

    subjects = ['S1', 'S2']
    electrodes = ['A1', 'F9']
    amps = [2.0, 3.0]
    for use_fullpath in [True, False]:
        data = []
        for subject in subjects:
            for electrode in electrodes:
                for amp in amps:
                    if use_fullpath:
                        fname = os.path.join(os.getcwd(), imgfile)
                    else:
                        fname = imgfile
                    row = {
                        'subject_id': subject,
                        'PTS_ELECTRODE': electrode,
                        'PTS_FILE': fname,
                        'PTS_AMP': amp,
                        'PTS_FREQ': 20.0,
                        'PTS_PULSE_DUR': 0.45,
                        'stim_class': 'SingleElectrode',
                        'date': '1985/09/30'
                    }
                    data.append(row)
        pd.DataFrame(data).to_csv(csvfile, index=False)
        X, _ = argus_shapes.load_data(csvfile)
        npt.assert_equal(np.sort(X.subject.unique()), subjects)
        npt.assert_equal(np.sort(X.electrode.unique()), electrodes)
        npt.assert_equal(len(X), len(subjects) * len(electrodes) * len(amps))

        for subject in subjects + ['nobody', 'S10']:
            X, _ = argus_shapes.load_data(csvfile, subject=subject)
            if subject in subjects:
                npt.assert_equal(np.sort(X.subject.unique()), subject)
                npt.assert_equal(np.sort(X.electrode.unique()), electrodes)
                npt.assert_equal(np.sort(X.amp.unique()), amps)
            else:
                npt.assert_equal(len(X), 0)
                npt.assert_equal(len(X.columns), 0)

        for electrode in electrodes + ['F10']:
            X, _ = argus_shapes.load_data(csvfile, electrodes=[electrode])
            if electrode in electrodes:
                npt.assert_equal(np.sort(X.subject.unique()), subjects)
                npt.assert_equal(np.sort(X.electrode.unique()), electrode)
                npt.assert_equal(np.sort(X.amp.unique()), amps)
            else:
                npt.assert_equal(len(X), 0)
                npt.assert_equal(len(X.columns), 0)

        for amp in amps + [1.5]:
            X, _ = argus_shapes.load_data(csvfile, amp=amp)
            if np.any([np.isclose(a, amp) for a in amps]):
                npt.assert_equal(np.sort(X.subject.unique()), subjects)
                npt.assert_equal(np.sort(X.electrode.unique()), electrodes)
                npt.assert_equal(np.sort(X.amp.unique()), amp)
            else:
                npt.assert_equal(len(X), 0)
                npt.assert_equal(len(X.columns), 0)

        with pytest.raises(ValueError):
            argus_shapes.load_data(csvfile, electrodes='A1')

    os.remove(csvfile)
    os.remove(imgfile)


def test_is_singlestim_dataframe():
    df = pd.DataFrame([
        {'PTS_ELECTRODE': 'A01'},
        {'PTS_ELECTRODE': 'A02'}
    ])
    npt.assert_equal(argus_shapes.is_singlestim_dataframe(df), True)

    df = pd.DataFrame([
        {'PTS_ELECTRODE1': 'A01', 'PTS_ELECTRODE2': 'A03'},
        {'PTS_ELECTRODE1': 'A02', 'PTS_ELECTRODE2': 'A04'}
    ])
    npt.assert_equal(argus_shapes.is_singlestim_dataframe(df), False)
