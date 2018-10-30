from __future__ import absolute_import, division, print_function
import os
import numpy as np
import pandas as pd
import shutil
import requests

import numpy.testing as npt
import pytest

import skimage.io as skio

from .. import argus_shapes as shapes
from .. import imgproc
import pulse2percept.implants as p2pi

try:
    FileNotFoundError
except NameError:
    # Python 2
    FileNotFoundError = IOError


def generate_dummy_data():
    X = pd.DataFrame()
    X['subject'] = pd.Series(['S1', 'S1', 'S2', 'S2', 'S3', 'S3'])
    X['feature1'] = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    X['feature2'] = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    y = pd.DataFrame()
    y['subject'] = pd.Series(['S1', 'S1', 'S2', 'S2', 'S3', 'S3'],
                             index=X.index)
    y['target'] = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                            index=X.index)
    y['image'] = pd.Series([np.random.rand(10, 10)] * 6)
    y['area'] = pd.Series([1, 2, 3, 4, 5, 6])
    return X, y


def test_download_file():
    fname = "test.zip"
    with pytest.raises(requests.exceptions.HTTPError):
        shapes.download_file("https://github.com/VisCog/blah", fname)
    shapes.download_file("https://osf.io/rduj4", fname)
    os.remove(fname)


def test_fetch_data():
    test_dir = "test"
    with pytest.raises(ValueError):
        shapes.fetch_data()
    shapes.fetch_data(save_path=test_dir)
    npt.assert_equal(
        os.path.exists(os.path.join(test_dir, 'argus_shapes.zip')),
        True
    )
    npt.assert_equal(os.path.isdir(os.path.join(test_dir, 'argus_shapes')),
                     True)
    npt.assert_equal(
        os.path.exists(os.path.join(test_dir, 'argus_shapes',
                                    'drawings_single.csv')),
        True
    )
    npt.assert_equal(
        os.path.exists(os.path.join(test_dir, 'argus_shapes', 'subjects.csv')),
        True
    )
    shutil.rmtree(test_dir)


def test_load_data():
    with pytest.raises(FileNotFoundError):
        shapes.load_data("doesforsurenotexist.csv", auto_fetch=False)

    csvfile = "data.csv"
    csvfile2 = "data2.csv"
    imgfile = "test_image.png"
    skio.imsave(imgfile, np.random.randint(256, size=(10, 10)))

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
        X = shapes.load_data(csvfile)
        npt.assert_equal(np.sort(X.subject.unique()), subjects)
        npt.assert_equal(np.sort(X.electrode.unique()), electrodes)
        npt.assert_equal(len(X), len(subjects) * len(electrodes) * len(amps))

        with pytest.raises(ValueError):
            XX = X.copy()
            XX['PTS_ELECTRODE1'] = XX['electrode']
            XX['PTS_ELECTRODE2'] = XX['electrode']
            XX.drop(columns='electrode', inplace=True)
            XX.to_csv(csvfile2, index=False)
            X = shapes.load_data(csvfile2)

        for subject in subjects + ['nobody', 'S10']:
            X = shapes.load_data(csvfile, subject=subject)
            if subject in subjects:
                npt.assert_equal(np.sort(X.subject.unique()), subject)
                npt.assert_equal(np.sort(X.electrode.unique()), electrodes)
                npt.assert_equal(np.sort(X.amp.unique()), amps)
            else:
                npt.assert_equal(len(X), 0)
                npt.assert_equal(len(X.columns), 0)

        for electrode in electrodes + ['F10']:
            X = shapes.load_data(csvfile, electrodes=[electrode])
            if electrode in electrodes:
                npt.assert_equal(np.sort(X.subject.unique()), subjects)
                npt.assert_equal(np.sort(X.electrode.unique()), electrode)
                npt.assert_equal(np.sort(X.amp.unique()), amps)
            else:
                npt.assert_equal(len(X), 0)
                npt.assert_equal(len(X.columns), 0)

        for amp in amps + [1.5]:
            X = shapes.load_data(csvfile, amp=amp)
            if np.any([np.isclose(a, amp) for a in amps]):
                npt.assert_equal(np.sort(X.subject.unique()), subjects)
                npt.assert_equal(np.sort(X.electrode.unique()), electrodes)
                npt.assert_equal(np.sort(X.amp.unique()), amp)
            else:
                npt.assert_equal(len(X), 0)
                npt.assert_equal(len(X.columns), 0)

        with pytest.raises(ValueError):
            shapes.load_data(csvfile, electrodes='A1')

    os.remove(csvfile)
    os.remove(csvfile2)
    os.remove(imgfile)


def test_load_subjects():
    with pytest.raises(FileNotFoundError):
        shapes.load_subjects("forsuredoesntexist.csv", auto_fetch=False)

    csvfile = "data.csv"
    data = [
        {'subject_id': 'S1', 'implant_type_str': 'ArgusI',
         'implant_x': 10, 'implant_y': 20, 'implant_rot': 0.5,
         'xmin': -30, 'xmax': 30, 'ymin': -20, 'ymax': 20,
         'loc_od_x': 15, 'loc_od_y': 2},
        {'subject_id': 'S2', 'implant_type_str': 'ArgusII',
         'implant_x': 20, 'implant_y': 40, 'implant_rot': 1.0,
         'xmin': -60, 'xmax': 60, 'ymin': -30, 'ymax': 30,
         'loc_od_x': 19, 'loc_od_y': 4},
    ]
    pd.DataFrame(data).to_csv(csvfile, index=False)
    X = shapes.load_subjects(csvfile)
    npt.assert_equal(np.sort(X.index.unique()), ['S1', 'S2'])
    print(X.columns)
    npt.assert_equal(X.loc['S1', 'implant_type'], p2pi.ArgusI)
    npt.assert_equal(X.loc['S2', 'implant_type'], p2pi.ArgusII)
    # etc.

    with pytest.raises(ValueError):
        # Missing 'subject_id' index:
        pd.DataFrame([{'subject': 'S1'}]).to_csv(csvfile, index=False)
        X = shapes.load_subjects(csvfile)

    with pytest.raises(ValueError):
        # Other missing columns:
        pd.DataFrame([{'subject_id': 'S1'}]).to_csv(csvfile, index=False)
        X = shapes.load_subjects(csvfile)
    with pytest.raises(ValueError):
        # Wrong implant type:
        data[0]['implant_type_str'] = 'ArgusIII'
        pd.DataFrame(data).to_csv(csvfile, index=False)
        X = shapes.load_subjects(csvfile)
    os.remove(csvfile)


def test_is_singlestim_dataframe():
    with pytest.raises(ValueError):
        shapes.is_singlestim_dataframe(pd.DataFrame())

    df = pd.DataFrame([
        {'PTS_ELECTRODE': 'A01'},
        {'PTS_ELECTRODE': 'A02'}
    ])
    npt.assert_equal(shapes.is_singlestim_dataframe(df), True)

    df = pd.DataFrame([
        {'PTS_ELECTRODE1': 'A01', 'PTS_ELECTRODE2': 'A03'},
        {'PTS_ELECTRODE1': 'A02', 'PTS_ELECTRODE2': 'A04'}
    ])
    npt.assert_equal(shapes.is_singlestim_dataframe(df), False)


def test_calc_mean_images():
    with pytest.raises(ValueError):
        # empty list not allowed
        shapes.calc_mean_images(pd.DataFrame([]), groupby=[])
    with pytest.raises(ValueError):
        # groupby columns not present:
        shapes.calc_mean_images(pd.DataFrame([]))
    with pytest.raises(ValueError):
        # 'image' not in columns:
        shapes.calc_mean_images(pd.DataFrame([{'subject': 'S1'}]),
                                groupby=['subject'])

    X, y = generate_dummy_data()
    Xy = pd.concat((X, y.drop(columns='subject')), axis=1)
    shapes.calc_mean_images(Xy, groupby=['subject'])
