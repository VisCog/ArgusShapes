from __future__ import absolute_import, division, print_function
import os
import six
import copy
import glob
import logging

import numpy as np
import pandas as pd

import scipy.interpolate as spi
import scipy.stats as sps

import pulse2percept as p2p

import skimage
import skimage.io as skio
import skimage.filters as skif
import skimage.transform as skit
import skimage.morphology as skimo
import skimage.measure as skime

import sklearn.base as sklb
import sklearn.metrics as sklm
import sklearn.utils as sklu

from .due import due, Doi
from . import imgproc

p2p.console.setLevel(logging.ERROR)

__all__ = ["load_data_raw", "load_data", "load_subjects", "calc_mean_images"]


# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi("10.1167/13.9.30"),
         description="Template project for small scientific Python projects",
         tags=["reference-implementation"],
         path='argus_shapes')


def _loads_data_row_a16(row):
    # Electrodes have legacy names in datafiles:
    old_names = names = ['L6', 'L2', 'M8', 'M4', 'L5', 'L1', 'M7', 'M3',
                         'L8', 'L4', 'M6', 'M2', 'L7', 'L3', 'M5', 'M1']
    # In newer papers, they go by A-D: A1, B1, C1, D1, A1, B2, ..., D4
    # Shortcut: Use `chr` to go from int to char
    new_names = [chr(i) + str(j) for j in range(1, 5)
                 for i in range(65, 69)]
    electrodes = row['electrode'].split('_')
    electrodes = [new_names[old_names.index(e)] for e in electrodes]
    electrodes = '_'.join(electrodes)

    # Date in folder name should match date in filename:
    date = row['exp_file'].split('_')[1].replace(".xls", "")
    if date != os.path.basename(row['exp_folder']):
        raise ValueError(("Inconsistent date: %s vs "
                          "%s.") % (date,
                                    os.path.basename(row['exp_folder'])))

    # Subject is in path name:
    subject = os.path.basename(os.path.dirname(row['exp_folder']))

    # Amp is followed by '_' as in '1.5_'. Multiple amps are like '1.5_2_':
    amp = [float(a) for a in row['amplitude'].split('_') if a != '']
    if len(amp) == 1:
        stim_class = "SingleElectrode"
        amp = amp[0]
    else:
        stim_class = "MultiElectrode"

    params = row['notes'].split('_')
    freq = params[4]
    if freq[:5] == "Freq:" and freq[-2:] == "Hz":
        freq = float(freq[5:-2])
    else:
        freq = 0.0

    # Assemble all feature values in a dict
    feat = {'filename': row['filename'],
            'folder': os.path.join(row['exp_folder'], row['foldername']),
            'subject': subject,
            'electrode': electrodes,
            'param_str': row['notes'],
            'stim_class': stim_class,
            'amp': amp,
            'freq': freq,
            'date': date}
    return feat


def _loads_data_row_a60(row):
    # Split the data strings to extract subject, electrode, etc.
    fname = row['Filename']
    date = fname.split('_')[0]
    params = row['Params'].split(' ')
    stim = params[0].split('_')
    if len(params) < 2 or len(stim) < 2:
        return None

    # Find the current amplitude in the folder name
    # It could have any of the following formats: '/3xTh', '_2.5xTh',
    # ' 2xTh'. Idea: Find the string 'xTh', then walk backwards to
    # find the last occurrence of '_', ' ', or '/'
    idx_end = row['exp_folder'].find('xTh')
    if idx_end == -1:
        return None
    idx_start = np.max([row['exp_folder'].rfind('_', 0, idx_end),
                        row['exp_folder'].rfind(' ', 0, idx_end),
                        row['exp_folder'].rfind(os.sep, 0, idx_end)])
    if idx_start == -1:
        return None
    amp = float(row['exp_folder'][idx_start + 1:idx_end])

    freq = stim[2]
    if freq[0] == 'f':
        freq = float(freq[1:])
    else:
        freq = 0

    # Assemble all feature values in a dict
    feat = {'filename': fname,
            'folder': row['exp_folder'],
            'subject': stim[0],
            'param_str': row['Params'],
            'electrode': params[1],
            'stim_class': stim[1],
            'amp': amp,
            'freq': freq,
            'date': date}
    return feat


def _loads_data_row(df_row, subject, electrodes, amplitude, frequency, date):
    _, row = df_row

    if np.all([c in row for c in ['Filename', 'Params']]):
        # Found all relevant Argus II fields:
        feat = _loads_data_row_a60(row)
    elif np.all([c in row for c in ['filename', 'notes']]):
        # Found all relevant Argus I fields:
        feat = _loads_data_row_a16(row)
    else:
        raise ValueError("row is neither Argus I or Argus II data.")

    if feat is None:
        return None

    # Subject string mismatch:
    if subject is not None and feat['subject'] != subject:
        return None
    # Electrode string mismatch:
    if electrodes is not None and feat['electrode'] not in electrodes:
        return None
    # Date string mismatch:
    if date is not None and feat['date'] != date:
        return None
    # Multiple electrodes mentioned:
    if '_' in feat['stim_class']:
        return None
    # Stimulus class mismatch:
    if feat['stim_class'] != 'SingleElectrode':
        return None
    if amplitude is not None and not np.isclose(feat['amp'], amplitude):
        return None
    if frequency is not None and not np.isclose(feat['freq'], frequency):
        return None

    # Load image
    if not os.path.isfile(os.path.join(feat['folder'], feat['filename'])):
        return None
    img = skio.imread(os.path.join(feat['folder'], feat['filename']),
                      as_grey=True)
    feat.update(img_shape=img.shape)

    target = {'image': img,
              'electrode': feat['electrode']}
    # Calculate shape descriptors:
    descriptors = imgproc.calc_shape_descriptors(img)
    target.update(descriptors)

    return feat, target


def load_data_raw(folder, subject=None, electrodes=None, date=None,
                  amplitude=2.0, frequency=20.0,
                  n_min_trials=5, n_max_trials=5,
                  verbose=False, random_state=None,
                  engine='joblib', scheduler='threading', n_jobs=-1):
    # Recursive search for all files whose name contains the string
    # '_rawDataFileList_': These contain the paths to the raw bmp images
    sstr = '*' if subject is None else subject
    search_patterns = [os.path.join(folder, sstr, '**', '*_rawDataFileList_*'),
                       os.path.join(folder, sstr, '**', 'VIDFileListNew_*')]
    dfs = []
    n_samples = 0
    for search_pattern in search_patterns:
        for fname in glob.iglob(search_pattern, recursive=True):
            if fname.endswith('.csv'):
                tmp = pd.read_csv(fname)
            elif fname.endswith('.xls'):
                tmp = pd.read_excel(fname)
            else:
                raise TypeError("Unknown file type for file '%s'." % fname)
            tmp['exp_folder'] = os.path.dirname(fname)
            tmp['exp_file'] = os.path.basename(fname)
            n_samples += len(tmp)
            if verbose:
                print('Found %d files in %s' % (len(tmp),
                                                tmp['exp_folder'].values[0]))
            dfs.append(tmp)
    if n_samples == 0:
        print('No data found in %s' % folder)
        return pd.DataFrame([]), pd.DataFrame([])

    df = pd.concat(dfs)
    if random_state is not None:
        if verbose:
            print('Shuffling data')
        df = sklu.shuffle(df, random_state=random_state)

    # Process rows of the data frame in parallel:
    if verbose:
        print('Parsing data')
    feat_target = p2p.utils.parfor(_loads_data_row, df.iterrows(),
                                   func_args=[subject, electrodes,
                                              amplitude, frequency,
                                              date],
                                   engine=engine, scheduler=scheduler,
                                   n_jobs=n_jobs)
    # Invalid rows are returned as None, filter them out:
    feat_target = list(filter(None, feat_target))
    # For all other rows, a tuple (X, y) is returned:
    features = pd.DataFrame([ft[0] for ft in feat_target])
    targets = pd.DataFrame([ft[1] for ft in feat_target], index=features.index)

    for electrode in features.electrode.unique():
        idx = features.electrode == electrode

        # Drop trials if we have more than `n_max_trials`
        features.drop(index=features[idx].index[n_max_trials:], inplace=True)
        targets.drop(index=targets[idx].index[n_max_trials:], inplace=True)

        # Drop electrodes if we have less than `n_min_trials`
        if n_min_trials > 0 and np.sum(idx) < n_min_trials:
            features.drop(index=features[idx].index, inplace=True)
            targets.drop(index=targets[idx].index, inplace=True)

    if verbose:
        print('Found %d samples: %d feature values, %d target values' % (
            features.shape[0], features.shape[1], targets.shape[1])
        )
    return features, targets


def load_subjects(fname):
    df = pd.read_csv(fname, index_col='subject_id')
    df['xrange'] = pd.Series([(a, b) for a, b in zip(df['xmin'], df['xmax'])],
                             index=df.index)
    df['yrange'] = pd.Series([(a, b) for a, b in zip(df['ymin'], df['ymax'])],
                             index=df.index)
    df['implant_type'] = pd.Series([(p2p.implants.ArgusI if i == 'ArgusI'
                                     else p2p.implants.ArgusII)
                                    for i in df['implant_type_str']],
                                   index=df.index)
    return df.drop(columns=['xmin', 'xmax', 'ymin', 'ymax', 'implant_type_str'])


def load_data(fname, subject=None, electrodes=None, amp=None, random_state=42):
    data = pd.read_csv(fname)
    if subject is not None:
        data = data[data.subject_id == subject]
    if electrodes is not None:
        if not isinstance(electrodes, (list, np.ndarray)):
            raise ValueError("`electrodes` must be a list or NumPy array")
        idx = np.zeros(len(data), dtype=np.bool)
        for e in electrodes:
            idx = np.logical_or(idx, data.PTS_ELECTRODE == e)
        data = data[idx]
    if amp is not None:
        data = data[np.isclose(data.PTS_AMP, amp)]

    if random_state is not None:
        data = sklu.shuffle(data, random_state=random_state)

    # Build feature and target matrices:
    features = []
    targets = []
    for _, row in data.iterrows():
        # Extract shape descriptors from phosphene drawing:
        if pd.isnull(row['PTS_FILE']):
            img = np.zeros((10, 10))
        else:
            try:
                img = skio.imread(os.path.join(os.path.dirname(fname),
                                               row['PTS_FILE']), as_grey=True)
            except FileNotFoundError:
                try:
                    img = skio.imread(row['PTS_FILE'], as_grey=True)
                except FileNotFoundError:
                    s = ('Column "PTS_FILE" must either specify an absolute '
                         'path or a relative path that starts in the '
                         'directory of `fname`.')
                    raise FileNotFoundError(s)
        props = imgproc.calc_shape_descriptors(img)
        target = {'image': img, 'electrode': row['PTS_ELECTRODE']}
        target.update(props)
        targets.append(target)

        # Save additional attributes:
        feat = {
            'subject': row['subject_id'],
            'electrode': row['PTS_ELECTRODE'],
            'filename': row['PTS_FILE'],
            'img_shape': img.shape,
            'stim_class': row['stim_class'],
            'amp': row['PTS_AMP'],
            'freq': row['PTS_FREQ'],
            'pdur': row['PTS_PULSE_DUR'],
            'date': row['date']
        }
        features.append(feat)
    features = pd.DataFrame(features, index=data.index)
    targets = pd.DataFrame(targets, index=data.index)
    return features, targets


def _calcs_mean_image(Xy, groupcols, thresh=True, max_area=1.5):
    for col in groupcols:
        assert len(Xy[col].unique()) == 1
    assert len(Xy.electrode.unique()) == 1

    # Calculate mean image
    images = Xy.image
    img_avg = None

    for img in images:
        if img_avg is None:
            img_avg = np.zeros_like(img, dtype=float)
        img_avg += imgproc.center_phosphene(img)

    # Adjust to [0, 1]
    if img_avg.max() > 0:
        img_avg /= img_avg.max()
    # Threshold if required:
    if thresh:
        img_avg = imgproc.get_thresholded_image(img_avg, thresh='otsu')
    # Move back to its original position:
    img_avg = imgproc.center_phosphene(img_avg, center=(np.mean(Xy.y_center),
                                                        np.mean(Xy.x_center)))

    # Calculate shape descriptors:
    descriptors = imgproc.calc_shape_descriptors(img_avg)

    # Compare area of mean image to the mean of trial images: If smaller than
    # some fraction, skip:
    if descriptors['area'] > max_area * np.mean(Xy.area):
        return None, None

    # Remove ambiguous (trial-related) parameters:
    target = {'electrode': Xy.electrode.unique()[0],
              'image': img_avg}
    target.update(descriptors)

    feat = {'img_shape': img_avg.shape}
    for col in groupcols:
        feat[col] = Xy[col].unique()[0]

    return feat, target


def calc_mean_images(Xraw, yraw, groupcols=['subject', 'amp', 'electrode'],
                     thresh=True, max_area=1.5):
    """Extract mean images on an electrode from all raw trial drawings

    Parameters
    ----------
    Xraw : pd.DataFrame
        Feature matrix, raw trial data
    yraw : pd.DataFrame
        Target values, raw trial data
    thresh : bool, optional, default: True
        Whether to binarize the averaged image.
    max_area : float, optional, default: 2
        Skip if mean image has area larger than a factor `max_area`
        of the mean of the individual images. A large area of the mean
        image indicates poor averaging: instead of maintaining area,
        individual nonoverlapping images are added.

    Returns
    =======
    Xout : pd.DataFrame
        Feature matrix, single entry per electrode
    yout : pd.DataFrame
        Target values, single entry per electrode
    """
    Xy = pd.concat((Xraw, yraw.drop(columns='electrode')), axis=1)
    assert np.allclose(Xy.index, Xraw.index)

    Xout = []
    yout = []
    for _, data in Xy.groupby(groupcols):
        f, t = _calcs_mean_image(data, groupcols, thresh=thresh,
                                 max_area=max_area)
        if f is not None and t is not None:
            Xout.append(f)
            yout.append(t)

    return pd.DataFrame(Xout), pd.DataFrame(yout)
