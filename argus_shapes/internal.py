from __future__ import absolute_import, division, print_function

from . import imgproc

import os
import six
import glob
import numpy as np
import pandas as pd
import skimage.io as skio
import sklearn.utils as sklu
import pulse2percept as p2p


def _loads_data_row_a16(row):
    # Electrodes have legacy names in datafiles:
    old_names = ['L6', 'L2', 'M8', 'M4', 'L5', 'L1', 'M7', 'M3',
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
    if 'Filename' not in row:
        print('No Filename:')
        print(row)
        return None
    fname = row['Filename']
    if not isinstance(fname, six.string_types):
        print('Wrong Filename type')
        print(row)
        return None
    date = fname.split('_')[0]

    if 'Params' not in row:
        print('No Params:')
        print(row)
        return None
    if not isinstance(row['Params'], six.string_types):
        print('Wrong Params type')
        print(row)
        return None
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


def _loads_data_row(df_row, subject, electrodes, amp, freq, stim_class, date):
    _, row = df_row

    if (np.all([c in row for c in ['Filename', 'Params']]) and
            np.all(pd.notnull([row[c] for c in ['Filename', 'Params']]))):
        # Found all relevant Argus II fields:
        feat = _loads_data_row_a60(row)
    elif (np.all([c in row for c in ['filename', 'notes']]) and
          np.all(pd.notnull([row[c] for c in ['filename', 'notes']]))):
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
    # if electrodes is not None and feat['electrode'] not in electrodes:
    #    return None
    # Date string mismatch:
    if date is not None and feat['date'] != date:
        return None
    # Stimulus class mismatch:
    # if stim_class is not None:
    #    if feat['stim_class'] != stim_class:
    #        return None
    # if amp is not None:
    #    if (isinstance(feat['amp'], (list, np.ndarray)) and
    #        np.all(np.isclose(feat['amp'], amp))) or
    #       (not isinstance(feat['amp'], (list, np.ndarray)) and
    #        np.isclose(feat['amp'], amp)):
    #        return None
    # if freq is not None and not np.isclose(feat['freq'], freq):
    #    return None

    # Load image
    if not os.path.isfile(os.path.join(feat['folder'], feat['filename'])):
        return None
    img = skio.imread(os.path.join(feat['folder'], feat['filename']),
                      as_gray=True)
    feat.update(img_shape=img.shape)

    target = {'image': img,
              'electrode': feat['electrode']}
    # Calculate shape descriptors:
    descriptors = imgproc.calc_shape_descriptors(img)
    target.update(descriptors)

    return feat, target


def load_data_raw(folder, subject=None, electrodes=None, date=None,
                  amplitude=2.0, frequency=20.0, stim_class='SingleElectrode',
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
                                   func_args=[subject, electrodes, amplitude,
                                              frequency, stim_class, date],
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
