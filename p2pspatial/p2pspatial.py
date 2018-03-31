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

__all__ = ["load_data", "ret2dva", "dva2ret",
           "adjust_drawing_bias", "calc_mean_images"]


# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi("10.1167/13.9.30"),
         description="Template project for small scientific Python projects",
         tags=["reference-implementation"],
         path='p2pspatial')


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
    props = imgproc.get_region_props(img, thresh=0)
    feat.update(img_shape=img.shape)

    target = {'image': img,
              'electrode': feat['electrode'],
              'x_center': props.centroid[1],
              'y_center': props.centroid[0],
              'area': props.area,
              'orientation': props.orientation,
              'major_axis_length': props.major_axis_length,
              'minor_axis_length': props.minor_axis_length}
    return feat, target


def load_data(folder, subject=None, electrodes=None,
              amplitude=2.0, frequency=20.0, n_min_trials=5, n_max_trials=5,
              date=None, verbose=False, random_state=None,
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


def adjust_drawing_bias(X, y, scale_major=(1, 1), scale_minor=(1, 1),
                        rotate=0):
    targets = []
    for _, row in y.iterrows():
        # Compact and elongated shapes are processed differently:
        img = row['image']
        props = imgproc.get_region_props(img)
        if props.major_axis_length / props.minor_axis_length > 2:
            # Phosphene is elongated:
            smajor = scale_major[1]
            sminor = scale_minor[1]
            rot = np.deg2rad(rotate)
        else:
            # Phosphene is compact:
            smajor = scale_major[0]
            sminor = scale_minor[0]
            rot = 0

        # Shift phosphene to (0, 0) and back
        mom = skime.moments(img, order=1)
        transl = np.array([-mom[1, 0] / mom[0, 0], -mom[0, 1] / mom[0, 0]])
        ts = skit.SimilarityTransform(translation=transl)
        tsi = skit.SimilarityTransform(translation=-transl)

        # Rotate the phosphene so the major axis is oriented along the
        # horizontal:
        props = imgproc.get_region_props(img)
        tr = skit.SimilarityTransform(rotation=props.orientation)
        # When undoing the rotation, add the specified extra rotation:
        tri = skit.SimilarityTransform(rotation=-props.orientation - rot)

        # Scale the phosphene with two different scaling factors along the
        # major and minor axes using a homogeneous transformation matrix:
        mat = np.array([[smajor, 0, 0],
                        [0, sminor, 0],
                        [0, 0, 1]])
        tp = skit.ProjectiveTransform(matrix=mat)

        # Put them all together:
        newimg = skit.warp(img, (ts + (tr + (tp + (tri + tsi)))).inverse)

        # Now we could adjust the size further using a scaling factor for the
        # overall area...

        # Calculate new props:
        props = imgproc.get_region_props(newimg)
        target = {'image': newimg,
                  'electrode': row['electrode'],
                  'x_center': props.centroid[1],
                  'y_center': props.centroid[0],
                  'area': props.area,
                  'orientation': props.orientation,
                  'major_axis_length': props.major_axis_length,
                  'minor_axis_length': props.minor_axis_length}
        targets.append(target)
    return pd.DataFrame(targets, index=X.index)


def _transforms_electrode_images(Xel, threshold=True):
    """Takes all trial images (given electrode) and computes mean image"""
    assert len(Xel.subject.unique()) == 1
    subject = Xel.subject.unique()[0]
    assert len(Xel.amp.unique()) == 1
    amplitude = Xel.amp.unique()[0]
    assert len(Xel.electrode.unique()) == 1
    electrode = Xel.electrode.unique()[0]

    imgs = []
    areas = []
    orientations = []
    for Xrow in Xel.iterrows():
        _, row = Xrow
        img = skio.imread(os.path.join(row['folder'],
                                       row['filename']),
                          as_grey=True)
        img = skimage.img_as_float(img)
        img = imgproc.center_phosphene(img)
        props = imgproc.get_region_props(img)
        assert not np.isnan(props.area)
        assert not np.isnan(props.orientation)
        areas.append(props.area)
        orientations.append(props.orientation)
        imgs.append(img)

    assert len(imgs) > 0
    if len(imgs) == 1:
        # Only one image found, save this one
        img_avg_th = imgproc.get_thresholded_image(img)
    else:
        # More than one image found: Save the first image as seed image to
        # which all other images will be compared:
        # img_seed = skimo.dilation(imgs[0], skimo.disk(5))
        img_seed = imgs[0]
        img_avg = np.zeros_like(img_seed)
        for img in imgs[1:]:
            # Dilate images slightly so matching is easier for streaks
            # (will be eroded later):
            img = skimo.dilation(img, skimo.disk(5))
            _, _, params = imgproc.srd_loss((img_seed, img),
                                            return_raw=True)
            img = imgproc.scale_phosphene(img, params['scale'])
            # There might be more than one optimal angle, choose the smallest:
            angle = params['angle'][np.argmin(np.abs(params['angle']))]
            img = imgproc.rotate_phosphene(img, angle)
            img_avg += img

        if threshold:
            # Binarize the average image:
            img_avg_th = imgproc.get_thresholded_image(img_avg,
                                                       thresh='otsu')
            assert np.isclose(img_avg_th.min(), 0)
            assert np.isclose(img_avg_th.max(), 1)
            # Remove "pepper" (fill small holes):
            img_avg_th = skimo.binary_closing(img_avg_th, selem=skimo.disk(11))
            # Erode back down
            img_avg_th = skimo.erosion(img_avg_th, skimo.disk(5))
            # Rotate the binarized image to have the same orientation as
            # the mean trial image:
            props = imgproc.get_region_props(img_avg_th)
            angle_rad = np.mean(orientations) - props.orientation
            img_avg_th = imgproc.rotate_phosphene(img_avg_th,
                                                  np.rad2deg(angle_rad))
            # Scale the binarized image to have the same area as the mean
            # trial image:
            props = imgproc.get_region_props(img_avg_th)
            scale = np.sqrt(np.mean(areas) / props.area)
            img_avg_th = imgproc.scale_phosphene(img_avg_th, scale)
        else:
            img_avg_th = img_avg

    # The result is an image that has the exact same area and
    # orientation as all trial images averaged. This is what we
    # save:
    props = imgproc.get_region_props(img_avg_th)
    target = {'electrode': electrode, 'image': img_avg_th}

    # Remove ambiguous (trial-related) parameters:
    feat = {'subject': subject, 'amplitude': amplitude,
            'electrode': electrode, 'img_shape': img_avg_th.shape}

    return feat, target


def _calcs_mean_image(Xy, thresh=True):
    assert len(Xy.subject.unique()) == 1
    subject = Xy.subject.unique()[0]
    assert len(Xy.amp.unique()) == 1
    amplitude = Xy.amp.unique()[0]
    assert len(Xy.electrode.unique()) == 1
    electrode = Xy.electrode.unique()[0]

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
    # Calculate props:
    props = imgproc.get_region_props(img_avg)

    # Remove ambiguous (trial-related) parameters:
    target = {'electrode': electrode, 'image': img_avg}
    feat = {'subject': subject, 'amplitude': amplitude,
            'electrode': electrode, 'img_shape': img_avg.shape}

    return feat, target


def calc_mean_images(Xraw, yraw, thresh=True):
    """Extract mean images on an electrode from all raw trial drawings

    Parameters
    ----------
    Xraw : pd.DataFrame
        Feature matrix, raw trial data
    yraw : pd.DataFrame
        Target values, raw trial data

    Returns
    =======
    Xout : pd.DataFrame
        Feature matrix, single entry per electrode
    yout : pd.DataFrame
        Target values, single entry per electrode
    """
    Xy = pd.concat((Xraw, yraw.drop(columns='electrode')), axis=1)
    assert np.allclose(Xy.index, Xraw.index)
    subjects = Xy.subject.unique()
    Xout = []
    yout = []
    for subject in subjects:
        X = Xy[Xy.subject == subject]
        amplitudes = X.amp.unique()

        for amp in amplitudes:
            Xamp = X[np.isclose(X.amp, amp)]
            electrodes = np.unique(Xamp.electrode)

            Xel = [Xamp[Xamp.electrode == e] for e in electrodes]
            feat_target = p2p.utils.parfor(_calcs_mean_image, Xel,
                                           func_kwargs={'thresh': thresh})
            Xout += [ft[0] for ft in feat_target]
            yout += [ft[1] for ft in feat_target]
    # Return feature matrix and target values as DataFrames
    return pd.DataFrame(Xout), pd.DataFrame(yout)


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
