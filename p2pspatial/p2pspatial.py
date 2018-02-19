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
import skimage.morphology as skim
import skimage.transform as skit

import sklearn.base as sklb
import sklearn.metrics as sklm
import sklearn.utils as sklu

from .due import due, Doi
from . import imgproc

p2p.console.setLevel(logging.ERROR)

__all__ = ["load_data", "transform_mean_images", "ret2dva", "dva2ret"]


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

    # Assemble all feature values in a dict
    feat = {'filename': row['filename'],
            'folder': os.path.join(row['exp_folder'], row['foldername']),
            'subject': subject,
            'electrode': electrodes,
            'param_str': row['notes'],
            'stim_class': stim_class,
            'amp': amp,
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

    # Assemble all feature values in a dict
    feat = {'filename': fname,
            'folder': row['exp_folder'],
            'subject': stim[0],
            'param_str': row['Params'],
            'electrode': params[1],
            'stim_class': stim[1],
            'amp': amp,
            'date': date}
    return feat


def _loads_data_row(df_row, subject, electrodes, amplitude, date, single_stim):
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
    if single_stim and '_' in feat['stim_class']:
        return None
    # Stimulus class mismatch:
    if single_stim and feat['stim_class'] != 'SingleElectrode':
        return None
    if amplitude is not None and not np.isclose(feat['amp'], amplitude):
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


def load_data(folder, subject=None, electrodes=None, amplitude=None,
              date=None, verbose=False, random_state=None, single_stim=True,
              engine='joblib', scheduler='threading', n_jobs=-1):
    # Recursive search for all files whose name contains the string
    # '_rawDataFileList_': These contain the paths to the raw bmp images
    search_patterns = [os.path.join(folder, '**', '*_rawDataFileList_*'),
                       os.path.join(folder, '*', 'VIDFileListNew_*')]
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
                                              date, single_stim],
                                   engine=engine, scheduler=scheduler,
                                   n_jobs=n_jobs)
    # Invalid rows are returned as None, filter them out:
    feat_target = list(filter(None, feat_target))
    # For all other rows, a tuple (X, y) is returned:
    features = [ft[0] for ft in feat_target]
    targets = [ft[1] for ft in feat_target]

    if verbose:
        print('Found %d samples: %d feature values, %d target values' % (
            len(features), len(features[0]), len(targets[0]))
        )
    return pd.DataFrame(features), pd.DataFrame(targets)


def _transforms_electrode_images(Xel):
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
        props = imgproc.get_region_props(img, thresh=0)
        assert not np.isnan(props.area)
        assert not np.isnan(props.orientation)
        areas.append(props.area)
        orientations.append(props.orientation)
        imgs.append(img)

    assert len(imgs) > 0
    if len(imgs) == 1:
        # Only one image found, save this one
        img_avg_th = imgproc.get_thresholded_image(img, thresh=0)
    else:
        # More than one image found: Save the first image as seed image to
        # which all other images will be compared:
        img_seed = skim.dilation(imgs[0], skim.disk(5))
        img_avg = np.zeros_like(img_seed)
        for img in imgs[1:]:
            # Dilate images slightly so matching is easier for streaks
            # (will be eroded later):
            img = skim.dilation(img, skim.disk(5))
            _, _, params = imgproc.srd_loss((img_seed, img),
                                            return_raw=True)
            img = imgproc.scale_phosphene(img, params['scale'])
            # There might be more than one optimal angle, choose the smallest:
            angle = params['angle'][np.argmin(np.abs(params['angle']))]
            img = skit.rotate(img, angle, order=3)
            img_avg += img

        # Binarize the average image:
        img_avg_th = imgproc.get_thresholded_image(img_avg,
                                                   thresh='otsu')
        assert np.isclose(img_avg_th.min(), 0)
        assert np.isclose(img_avg_th.max(), 1)
        # Remove "pepper" (fill small holes):
        img_avg_th = skim.binary_closing(img_avg_th, selem=skim.disk(11))
        # Erode back down
        img_avg_th = skim.erosion(img_avg_th, skim.disk(5))
        # Remove "salt" (remove small bright spots):
        # img_avg_morph = skim.binary_opening(img_avg_morph,
        #                                     selem=skim.square(9))
        # if not np.allclose(img_avg_morph, np.zeros_like(img_avg_morph)):
        #     img_avg_th = img_avg_morph
        # Rotate the binarized image to have the same orientation as
        # the mean trial image:
        props = imgproc.get_region_props(img_avg_th, thresh=0)
        angle_rad = np.mean(orientations) - props.orientation
        img_avg_th = skit.rotate(img_avg_th, np.rad2deg(angle_rad),
                                 order=3)
        # Scale the binarized image to have the same area as the mean
        # trial image:
        props = imgproc.get_region_props(img_avg_th, thresh=0)
        scale = np.sqrt(np.mean(areas) / props.area)
        img_avg_th = imgproc.scale_phosphene(img_avg_th, scale)

    # The result is an image that has the exact same area and
    # orientation as all trial images averaged. This is what we
    # save:
    target = {'electrode': electrode, 'image': img_avg_th}

    # Remove ambiguous (trial-related) parameters:
    feat = {'subject': subject, 'amplitude': amplitude,
            'electrode': electrode, 'img_shape': img_avg_th.shape}

    return feat, target


def transform_mean_images(Xraw, yraw):
    subjects = Xraw.subject.unique()
    Xout = []
    yout = []

    for subject in subjects:
        X = Xraw[Xraw.subject == subject]
        amplitudes = X.amp.unique()

        for amp in amplitudes:
            Xamp = X[X.amp == amp]
            electrodes = Xamp.electrode.unique()

            Xel = [Xamp[Xamp.electrode == e] for e in electrodes]
            feat_target = p2p.utils.parfor(_transforms_electrode_images, Xel)
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

