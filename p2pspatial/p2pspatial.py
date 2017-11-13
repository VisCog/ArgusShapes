from __future__ import absolute_import, division, print_function
import os
import six
import copy
import glob
import logging

import numpy as np
import pandas as pd

import pulse2percept as p2p

import skimage.io as skio
import skimage.filters as skif
import skimage.transform as skit
import skimage.measure as skim
import sklearn.base as sklb

from .due import due, Doi

p2p.console.setLevel(logging.ERROR)

__all__ = ["get_thresholded_image", "get_region_props", "load_data",
           "DataPreprocessor", "SpatialSimulation", "SpatialModelRegressor"]


# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi("10.1167/13.9.30"),
         description="Template project for small scientific Python projects",
         tags=["reference-implementation"],
         path='p2pspatial')


def get_thresholded_image(img, res_shape=None):
    if res_shape is not None:
        img = skit.resize(img, res_shape, mode='reflect')
    try:
        img = (img > skif.threshold_minimum(img)).astype(np.uint8)
        return img
    except RuntimeError:
        print('Runtime error with minimum threshold')
        img = (img >= 128).astype(np.uint8)
    return img


def get_region_props(img, res_shape=None, verbose=True, return_all=False):
    img = get_thresholded_image(img)
    if img is None:
        return None

    regions = skim.regionprops(img)
    if len(regions) == 0:
        return None
    elif len(regions) == 1:
        return regions[0]
    else:
        if return_all:
            return regions
        else:
            # Multiple props, choose largest
            areas = np.array([r.area for r in regions])
            idx = np.argmax(areas)
            if verbose:
                print(('Found multiple regions, chose regions[%d] with '
                       'area %f') % (idx, regions[idx].area))
            return regions[idx]


def load_data(folder, subject=None, electrodes=None, date=None, verbose=True,
              random_state=None, scaling=4, img_shape=(41, 61)):
    # Recursive search for all files whose name contains the string
    # '_rawDataFileList_': These contain the paths to the raw bmp images
    search_pattern = os.path.join(folder, '**', '*_rawDataFileList_*')
    dfs = []
    for fname in glob.iglob(search_pattern, recursive=True):
        tmp = pd.read_csv(fname)
        tmp['Folder'] = os.path.dirname(fname)
        dfs.append(tmp)
    df = pd.concat(dfs)
    if random_state is not None:
        df = shuffle(df, random_state=random_state)

    features = []
    targets = []
    for _, row in df.iterrows():
        # Split the data strings to extract subject, electrode, etc.
        fname = row['Filename']
        date = fname.split('_')[0]
        params = row['Params'].split(' ')
        stim = params[0].split('_')
        if len(params) < 2 or len(stim) < 2:
            if verbose:
                print('Could not parse row:', row['Filename'],
                      row['Params'])
            continue
        if subject is not None and stim[0] != subject:
            continue
        if electrodes is not None:
            if params[1] not in electrodes:
                continue
        if date is not None and date != date:
            continue

        # Find the Hu momemnts of the image: Calculate area in deg^2, but
        # operate on image larger than 1px = 1deg so that thin lines
        # are still visible
        if not os.path.isfile(os.path.join(row['Folder'], row['Filename'])):
            if verbose:
                print('Could not find file:', row['Folder'], row['Filename'])
            continue
        img = skio.imread(os.path.join(row['Folder'], row['Filename']),
                          as_grey=True)
        res_shape = (img_shape[0] * scaling, img_shape[1] * scaling)
        props = get_region_props(img, res_shape=res_shape,
                                 verbose=verbose)
        if props is None:
            if verbose:
                print('Found empty props:', row['Folder'], row['Filename'])

        # Assemble all feature values in a dict
        feat = {'filename': fname,
                'folder': row['Folder'],
                'param_str': row['Params'],
                'subject': stim[0],
                'electrode': params[1],
                'stim_class': stim[1],
                'date': date,
                'scaling': scaling,
                'img_shape': img_shape,
                'area': props.area / scaling ** 2,
                'orientation': props.orientation,
                'major_axis_length': props.major_axis_length / scaling,
                'minor_axis_length': props.minor_axis_length / scaling}
        features.append(feat)
        targets.append(props.moments_hu)
    if verbose:
        print('Found %d samples: %d feature values, %d target values' % (
            len(features), len(features[0]), len(targets[0]))
        )
    return pd.DataFrame(features), pd.DataFrame(targets)


class DataPreprocessor(sklb.TransformerMixin):

    def __init__(self, subject=None, electrodes=None, date=None, verbose=True):
        self.subject = subject
        self.electrodes = electrodes
        self.date = date
        self.verbose = verbose

    def get_params(self, deep=True):
        return {'subject': self.subject,
                'electrodes': self.electrodes,
                'date': self.date,
                'verbose': self.verbose}

    def set_params(self, **params):
        for param, value in six.iteritems(params):
            setattr(self, param, value)

    def fit(self, *_):
        return self

    def transform(self, X, *_):
        assert isinstance(X, pd.core.frame.DataFrame)

        features = []
        for _, row in X.iterrows():
            # Split the data strings to extract subject, electrode, etc.
            fname = row['Filename']
            date = fname.split('_')[0]
            params = row['Params'].split(' ')
            stim = params[0].split('_')
            if len(params) < 2 or len(stim) < 2:
                if self.verbose:
                    print('Could not parse row:', row['Filename'],
                          row['Params'])
                continue
            if self.subject is not None and stim[0] != self.subject:
                continue
            if self.electrodes is not None:
                if params[1] not in self.electrodes:
                    continue
            if self.date is not None and date != self.date:
                continue

            # Find the Hu momemnts of the image: Calculate area in deg^2, but
            # operate on image larger than 1px = 1deg so that thin lines
            # are still visible
            img = skio.imread(os.path.join(row['Folder'], row['Filename']),
                              as_grey=True)
            res_shape = (row['img_rows'] * row['scaling'],
                         row['img_cols'] * row['scaling'])
            props = get_region_props(img, res_shape=res_shape,
                                     verbose=self.verbose)
            if props is None:
                if self.verbose:
                    print('Found empty props:', row['Folder'], row['Filename'])

            # Assemble all feature values in a dict
            feat = {'filename': fname,
                    'folder': row['Folder'],
                    'param_str': row['Params'],
                    'subject': stim[0],
                    'electrode': params[1],
                    'stim_class': stim[1],
                    'date': date,
                    'area': props.area / sc_fact ** 2,
                    'orientation': props.orientation,
                    'major_axis_length': props.major_axis_length / sc_fact,
                    'minor_axis_length': props.minor_axis_length / sc_fact}
            features.append(feat)
        return features


class SpatialSimulation(p2p.Simulation):

    def set_ganglion_cell_layer(self):
        pass

    def pulse2percept(self, electrode, return_both=False):
        cs = self.implant[electrode].current_spread(self.ofl.gridx,
                                                    self.ofl.gridy,
                                                    layer='OFL')
        ecs = self.ofl.current2effectivecurrent(cs)
        if return_both:
            return cs, ecs
        else:
            return ecs


class SpatialModelRegressor(sklb.BaseEstimator, sklb.RegressorMixin):

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        for param, value in six.iteritems(params):
            setattr(self, param, value)

    def fit(self, X, y=None, **fit_params):
        """Gather all parameters needed to instantiate a model"""
        # The grid search call will add all `search_params` as attributes of
        # this class. They need to be combined with `fit_params` and all
        # passed to the model constructor.
        model_params = copy.deepcopy(vars(self))

        # Remove elements that might exist after multiple `fit` calls
        pop_elements = self.get_params()
        pop_elements.update({'_model': None, 'model_params': None})
        for key, _ in six.iteritems(pop_elements):
            if key in model_params:
                model_params.pop(key)

        # Combine with `fit_params`
        model_params.update(fit_params)
        self.model_params = model_params
        return self

    def predict(self, X):
        assert isinstance(X, pd.core.frame.DataFrame)
        assert self.model_params is not None
        mp = self.model_params

        y_pred = []
        for _, row in X.iterrows():
            implant = p2p.implants.ArgusII(x_center=mp['implant_x'],
                                           y_center=mp['implant_y'],
                                           rot=mp['implant_rot'])
            sim = SpatialSimulation(implant)
            sim.set_optic_fiber_layer(sampling=mp['sampling'],
                                      x_range=p2p.retina.dva2ret((-30, 30)),
                                      y_range=p2p.retina.dva2ret((-20, 20)),
                                      decay_const=mp['decay_const'],
                                      sensitivity_rule=mp['sensitivity_rule'])
            # get rid of the leading zeros
            electrode = '%s%d' % (row['electrode'][0],
                                  int(row['electrode'][1:]))
            img = sim.pulse2percept(electrode)
            res_shape = (row['img_shape'][0] * row['scaling'],
                         row['img_shape'][1] * row['scaling'])
            props = get_region_props(img, res_shape=res_shape, verbose=False)
            if props is None:
                print('Could not extract regions:', electrode)
                y_pred.append(np.zeros(7))
            else:
                y_pred.append(props.moments_hu)
        return y_pred
