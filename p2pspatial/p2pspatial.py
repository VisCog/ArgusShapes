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
import sklearn.metrics as sklm

from .due import due, Doi

p2p.console.setLevel(logging.ERROR)

__all__ = ["get_thresholded_image", "get_region_props", "load_data",
           "SpatialSimulation", "SpatialModelRegressor"]


# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi("10.1167/13.9.30"),
         description="Template project for small scientific Python projects",
         tags=["reference-implementation"],
         path='p2pspatial')


def get_thresholded_image(img, thresh='min', res_shape=None, verbose=True):
    if res_shape is not None:
        img = skit.resize(img, res_shape, mode='reflect')
    if thresh is 'min':
        try:
            thresh = skif.threshold_minimum(img)
        except RuntimeError:
            if verbose:
                print('Runtime error with minimum threshold')
            halfway = (img.max() - img.min()) // 2

    return (img > thresh).astype(np.uint8)


def get_region_props(img, thresh='min', res_shape=None, verbose=True,
                     return_all=False):
    img = get_thresholded_image(img, thresh=thresh, verbose=verbose)
    if img is None:
        return None

    regions = skim.regionprops(img)
    if len(regions) == 0:
        print('No regions: min=%f max=%f' % (img.min(), img.max()))
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


def load_data(folder, subject=None, electrodes=None, date=None, verbose=False,
              random_state=None, scaling=4, img_shape=(41, 61),
              single_stim=True):
    # Recursive search for all files whose name contains the string
    # '_rawDataFileList_': These contain the paths to the raw bmp images
    search_pattern = os.path.join(folder, '**', '*_rawDataFileList_*')
    dfs = []
    n_samples = 0
    for fname in glob.iglob(search_pattern, recursive=True):
        tmp = pd.read_csv(fname)
        tmp['Folder'] = os.path.dirname(fname)
        n_samples += len(tmp)
        if verbose:
            print('Found %d samples in %s' % (len(tmp), tmp['Folder']))
        dfs.append(tmp)
    if n_samples == 0:
        print('No data found in %s' % folder)
        return pd.DataFrame([]), pd.DataFrame([])

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
        if single_stim and '_' in params[1]:
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
        props = get_region_props(img, thresh=128, res_shape=res_shape,
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


class SpatialSimulation(p2p.Simulation):

    def set_ganglion_cell_layer(self):
        self.gcl = {}
        pass

    def calc_electrode_ecs(self, electrode, gridx, gridy):
        assert isinstance(electrode, six.string_types)
        ename = '%s%d' % (electrode[0], int(electrode[1:]))
        cs = self.implant[ename].current_spread(gridx, gridy, layer='OFL')
        ecs = self.ofl.current2effectivecurrent(cs)
        return ecs

    def calc_currents(self, electrodes, verbose=True):
        assert isinstance(electrodes, (list, np.ndarray))

        # Multiple electrodes possible, separated by '_'
        list_2d = [e.split('_') for e in list(electrodes)]
        list_1d = [item for sublist in list_2d for item in sublist]
        electrodes = np.unique(list_1d)
        if verbose:
            print('Calculating effective current for electrodes:', electrodes)

        ecs = p2p.utils.parfor(self.calc_electrode_ecs, electrodes,
                               func_args=[self.ofl.gridx, self.ofl.gridy],
                               engine=self.engine, scheduler=self.scheduler,
                               n_jobs=self.n_jobs)
        if not hasattr(self, 'ecs'):
            self.ecs = {}
        for k, v in zip(electrodes, ecs):
            self.ecs[k] = v
        if verbose:
            print('Done.')

    def pulse2percept(self, el_str):
        assert isinstance(el_str, six.string_types)

        ecs = np.zeros_like(self.ofl.gridx)
        electrodes = el_str.split('_')
        for e in electrodes:
            if e not in self.ecs:
                # It's possible that the test set contains an electrode that
                # was not in the training set (and thus not in ``fit``)
                self.calc_currents([e])
            ecs += self.ecs[e]
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
        assert isinstance(self.model_params['implant_x'], (int, float))
        assert isinstance(self.model_params['implant_y'], (int, float))
        assert isinstance(self.model_params['implant_rot'], (int, float))
        assert isinstance(self.model_params['loc_od'], tuple)
        assert isinstance(self.model_params['decay_const'], (int, float))
        assert isinstance(self.model_params['thresh'], (int, float,
                                                        six.string_types))

        mp = self.model_params
        print('implant (x, y): (%.2f, %.2f), rot: %f' % (mp['implant_x'],
                                                         mp['implant_y'],
                                                         mp['implant_rot']))
        implant = p2p.implants.ArgusII(x_center=mp['implant_x'],
                                       y_center=mp['implant_y'],
                                       rot=mp['implant_rot'])
        sim = SpatialSimulation(implant)

        print('Set loc_od:', mp['loc_od'], 'decay_const:', mp['decay_const'],
              'sensitivity_rule:', mp['sensitivity_rule'])
        sim.set_optic_fiber_layer(sampling=mp['sampling'],
                                  x_range=p2p.retina.dva2ret((-30, 30)),
                                  y_range=p2p.retina.dva2ret((-20, 20)),
                                  loc_od=mp['loc_od'],
                                  decay_const=mp['decay_const'],
                                  sensitivity_rule=mp['sensitivity_rule'])
        sim.calc_currents(np.unique(X['electrode']))
        self.sim = sim
        return self

    def _predict(self, Xrow):
        _, row = Xrow
        img = self.sim.pulse2percept(row['electrode'])
        res_shape = (row['img_shape'][0] * row['scaling'],
                     row['img_shape'][1] * row['scaling'])
        props = get_region_props(img, thresh=self.model_params['thresh'],
                                 res_shape=res_shape, verbose=False)
        if props is None:
            print('Could not extract regions:', row['electrode'])
            return np.zeros(7)
        return props.moments_hu

    def predict(self, X):
        assert isinstance(X, pd.core.frame.DataFrame)
        assert self.model_params is not None
        assert self.sim is not None

        y_pred = p2p.utils.parfor(self._predict, X.iterrows(),
                                  engine=self.sim.engine,
                                  scheduler=self.sim.scheduler,
                                  n_jobs=self.sim.n_jobs)
        return y_pred

    def rmse(self, X, y, sample_weight=None):
        return np.sqrt(sklm.mean_squared_error(y, self.predict(X),
                                               sample_weight=sample_weight))
