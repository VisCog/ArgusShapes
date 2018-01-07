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
import sklearn.utils as sklu

from .due import due, Doi

p2p.console.setLevel(logging.ERROR)

__all__ = ["get_thresholded_image", "get_avg_image", "get_region_props",
           "load_data", "transform_data",
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
    if thresh == 'min':
        try:
            thresh = skif.threshold_minimum(img)
        except RuntimeError:
            if verbose:
                print('Runtime error with minimum threshold')
            thresh = (img.max() - img.min()) // 2
    else:
        assert isinstance(thresh, (int, float))
    img_th = img > thresh
    return img_th.astype(np.uint8) * 255


def get_avg_image(X, subject, electrode, amp=None, align_center=None):
    idx = np.logical_and(X['subject'] == subject, X['electrode'] == electrode)
    if amp is not None:
        idx_amp = np.isclose(amp, X['amp'])
        assert np.any(idx_amp)
        idx = np.logical_and(idx, idx_amp)

    avg_img = None
    for _, row in X[idx].iterrows():
        img = skio.imread(os.path.join(row['folder'], row['filename']),
                          as_grey=True)
        if align_center is None:
            # Choose center of image
            img_shape = img.shape[:2]
            align_center = [img_shape[1] // 2, img_shape[0] // 2]

        transl = [align_center[0] - row['centroid'][1],
                  align_center[1] - row['centroid'][0]]
        trafo = skit.EuclideanTransform(translation=transl)
        if avg_img is None:
            avg_img = skit.warp(img, trafo.inverse)
        else:
            avg_img += skit.warp(img, trafo.inverse)
    return avg_img


def get_region_props(img, thresh='min', res_shape=None, verbose=True,
                     return_all=False):
    img = get_thresholded_image(img, thresh=thresh, res_shape=res_shape,
                                verbose=verbose)
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


def load_data(folder, subject=None, electrodes=None, amplitude=None,
              date=None, verbose=False, random_state=None, single_stim=True):
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
            print('Found %d samples in %s' % (len(tmp),
                                              tmp['Folder'].values[0]))
        dfs.append(tmp)
    if n_samples == 0:
        print('No data found in %s' % folder)
        return pd.DataFrame([]), pd.DataFrame([])

    df = pd.concat(dfs)
    if random_state is not None:
        df = sklu.shuffle(df, random_state=random_state)

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

        # Find the current amplitude in the folder name
        # It could have any of the following formats: '/3xTh', '_2.5xTh',
        # ' 2xTh'. Idea: Find the string 'xTh', then walk backwards to
        # find the last occurrence of '_', ' ', or '/'
        idx_end = row['Folder'].find('xTh')
        if idx_end == -1:
            if verbose:
                print('Could not find "xTh" in row:', row['Folder'])
            continue
        idx_start = np.max([row['Folder'].rfind('_', 0, idx_end),
                            row['Folder'].rfind(' ', 0, idx_end),
                            row['Folder'].rfind('/', 0, idx_end)])
        if idx_start == -1:
            if verbose:
                print('Could not find amplitude in row:', row['Folder'])
            continue
        amp = float(row['Folder'][idx_start + 1:idx_end])
        if amplitude is not None:
            if not np.isclose(amp, amplitude):
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

        # We use the image at original resolution
        props = get_region_props(img, thresh=128,
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
                'amp': amp,
                'date': date,
                'img_shape': img.shape,
                'centroid': props.centroid}
        features.append(feat)
        target = {'area': props.area,
                  'orientation': props.orientation,
                  'major_axis_length': props.major_axis_length,
                  'minor_axis_length': props.minor_axis_length}
        targets.append(target)
    if verbose:
        print('Found %d samples: %d feature values, %d target values' % (
            len(features), len(features[0]), len(targets[0]))
        )
    return pd.DataFrame(features), pd.DataFrame(targets)


def transform_data(Xold, yold):
    """Average trials to yield mean images"""
    Xy = pd.concat((Xold, yold), axis=1).groupby(['electrode', 'amp'])
    df = pd.DataFrame(Xy[yold.columns].mean()).reset_index()

    Xnew = df.loc[:, ['electrode', 'amp']]
    for col in set(Xold.columns) - set(Xnew.columns) & set(Xold.columns):
        Xnew[col] = Xold[col]
    ynew = df.loc[:, yold.columns]
    return Xnew, ynew


class SpatialSimulation(p2p.Simulation):

    def set_params(self, **params):
        for param, value in six.iteritems(params):
            setattr(self, param, value)

    def set_ganglion_cell_layer(self):
        self.gcl = {}

    def calc_electrode_ecs(self, electrode, gridx, gridy):
        assert isinstance(electrode, six.string_types)
        assert isinstance(self.csmode, six.string_types)
        ename = '%s%d' % (electrode[0], int(electrode[1:]))

        # Current spread either from Nanduri model or with fitted radius
        if self.csmode.lower() == 'ahuja':
            cs = self.implant[ename].current_spread(gridx, gridy, layer='OFL')
        elif self.csmode.lower() == 'gaussian':
            assert isinstance(self.cswidth, (int, float))
            assert self.cswidth > 0
            r2 = (gridx - self.implant[ename].x_center) ** 2
            r2 += (gridy - self.implant[ename].y_center) ** 2
            cs = np.exp(-r2 / (2.0 * self.cswidth ** 2))
        else:
            raise ValueError('Unknown csmode "%s"' % self.csmode)

        # Take into account axonal stimulation
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

    def pulse2percept(self, el_str, amp):
        assert isinstance(el_str, six.string_types)
        assert isinstance(amp, (int, float))
        assert amp >= 0
        if np.isclose(amp, 0):
            print('Warning: amp is zero on %s' % el_str)

        ecs = np.zeros_like(self.ofl.gridx)
        electrodes = el_str.split('_')
        for e in electrodes:
            if e not in self.ecs:
                # It's possible that the test set contains an electrode that
                # was not in the training set (and thus not in ``fit``)
                self.calc_currents([e])
            ecs += self.ecs[e]
        if ecs.max() > 0:
            ecs = ecs / ecs.max() * amp
        return np.flipud(ecs)


class SpatialModelRegressor(sklb.BaseEstimator, sklb.RegressorMixin):

    def __init__(self, sampling=100, x_range=(-30, 30), y_range=(-20, 20),
                 sensitivity_rule='decay', thresh=1.0, csmode='gaussian'):
        self.sampling = sampling
        self.x_range = x_range
        self.y_range = y_range
        self.sensitivity_rule = sensitivity_rule
        self.thresh = thresh
        self.csmode = csmode

    def set_params(self, **params):
        for param, value in six.iteritems(params):
            setattr(self, param, value)

    def fit(self, X, y=None, **fit_params):
        """Gather all parameters needed to instantiate a model"""
        assert isinstance(X, pd.core.frame.DataFrame)
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
        self.set_params(**model_params)
        assert isinstance(self.implant_x, (int, float))
        assert isinstance(self.implant_y, (int, float))
        assert isinstance(self.implant_rot, (int, float))
        assert isinstance(self.loc_od_x, (int, float))
        assert isinstance(self.loc_od_y, (int, float))
        assert isinstance(self.decay_const, (int, float))
        assert isinstance(self.scoring_weights, dict)

        print('implant (x, y): (%.2f, %.2f), rot: %f' % (self.implant_x,
                                                         self.implant_y,
                                                         self.implant_rot))
        if np.abs(self.implant_rot) > 2 * np.pi:
            print('[WARNING] implant_rot should be set in radians!!')

        implant = p2p.implants.ArgusII(x_center=self.implant_x,
                                       y_center=self.implant_y,
                                       rot=self.implant_rot)
        sim = SpatialSimulation(implant)
        sim.set_params(csmode=self.csmode, cswidth=self.cswidth)

        print('Set loc_od:', self.loc_od_x, self.loc_od_y,
              'decay_const:', self.decay_const,
              'sensitivity_rule:', self.sensitivity_rule,
              'thresh:', self.thresh)
        sim.set_optic_fiber_layer(sampling=self.sampling,
                                  x_range=p2p.retina.dva2ret(self.x_range),
                                  y_range=p2p.retina.dva2ret(self.y_range),
                                  loc_od=(self.loc_od_x, self.loc_od_y),
                                  decay_const=self.decay_const,
                                  sensitivity_rule=self.sensitivity_rule)
        sim.calc_currents(np.unique(X['electrode']))
        self.sim = sim
        return self

    def _predict(self, Xrow):
        _, row = Xrow
        img = self.sim.pulse2percept(row['electrode'], row['amp'])
        assert np.isclose(img.max(), row['amp'])

        props = get_region_props(img, thresh=self.thresh,
                                 res_shape=row['img_shape'], verbose=False)
        if props is None:
            print('Could not extract regions:', row['electrode'])
            y_pred = {'area': 0, 'orientation': 0, 'major_axis_length': 0,
                      'minor_axis_length': 0}
        else:
            y_pred = {'area': props.area,
                      'orientation': props.orientation,
                      'major_axis_length': props.major_axis_length,
                      'minor_axis_length': props.minor_axis_length}
        return y_pred

    def predict_image(self, X):
        y_pred = []
        for _, row in X.iterrows():
            y_pred.append(self.sim.pulse2percept(row['electrode'], row['amp']))
        return y_pred

    def predict(self, X):
        assert isinstance(X, pd.core.frame.DataFrame)
        assert self.model_params is not None
        assert self.sim is not None

        y_pred = p2p.utils.parfor(self._predict, X.iterrows(),
                                  engine=self.sim.engine,
                                  scheduler=self.sim.scheduler,
                                  n_jobs=self.sim.n_jobs)
        return y_pred

    def score(self, X, y, sample_weight=None):
        assert isinstance(X, pd.core.frame.DataFrame)
        assert isinstance(y, pd.core.frame.DataFrame)
        assert isinstance(self.scoring_weights, dict)
        scoring_weights = self.scoring_weights

        y_pred = pd.DataFrame(self.predict(X))
        rmse = 0
        for key, colweight in six.iteritems(scoring_weights):
            if colweight is None or np.isclose(colweight, 0):
                continue

            if key == 'orientation':
                # Error is periodic with 2pi
                err = np.mod(y_pred.loc[:, key] - y.loc[:, key], 2 * np.pi)
                err = np.where(err > np.pi, 2 * np.pi - err, err)
                mse = np.average(err ** 2, axis=0, weights=sample_weight)
            else:
                mse = np.average((y.loc[:, key] - y_pred.loc[:, key]) ** 2,
                                 axis=0, weights=sample_weight)
            rmse += colweight * np.sqrt(mse)
        print('RMSE:', rmse)
        return rmse
