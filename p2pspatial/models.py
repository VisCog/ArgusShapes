import numpy as np
import pandas as pd
import abc
import six

import pulse2percept.retina as p2pr
import pulse2percept.implants as p2pi
import pulse2percept.utils as p2pu

import scipy.stats as sps

import sklearn.base as sklb
import sklearn.exceptions as skle

from . import imgproc


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def calc_displacement(r, meridian='temporal'):
    alpha = np.where(meridian == 'temporal', 1.8938, 2.4607)
    beta = np.where(meridian == 'temporal', 2.4598, 1.7463)
    gamma = np.where(meridian == 'temporal', 0.91565, 0.77754)
    delta = np.where(meridian == 'temporal', 14.904, 15.111)
    mu = np.where(meridian == 'temporal', -0.09386, -0.15933)
    scale = np.where(meridian == 'temporal', 12.0, 10.0)

    rmubeta = (np.abs(r) - mu) / beta
    numer = delta * gamma * np.exp(-rmubeta ** gamma)
    numer *= rmubeta ** (alpha * gamma - 1)
    denom = beta * sps.gamma.pdf(alpha, 5)

    return numer / denom / scale


def displace(xy, eye='RE'):
    if eye == 'LE':
        # Let's not think about eyes right now...
        raise NotImplementedError

    # Convert x, y (dva) into polar coordinates
    theta, rho_dva = cart2pol(xy[:, 0], xy[:, 1])

    # Add displacement
    meridian = np.where(xy[:, 0] < 0, 'temporal', 'nasal')
    rho_dva += calc_displacement(rho_dva, meridian=meridian)

    # Convert back to x, y (dva)
    x, y = pol2cart(theta, rho_dva)

    # Convert to retinal coords
    return p2pr.dva2ret(x), p2pr.dva2ret(y)


class CoordTrafoMixin(object):

    def _build_retinal_grid(self):
        # Build the grid from `x_range`, `y_range`:
        nx = int(np.ceil((np.diff(self.xrange) + 1) / self.xystep))
        ny = int(np.ceil((np.diff(self.yrange) + 1) / self.xystep))
        xdva, ydva = np.meshgrid(np.linspace(*self.xrange, num=nx),
                                 np.linspace(*self.yrange, num=ny),
                                 indexing='xy')

        # Convert dva to retinal coordinates
        xydva = np.vstack((xdva.ravel(), ydva.ravel())).T
        xret, yret = displace(xydva)
        self.xret = xret.reshape(xdva.shape)
        self.yret = yret.reshape(ydva.shape)


class RetinalGridMixin(object):

    def _build_retinal_grid(self):
        # Build the grid from `x_range`, `y_range`:
        nx = int(np.ceil((np.diff(self.xrange) + 1) / self.xystep))
        ny = int(np.ceil((np.diff(self.yrange) + 1) / self.xystep))
        xdva, ydva = np.meshgrid(np.linspace(*self.xrange, num=nx),
                                 np.linspace(*self.yrange, num=ny),
                                 indexing='xy')

        self.xret = p2pr.dva2ret(xdva)
        self.yret = p2pr.dva2ret(ydva)


@six.add_metaclass(abc.ABCMeta)
class BaseModel(sklb.BaseEstimator):

    def __init__(self, **kwargs):
        # if not isinstance(implant, p2pi.ElectrodeArray):
        #     raise TypeError(("'implant' must be of type 'ElectrodeArray', "
        #                      "not '%s'." % type(implant)))
        # The following parameters serve as default values and can be
        # overwritten via `kwargs`
        self.implant_type = p2pi.ArgusII
        self.implant_x = 0
        self.implant_y = 0
        self.implant_rot = 0

        # We will be simulating an x,y patch of the visual field (min, max)
        # at a given spatial resolution (step size):
        self.xrange = (-30, 30)  # dva
        self.yrange = (-20, 20)  # dva
        self.xystep = 0.1  # dva

        # Current maps are thresholded to produce a binary image:
        self.img_thresh = 0.1

        # The new scoring function is actually a loss function, so that
        # greater values do *not* imply that the estimator is better
        self.greater_is_better = False

        # JobLib or Dask can be used to parallelize computations:
        self.engine = 'joblib'
        self.scheduler = 'threading'
        self.n_jobs = -1

        # We will store the current map for each electrode in a dict: Since we
        # are usually fitting to individual drawings, we don't want to
        # recompute the current maps for the same electrode on each trial.
        self._curr_map = {}

        # This flag will be flipped once the ``fit`` method was called
        self._is_fitted = False

        # Additional parameters can be set using ``_set_default_params``
        self._set_default_params()
        self.set_params(**kwargs)

    def get_params(self, deep=True):
        """Returns all params that can be set on-the-fly via 'set_params'"""
        return {'implant_type': self.implant_type,
                'implant_x': self.implant_x,
                'implant_y': self.implant_y,
                'implant_rot': self.implant_rot,
                'xrange': self.xrange,
                'yrange': self.yrange,
                'xystep': self.xystep,
                'img_thresh': self.img_thresh,
                'engine': self.engine,
                'scheduler': self.scheduler,
                'n_jobs': self.n_jobs}

    def _set_default_params(self):
        """Derived classes can set additional default parameters here"""
        pass

    @abc.abstractmethod
    def _build_retinal_grid(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _calc_curr_map(self, Xrow):
        raise NotImplementedError

    def calc_curr_map(self, X):
        # Calculate current maps only if necessary:
        # - Get a list of all electrodes for which we already have a curr map:
        has_el = set(self._curr_map.keys())
        # - Compare with electrodes in `X` to find the ones we don't have:
        needs_el = set(X.electrode).difference(has_el)
        # - Calculate the current maps for the missing electrodes:
        curr_map = p2pu.parfor(self._calc_curr_map, needs_el,
                               engine=self.engine, scheduler=self.scheduler,
                               n_jobs=self.n_jobs)
        # - Store the new current maps:
        for key, cm in curr_map:
            # We should process each key only once:
            assert key not in self._curr_map
            self._curr_map[key] = cm

    def fit(self, X, y=None, **fit_params):
        """Fits the model"""
        if not isinstance(X, pd.core.frame.DataFrame):
            raise TypeError("'X' must be a pandas DataFrame, not %s" % type(X))
        if y is not None and not isinstance(y, pd.core.frame.DataFrame):
            raise TypeError("'y' must be a pandas DataFrame, not %s" % type(y))
        # Set additional parameters:
        self.set_params(**fit_params)
        # Instantiate implant:
        self.implant = self.implant_type(x_center=self.implant_x,
                                         y_center=self.implant_y,
                                         rot=self.implant_rot)
        # Convert dva to retinal coordinates:
        self._build_retinal_grid()
        # Calculate current spread for every electrode in `X`:
        self.calc_curr_map(X)
        # Inform the object that is has been fitted:
        self._is_fitted = True
        return self

    def _predict(self, Xrow):
        """Predicts a single data point"""
        _, row = Xrow
        assert isinstance(row, pd.core.series.Series)
        res_shape = None
        if 'img_shape' in row:
            res_shape = row['img_shape']
        elif 'image' in row:
            res_shape = row['image'].image
        curr_map = self._curr_map[row['electrode']]
        img = imgproc.get_thresholded_image(curr_map, thresh=self.img_thresh,
                                            res_shape=res_shape)
        return {'image': img}

    def predict(self, X):
        """Compute predicted drawing"""
        if not self._is_fitted:
            raise skle.NotFittedError("This model is not fitted yet. Call "
                                      "'fit' with appropriate arguments "
                                      "before using this method.")
        if not isinstance(X, pd.core.frame.DataFrame):
            raise TypeError("`X` must be a pandas DataFrame, not %s" % type(X))

        # Make sure we calculated the current maps for all electrodes in `X`:
        self.calc_curr_map(X)

        # Predict attributes of region props (area, orientation, etc.)
        y_pred = p2pu.parfor(self._predict, X.iterrows(),
                             engine=self.engine, scheduler=self.scheduler,
                             n_jobs=self.n_jobs)

        # Convert to DataFrame, preserving the index of `X` (otherwise
        # subtraction in the scoring function produces nan)
        return pd.DataFrame(y_pred, index=X.index)

    def score(self, X, y, sample_weight=None):
        """Score the model using the new loss function"""
        if not isinstance(X, pd.core.frame.DataFrame):
            raise TypeError("'X' must be a pandas DataFrame, not %s" % type(X))
        if not isinstance(y, pd.core.frame.DataFrame):
            raise TypeError("'y' must be a pandas DataFrame, not %s" % type(y))

        y_pred = self.predict(X)

        # `y` and `y_pred` must have the same index, otherwise subtraction
        # produces nan
        assert np.allclose(y_pred.index, y.index)

        # Compute the scaling factor / rotation angle / dice coefficient loss:
        # The loss function expects a tupel of two DataFrame rows
        losses = p2pu.parfor(imgproc.scale_rot_dice_loss,
                             zip(y.iterrows(), y_pred.iterrows()),
                             engine=self.engine, scheduler=self.scheduler,
                             n_jobs=self.n_jobs)
        return np.mean(losses)


class ModelA(RetinalGridMixin, BaseModel):
    """Scoreboard model"""

    def _set_default_params(self):
        """Sets default parameters of the scoreboard model"""
        # Current spread falls off exponentially from electrode center:
        self.rho = 100

    def get_params(self, deep=True):
        params = super(ModelA, self).get_params(deep=deep)
        params.update(rho=self.rho)
        return params

    def _calc_curr_map(self, electrode):
        """Calculates the current map for a specific electrode"""
        assert isinstance(electrode, six.string_types)
        # Find the `implant` index by trimming zeros in the electrode name
        ename = '%s%d' % (electrode[0], int(electrode[1:]))
        if not self.implant[ename]:
            raise ValueError("Electrode '%s' could not be found." % ename)
        r2 = (self.xret - self.implant[ename].x_center) ** 2
        r2 += (self.yret - self.implant[ename].y_center) ** 2
        cm = np.exp(-r2 / (2.0 * self.rho ** 2))
        return electrode, cm


class ModelB(CoordTrafoMixin, ModelA):
    """Scoreboard model with perspective transform"""

    pass
