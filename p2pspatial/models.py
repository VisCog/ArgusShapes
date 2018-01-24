import numpy as np
import pandas as pd
import abc
import six
import time

import pulse2percept.retina as p2pr
import pulse2percept.implants as p2pi
import pulse2percept.utils as p2pu

import scipy.stats as spst
import scipy.spatial as spsp

import sklearn.base as sklb
import sklearn.exceptions as skle

from . import imgproc

try:
    # Python 2
    reduce(lambda x, y: x + y, [1, 2, 3])
except NameError:
    # Python 3
    from functools import reduce


@six.add_metaclass(abc.ABCMeta)
class BaseModel(sklb.BaseEstimator):

    def __init__(self, **kwargs):
        # The following parameters serve as default values and can be
        # overwritten via `kwargs`

        # The model operates on an electrode array, but we cannot instantiate
        # it here since we might pass the array's location as search params.
        # So we save the array type and set some default values for its
        # location:
        self.implant_type = p2pi.ArgusII
        self.implant_x = 0
        self.implant_y = 0
        self.implant_rot = 0

        # Current maps are thresholded to produce a binary image:
        self.img_thresh = 0.1

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

        # Additional parameters can be set using ``_sets_default_params``
        self._sets_default_params()
        self.set_params(**kwargs)

    def get_params(self, deep=True):
        """Returns all params that can be set on-the-fly via 'set_params'"""
        return {'implant_type': self.implant_type,
                'implant_x': self.implant_x,
                'implant_y': self.implant_y,
                'implant_rot': self.implant_rot,
                'img_thresh': self.img_thresh,
                'engine': self.engine,
                'scheduler': self.scheduler,
                'n_jobs': self.n_jobs}

    def _sets_default_params(self):
        """Derived classes can set additional default parameters here"""
        pass

    def _ename(self, electrode):
        """Returns electrode name with zeros trimmed"""
        return '%s%d' % (electrode[0], int(electrode[1:]))

    @abc.abstractmethod
    def build_ganglion_cell_layer(self):
        """Builds the ganglion cell layer"""
        raise NotImplementedError

    def build_optic_fiber_layer(self):
        """Builds the optic fiber layer"""
        pass

    @abc.abstractmethod
    def _calcs_el_curr_map(self, electrode):
        """Must return a tuple `current_map`"""
        raise NotImplementedError

    def calc_curr_map(self, X):
        # Calculate current maps only if necessary:
        # - Get a list of all electrodes for which we already have a curr map,
        #   but trim the zeros before the number, e.g. 'A01' => 'A1'
        has_el = set([self._ename(k) for k in self._curr_map.keys()])
        # - Compare with electrodes in `X` to find the ones we don't have,
        #   but trim the zeros:
        wants_el = set([self._ename(e) for e in set(X.electrode)])
        needs_el = wants_el.difference(has_el)
        # - Calculate the current maps for the missing electrodes:
        curr_map = p2pu.parfor(self._calcs_el_curr_map, needs_el,
                               engine=self.engine, scheduler=self.scheduler,
                               n_jobs=self.n_jobs)
        # - Store the new current maps:
        for key, cm in zip(needs_el, curr_map):
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
        if not isinstance(self.implant_type, type):
            raise TypeError(("'implant_type' must be a type, not "
                             "'%s'." % type(self.implant_type)))
        self.implant = self.implant_type(x_center=self.implant_x,
                                         y_center=self.implant_y,
                                         rot=self.implant_rot)
        # Build the ganglion cell layer:
        self.build_ganglion_cell_layer()
        # Build the ganglion axon layer (optional):
        self.build_optic_fiber_layer()
        # Calculate current spread for every electrode in `X`:
        self.calc_curr_map(X)
        # Inform the object that is has been fitted:
        self._is_fitted = True
        return self

    @abc.abstractmethod
    def _predicts_target_values(self, img):
        """Must return a dict of predicted values, e.g {'image': img}"""
        raise NotImplementedError

    def _predicts(self, Xrow):
        """Predicts a single data point"""
        _, row = Xrow
        assert isinstance(row, pd.core.series.Series)
        # Calculate current map with method from derived class:
        curr_map = self._curr_map[self._ename(row['electrode'])]
        if not isinstance(curr_map, np.ndarray):
            raise TypeError(("Method '_curr_map' must return a np.ndarray, "
                             "not '%s'." % type(curr_map)))
        # Rescale output if specified:
        out_shape = None
        if hasattr(row, 'img_shape'):
            out_shape = row['img_shape']
        elif hasattr(row, 'image'):
            out_shape = row['image'].shape
        # Apply threshold to arrive at binarized image:
        assert hasattr(self, 'img_thresh')
        img = imgproc.get_thresholded_image(curr_map, thresh=self.img_thresh,
                                            out_shape=out_shape)
        return self._predicts_target_values(img)

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

        # Predict percept
        y_pred = p2pu.parfor(self._predicts, X.iterrows(),
                             engine=self.engine, scheduler=self.scheduler,
                             n_jobs=self.n_jobs)

        # Convert to DataFrame, preserving the index of `X` (otherwise
        # subtraction in the scoring function produces nan)
        return pd.DataFrame(y_pred, index=X.index)

    @abc.abstractmethod
    def score(self, X, y, sample_weight=None):
        """Scores the model"""
        raise NotImplementedError


class ScoreboardModel(BaseModel):
    """Scoreboard model"""

    def _sets_default_params(self):
        """Sets default parameters of the scoreboard model"""
        # Current spread falls off exponentially from electrode center:
        super(ScoreboardModel, self)._sets_default_params()
        self.rho = 100

    def get_params(self, deep=True):
        params = super(ScoreboardModel, self).get_params(deep=deep)
        params.update(rho=self.rho)
        return params

    def _calcs_el_curr_map(self, electrode):
        """Calculates the current map for a specific electrode"""
        assert isinstance(electrode, six.string_types)
        if not self.implant[electrode]:
            raise ValueError("Electrode '%s' could not be found." % electrode)
        r2 = (self.xret - self.implant[electrode].x_center) ** 2
        r2 += (self.yret - self.implant[electrode].y_center) ** 2
        cm = np.exp(-r2 / (2.0 * self.rho ** 2))
        return cm


class AxonMapModel(BaseModel):
    """Axon map model"""

    def _sets_default_params(self):
        """Sets default parameters of the axon map model"""
        super(AxonMapModel, self)._sets_default_params()
        self.rho = 100
        self.axlambda = 100
        # Set the (x,y) location of the optic disc:
        self.loc_od_x = 15.5
        self.loc_od_y = 1.5
        # Set parameters of the Jansonius model: Number of axons and number of
        # segments per axon can be overriden by the user:
        self.n_axons = 301
        self.n_ax_segments = 71
        self._phi_range = (-180, 180)
        self._rho_range = (3, 50)

    def get_params(self, deep=True):
        params = super(AxonMapModel, self).get_params(deep=deep)
        params.update(rho=self.rho, axlambda=self.axlambda,
                      n_axons=self.n_axons, n_ax_segments=self.n_ax_segments,
                      loc_od_x=self.loc_od_x, loc_od_y=self.loc_od_y)
        return params

    def _grows_axon_bundles(self):
        # Build the Jansonius model: Grow a number of axon bundles in all dirs:
        phi = np.linspace(*self._phi_range, num=self.n_axons)
        jans_kwargs = {'n_rho': self.n_ax_segments, 'eye': self.implant.eye,
                       'rho_range': self._rho_range,
                       'loc_od': (self.loc_od_x, self.loc_od_y)}
        bundles = p2pu.parfor(p2pr.jansonius2009, phi, func_kwargs=jans_kwargs,
                              engine=self.engine, n_jobs=self.n_jobs,
                              scheduler=self.scheduler)
        # Remove axon bundles outside the simulated area:
        bundles = list(filter(lambda x: (np.max(x[:, 0]) >= self.xrange[0] and
                                         np.min(x[:, 0]) <= self.xrange[1] and
                                         np.max(x[:, 1]) >= self.yrange[0] and
                                         np.min(x[:, 1]) <= self.yrange[1]),
                              bundles))
        # Remove short axon bundles:
        bundles = list(filter(lambda x: len(x) > 10, bundles))
        # Convert to um:
        bundles = [p2pr.dva2ret(b) for b in bundles]
        return bundles

    def _finds_closest_axons(self, bundles):
        # Build a KDTree from all axon segment locations: This allows for a
        # quick nearest-neighbor lookup. Need to store the axon ID for every
        # segment:
        axon_idx = [idx * np.ones(len(ax)) for idx, ax in enumerate(bundles)]
        axon_idx = reduce(lambda x, y: np.concatenate((x, y)), axon_idx)
        axon_idx = np.array(axon_idx, dtype=np.int32)
        # Then build the KDTree with all axon segment locations:
        tree = spsp.cKDTree(reduce(lambda x, y: np.vstack((x, y)), bundles))
        # Find the closest axon segment for every grid location:
        xyret = np.column_stack((self.xret.ravel(), self.yret.ravel()))
        nearest_segment = [tree.query(xy)[1] for xy in xyret]
        # Look up the axon ID for every axon segment:
        nearest_axon = axon_idx[nearest_segment]
        axons = []
        for xy, n in zip(xyret, nearest_axon):
            bundle = bundles[n]
            ax_tree = spsp.cKDTree(bundle)
            _, idx = ax_tree.query(xy)
            # Cut off the part of the fiber that goes beyond the soma:
            axon = bundle[idx:0:-1, :]
            # Add the exact location of the soma:
            axon = np.insert(axon, 0, xy, axis=0)
            # For every axon segment, calculate distance from soma by summing
            # up the individual distances between neighboring axon segments
            # (by "walking along the axon"):
            d2 = np.cumsum(np.diff(axon[:, 0]) ** 2 + np.diff(axon[:, 1]) ** 2)
            sensitivity = np.exp(-d2 / (2.0 * self.axlambda ** 2))
            axons.append(np.column_stack((axon[1:, :], sensitivity)))
        return axons

    def build_optic_fiber_layer(self):
        if self.implant.eye == 'LE':
            raise NotImplementedError
        # Build the Jansonius model: Grow a number of axon bundles in all dirs:
        bundles = self._grows_axon_bundles()
        self.axons = self._finds_closest_axons(bundles)

    def _calcs_el_curr_map(self, electrode):
        """Calculates the current map for a specific electrode"""
        assert isinstance(electrode, six.string_types)
        if not self.implant[electrode]:
            raise ValueError("Electrode '%s' could not be found." % electrode)
        ecm = []
        for ax in self.axons:
            if ax.shape[0] == 0:
                ecm.append(0)
                continue
            r2 = (ax[:, 0] - self.implant[electrode].x_center) ** 2
            r2 += (ax[:, 1] - self.implant[electrode].y_center) ** 2
            curr = np.exp(-r2 / (2.0 * self.rho ** 2))
            act = np.multiply(curr, ax[:, 2])
            ecm.append(np.max(act))
        return np.array(ecm, dtype=float).reshape(self.xret.shape)


class RetinalCoordTrafo(BaseModel):

    def _sets_default_params(self):
        super(RetinalCoordTrafo, self)._sets_default_params()
        # We will be simulating an x,y patch of the visual field (min, max) in
        # degrees of visual angle, at a given spatial resolution (step size):
        self.xrange = (-30, 30)  # dva
        self.yrange = (-20, 20)  # dva
        self.xystep = 0.2  # dva

    def get_params(self, deep=True):
        params = super(RetinalCoordTrafo, self).get_params(deep=deep)
        params.update(xrange=self.xrange, yrange=self.yrange,
                      xystep=self.xystep)
        return params

    @staticmethod
    def _cart2pol(x, y):
        theta = np.arctan2(y, x)
        rho = np.hypot(x, y)
        return theta, rho

    @staticmethod
    def _pol2cart(theta, rho):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y

    @staticmethod
    def _watson_displacement(r, meridian='temporal'):
        if (not isinstance(meridian, (np.ndarray, six.string_types)) or
                not np.all([m in ['temporal', 'nasal']
                            for m in np.array([meridian]).ravel()])):
            print(meridian)
            raise ValueError("'meridian' must be either 'temporal' or 'nasal'")
        alpha = np.where(meridian == 'temporal', 1.8938, 2.4607)
        beta = np.where(meridian == 'temporal', 2.4598, 1.7463)
        gamma = np.where(meridian == 'temporal', 0.91565, 0.77754)
        delta = np.where(meridian == 'temporal', 14.904, 15.111)
        mu = np.where(meridian == 'temporal', -0.09386, -0.15933)
        scale = np.where(meridian == 'temporal', 12.0, 10.0)

        rmubeta = (np.abs(r) - mu) / beta
        numer = delta * gamma * np.exp(-rmubeta ** gamma)
        numer *= rmubeta ** (alpha * gamma - 1)
        denom = beta * spst.gamma.pdf(alpha, 5)

        return numer / denom / scale

    def _displaces_rgc(self, xy, eye='RE'):
        if not isinstance(xy, np.ndarray) or xy.shape[1] != 2:
            raise ValueError("'xy' must be a Nx2 NumPy array.")
        if eye == 'LE':
            # Let's not think about eyes right now...
            raise NotImplementedError

        # Convert x, y (dva) into polar coordinates
        theta, rho_dva = self._cart2pol(xy[:, 0], xy[:, 1])

        # Add displacement
        meridian = np.where(xy[:, 0] < 0, 'temporal', 'nasal')
        rho_dva += self._watson_displacement(rho_dva, meridian=meridian)

        # Convert back to x, y (dva)
        x, y = self._pol2cart(theta, rho_dva)

        # Convert to retinal coords
        return p2pr.dva2ret(x), p2pr.dva2ret(y)

    def build_ganglion_cell_layer(self):
        # Build the grid from `x_range`, `y_range`:
        nx = int(np.ceil((np.diff(self.xrange) + 1) / self.xystep))
        ny = int(np.ceil((np.diff(self.yrange) + 1) / self.xystep))
        xdva, ydva = np.meshgrid(np.linspace(*self.xrange, num=nx),
                                 np.linspace(*self.yrange, num=ny),
                                 indexing='xy')

        # Convert dva to retinal coordinates
        xydva = np.column_stack((xdva.ravel(), ydva.ravel()))
        xret, yret = self._displaces_rgc(xydva)
        self.xret = xret.reshape(xdva.shape)
        self.yret = yret.reshape(ydva.shape)


class RetinalGrid(BaseModel):

    def _sets_default_params(self):
        super(RetinalGrid, self)._sets_default_params()
        # We will be simulating an x,y patch of the visual field (min, max) in
        # degrees of visual angle, at a given spatial resolution (step size):
        self.xrange = (-30, 30)  # dva
        self.yrange = (-20, 20)  # dva
        self.xystep = 0.2  # dva

    def get_params(self, deep=True):
        params = super(RetinalGrid, self).get_params(deep=deep)
        params.update(xrange=self.xrange, yrange=self.yrange,
                      xystep=self.xystep)
        return params

    def build_ganglion_cell_layer(self):
        # Build the grid from `x_range`, `y_range`:
        nx = int(np.ceil((np.diff(self.xrange) + 1) / self.xystep))
        ny = int(np.ceil((np.diff(self.yrange) + 1) / self.xystep))
        xdva, ydva = np.meshgrid(np.linspace(*self.xrange, num=nx),
                                 np.linspace(*self.yrange, num=ny),
                                 indexing='xy')

        self.xret = p2pr.dva2ret(xdva)
        self.yret = p2pr.dva2ret(ydva)


class ImageMomentsLoss(BaseModel):

    def _sets_default_params(self):
        super(ImageMomentsLoss, self)._sets_default_params()
        self.greater_is_better = False

    def get_params(self, deep=True):
        params = super(ImageMomentsLoss, self).get_params(deep=deep)
        params.update(greater_is_better=self.greater_is_better)
        return params

    def _predicts_target_values(self, img):
        area = 0
        orientation = 0
        major_axis_length = 0
        minor_axis_length = 0
        return {'area': area,
                'orientation': orientation,
                'major_axis_length': major_axis_length,
                'minor_axis_length': minor_axis_length}

    def score(self, X, y, sample_weight=None):
        return 100


class SRDLoss(BaseModel):
    """Scale-Rotation-Dice (SRD) loss

    This class provides a ``score`` method that calculates a loss in [0, 100]
    made of three components:
    - the scaling factor needed to match the area of predicted and target
      percept
    - the rotation angle needed to achieve the greatest dice coefficient
      between predicted and target percept
    - the dice coefficient between (adjusted) predicted and target percept
    """

    def _sets_default_params(self):
        super(SRDLoss, self)._sets_default_params()
        # The new scoring function is actually a loss function, so that
        # greater values do *not* imply that the estimator is better (required
        # for ParticleSwarmOptimizer)
        self.greater_is_better = False
        # By default, the loss function will return values in [0, 100], scoring
        # the scaling factor, rotation angle, and dice coefficient of precition
        # vs ground truth with the following weights:
        self.w_scale = 34
        self.w_rot = 33
        self.w_dice = 34

    def get_params(self, deep=True):
        params = super(SRDLoss, self).get_params(deep=deep)
        params.update(greater_is_better=self.greater_is_better,
                      w_scale=self.w_scale, w_rot=self.w_rot,
                      w_dice=self.w_dice)
        return params

    def _predicts_target_values(self, img):
        return {'image': img}

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
        losses = p2pu.parfor(imgproc.srd_loss,
                             zip(y.iterrows(), y_pred.iterrows()),
                             func_kwargs={'w_scale': self.w_scale,
                                          'w_rot': self.w_rot,
                                          'w_dice': self.w_dice},
                             engine=self.engine, scheduler=self.scheduler,
                             n_jobs=self.n_jobs)
        return np.mean(losses)


class ModelA(SRDLoss, RetinalGrid, ScoreboardModel):
    """Scoreboard model with SRD loss"""
    pass


class ModelB(SRDLoss, RetinalCoordTrafo, ScoreboardModel):
    """Scoreboard model with perspective transform and SRD loss"""
    pass


class ModelC(SRDLoss, RetinalGrid, AxonMapModel):
    """Axon map model with SRD loss"""
    pass


class ModelD(SRDLoss, RetinalCoordTrafo, AxonMapModel):
    """Axon map model with perspective transform and SRD loss"""
    pass
