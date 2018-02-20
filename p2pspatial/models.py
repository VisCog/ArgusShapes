import os
import abc
import six
import time
import pickle

import numpy as np
import pandas as pd

import pulse2percept.implants as p2pi
import pulse2percept.utils as p2pu

import scipy.stats as spst

import sklearn.base as sklb
import sklearn.exceptions as skle

import skimage

from .p2pspatial import *
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


class ScoreboardMixin(BaseModel):
    """Scoreboard model"""

    def _sets_default_params(self):
        """Sets default parameters of the scoreboard model"""
        # Current spread falls off exponentially from electrode center:
        super(ScoreboardMixin, self)._sets_default_params()
        self.rho = 100

    def get_params(self, deep=True):
        params = super(ScoreboardMixin, self).get_params(deep=deep)
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


class AxonMapMixin(BaseModel):
    """Axon map model"""

    def _sets_default_params(self):
        """Sets default parameters of the axon map model"""
        super(AxonMapMixin, self)._sets_default_params()
        self.rho = 100
        self.axlambda = 100
        # Set the (x,y) location of the optic disc:
        self.loc_od_x = 15.5
        self.loc_od_y = 1.5
        # Set parameters of the Jansonius model: Number of axons and number of
        # segments per axon can be overriden by the user:
        self.n_axons = 301
        self.axons_range = (-180, 180)
        # Number of sampling points along the radial axis(polar coordinates):
        self.n_ax_segments = 71
        # Lower and upper bounds for the radial position values(polar
        # coordinates):
        self.ax_segments_range = (3, 50)

    def get_params(self, deep=True):
        params = super(AxonMapMixin, self).get_params(deep=deep)
        params.update(rho=self.rho, axlambda=self.axlambda,
                      n_axons=self.n_axons, axons_range=self.axons_range,
                      n_ax_segments=self.n_ax_segments,
                      ax_segments_range=self.ax_segments_range,
                      loc_od_x=self.loc_od_x, loc_od_y=self.loc_od_y)
        return params

    def _jansonius2009(self, phi0, beta_sup=-1.9, beta_inf=0.5, eye='RE'):
        """Grows a single axon bundle based on the model by Jansonius (2009)

        This function generates the trajectory of a single nerve fiber bundle
        based on the mathematical model described in [1]_.

        Parameters
        ----------
        phi0: float
            Angular position of the axon at its starting point(polar
            coordinates, degrees). Must be within[-180, 180].
        beta_sup: float, optional, default: -1.9
            Scalar value for the superior retina(see Eq. 5, `\beta_s` in the
            paper).
        beta_inf: float, optional, default: 0.5
            Scalar value for the inferior retina(see Eq. 6, `\beta_i` in the
            paper.)

        Returns
        -------
        ax_pos: Nx2 array
            Returns a two - dimensional array of axonal positions, where
            ax_pos[0, :] contains the(x, y) coordinates of the axon segment
            closest to the optic disc, and aubsequent row indices move the axon
            away from the optic disc. Number of rows is at most `n_rho`, but
            might be smaller if the axon crosses the meridian.

        Notes
        -----
        The study did not include axons with phi0 in [-60, 60] deg.

        .. [1] N. M. Jansionus, J. Nevalainen, B. Selig, L.M. Zangwill, P.A.
               Sample, W. M. Budde, J. B. Jonas, W. A. Lagrèze, P. J.
               Airaksinen, R. Vonthein, L. A. Levin, J. Paetzold, and U.
               Schieferd, "A mathematical description of nerve fiber bundle
               trajectories and their variability in the human retina. Vision
               Research 49:2157-2163, 2009.
        """
        # Check for the location of the optic disc:
        loc_od = (self.loc_od_x, self.loc_od_y)
        if eye.upper() not in ['LE', 'RE']:
            e_s = "Unknown eye string '%s': Choose from 'LE', 'RE'." % eye
            raise ValueError(e_s)
        if eye.upper() == 'LE':
            # The Jansonius model doesn't know about left eyes: We invert the x
            # coordinate of the optic disc here, run the model, and then invert
            # all x coordinates of all axon fibers back.
            loc_od = (-loc_od[0], loc_od[1])
        if np.abs(phi0) > 180.0:
            raise ValueError('phi0 must be within [-180, 180].')
        if self.n_ax_segments < 1:
            raise ValueError('Number of radial sampling points must be >= 1.')
        if np.any(np.array(self.ax_segments_range) < 0):
            raise ValueError('ax_segments_range cannot be negative.')
        if self.ax_segments_range[0] > self.ax_segments_range[1]:
            raise ValueError('Lower bound on rho cannot be larger than the '
                             ' upper bound.')
        is_superior = phi0 > 0
        rho = np.linspace(*self.ax_segments_range, num=self.n_ax_segments)

        if is_superior:
            # Axon is in superior retina, compute `b` (real number) from Eq. 5:
            b = np.exp(beta_sup + 3.9 * np.tanh(-(phi0 - 121.0) / 14.0))
            # Equation 3, `c` a positive real number:
            c = 1.9 + 1.4 * np.tanh((phi0 - 121.0) / 14.0)
        else:
            # Axon is in inferior retina: compute `b` (real number) from Eq. 6:
            b = -np.exp(beta_inf + 1.5 * np.tanh(-(-phi0 - 90.0) / 25.0))
            # Equation 4, `c` a positive real number:
            c = 1.0 + 0.5 * np.tanh((-phi0 - 90.0) / 25.0)

        # Spiral as a function of `rho`:
        phi = phi0 + b * (rho - rho.min()) ** c
        # Convert to Cartesian coordinates:
        xprime = rho * np.cos(np.deg2rad(phi))
        yprime = rho * np.sin(np.deg2rad(phi))
        # Find the array elements where the axon crosses the meridian:
        if is_superior:
            # Find elements in inferior retina
            idx = np.where(yprime < 0)[0]
        else:
            # Find elements in superior retina
            idx = np.where(yprime > 0)[0]
        if idx.size:
            # Keep only up to first occurrence
            xprime = xprime[:idx[0]]
            yprime = yprime[:idx[0]]
        # Adjust coordinate system, having fovea=[0, 0] instead of
        # `loc_od`=[0, 0]:
        xmodel = xprime + loc_od[0]
        ymodel = yprime
        if loc_od[0] > 0:
            # If x-coordinate of optic disc is positive, use Appendix A
            idx = xprime > -loc_od[0]
        else:
            # Else we need to flip the sign
            idx = xprime < -loc_od[0]
        ymodel[idx] = yprime[idx] + loc_od[1] * (xmodel[idx] / loc_od[0]) ** 2
        # In a left eye, need to flip back x coordinates:
        if eye.upper() == 'LE':
            xmodel *= -1
        # Return as Nx2 array:
        return np.vstack((xmodel, ymodel)).T

    def _grows_axon_bundles(self):
        # Build the Jansonius model: Grow a number of axon bundles in all dirs:
        phi = np.linspace(*self.axons_range, num=self.n_axons)
        bundles = p2pu.parfor(self._jansonius2009, phi,
                              engine=self.engine, n_jobs=self.n_jobs,
                              scheduler=self.scheduler)
        assert len(bundles) == self.n_axons
        # Remove axon bundles outside the simulated area:
        bundles = list(filter(lambda x: (np.max(x[:, 0]) >= self.xrange[0] and
                                         np.min(x[:, 0]) <= self.xrange[1] and
                                         np.max(x[:, 1]) >= self.yrange[0] and
                                         np.min(x[:, 1]) <= self.yrange[1]),
                              bundles))
        # Remove short axon bundles:
        bundles = list(filter(lambda x: len(x) > 10, bundles))
        # Convert to um:
        bundles = [dva2ret(b) for b in bundles]
        return bundles

    def _finds_closest_axons(self, bundles):
        # For every axon segment, store the corresponding axon ID:
        axon_idx = [[idx] * len(ax) for idx, ax in enumerate(bundles)]
        axon_idx = [item for sublist in axon_idx for item in sublist]
        axon_idx = np.array(axon_idx, dtype=np.int32)
        # Build a long list of all axon segments - their corresponding axon IDs
        # is given by `axon_idx` above:
        flat_bundles = np.concatenate(bundles)
        # For every pixel on the grid, find the closest axon segment:
        closest_seg = [np.argmin((flat_bundles[:, 0] - x) ** 2 +
                                 (flat_bundles[:, 1] - y) ** 2)
                       for x, y in zip(self.xret.ravel(), self.yret.ravel())]
        # Look up the axon ID for every axon segment:
        closest_axon = axon_idx[closest_seg]
        return [bundles[n] for n in closest_axon]

    def _calcs_axon_contribution(self, axons):
        xyret = np.column_stack((self.xret.ravel(), self.yret.ravel()))
        axon_contrib = []
        for xy, bundle in zip(xyret, axons):
            idx = np.argmin((bundle[:, 0] - xy[0]) ** 2 +
                            (bundle[:, 1] - xy[1]) ** 2)
            # Cut off the part of the fiber that goes beyond the soma:
            axon = np.flipud(bundle[0: idx + 1, :])
            # Add the exact location of the soma:
            axon = np.insert(axon, 0, xy, axis=0)
            # For every axon segment, calculate distance from soma by summing
            # up the individual distances between neighboring axon segments
            # (by "walking along the axon"):
            d2 = np.cumsum(np.diff(axon[:, 0], axis=0) ** 2 +
                           np.diff(axon[:, 1], axis=0) ** 2)
            sensitivity = np.exp(-d2 / (2.0 * self.axlambda ** 2))
            axon_contrib.append(np.column_stack((axon[1:, :], sensitivity)))
        return axon_contrib

    def build_optic_fiber_layer(self):
        if self.implant.eye == 'LE':
            raise NotImplementedError
        need_axons = True
        # Check if math for Jansonius model has been done before:
        if os.path.isfile('axons.pickle'):
            params, axons = pickle.load(open('axons.pickle', 'rb'))
            for key, value in six.iteritems(params):
                if getattr(self, key) != value:
                    break
            need_axons = False
        # Build the Jansonius model: Grow a number of axon bundles in all dirs:
        if need_axons:
            bundles = self._grows_axon_bundles()
            axons = self._finds_closest_axons(bundles)
        # Calculate axon contributions (depends on axlambda):
        self.axon_contrib = self._calcs_axon_contribution(axons)
        # Pickle axons:
        params = {'loc_od_x': self.loc_od_x, 'loc_od_y': self.loc_od_y,
                  'n_axons': self.n_axons, 'axons_range': self.axons_range,
                  'n_ax_segments': self.n_ax_segments,
                  'ax_segments_range': self.ax_segments_range}
        pickle.dump((params, axons), open('axons.pickle', 'wb'))

    def _calcs_el_curr_map(self, electrode):
        """Calculates the current map for a specific electrode"""
        assert isinstance(electrode, six.string_types)
        if not self.implant[electrode]:
            raise ValueError("Electrode '%s' could not be found." % electrode)
        ecm = []
        for ax in self.axon_contrib:
            if ax.shape[0] == 0:
                ecm.append(0)
                continue
            r2 = (ax[:, 0] - self.implant[electrode].x_center) ** 2
            r2 += (ax[:, 1] - self.implant[electrode].y_center) ** 2
            curr = np.exp(-r2 / (2.0 * self.rho ** 2))
            act = np.multiply(curr, ax[:, 2])
            ecm.append(np.max(act))
        return np.array(ecm, dtype=float).reshape(self.xret.shape)


class RetinalCoordTrafoMixin(BaseModel):

    def _sets_default_params(self):
        super(RetinalCoordTrafoMixin, self)._sets_default_params()
        # We will be simulating an x,y patch of the visual field (min, max) in
        # degrees of visual angle, at a given spatial resolution (step size):
        self.xrange = (-30, 30)  # dva
        self.yrange = (-20, 20)  # dva
        self.xystep = 0.2  # dva

    def get_params(self, deep=True):
        params = super(RetinalCoordTrafoMixin, self).get_params(deep=deep)
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
        return dva2ret(x), dva2ret(y)

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


class RetinalGridMixin(BaseModel):

    def _sets_default_params(self):
        super(RetinalGridMixin, self)._sets_default_params()
        # We will be simulating an x,y patch of the visual field (min, max) in
        # degrees of visual angle, at a given spatial resolution (step size):
        self.xrange = (-30, 30)  # dva
        self.yrange = (-20, 20)  # dva
        self.xystep = 0.2  # dva

    def get_params(self, deep=True):
        params = super(RetinalGridMixin, self).get_params(deep=deep)
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

        self.xret = dva2ret(xdva)
        self.yret = dva2ret(ydva)


class ImageMomentsLossMixin(BaseModel):

    def _sets_default_params(self):
        super(ImageMomentsLossMixin, self)._sets_default_params()
        self.greater_is_better = False

    def get_params(self, deep=True):
        params = super(ImageMomentsLossMixin, self).get_params(deep=deep)
        params.update(greater_is_better=self.greater_is_better)
        return params

    def _predicts_target_values(self, img):
        if not isinstance(img, np.ndarray):
            raise TypeError("`img` must be a NumPy array.")
        # The image has already been thresholded using `self.img_thresh`:
        props = imgproc.get_region_props(img, thresh=0.5)
        if props is None:
            return {'area': 0, 'orientation': 0}
        return {'area': props.area, 'orientation': props.orientation}

    def _scores_props(self, ytyp, w_area=0.001, w_orient=1):
        (_, yt), (_, yp) = ytyp
        if not isinstance(yt, pd.core.series.Series):
            raise TypeError("`yt` must be a pandas Series")
        if not isinstance(yp, pd.core.series.Series):
            raise TypeError("`yp` must be a pandas Series")
        # Extract props from ground-truth image:
        img_yt = imgproc.center_phosphene(skimage.img_as_float(yt['image']))
        props_yt = imgproc.get_region_props(img_yt, thresh=0.5)
        assert props_yt is not None
        # Calculate area error:
        err_area = (props_yt.area - yp['area']) ** 2
        # Calculate orient error (degrees, < 180 deg):
        err_orient = np.abs(props_yt.orientation - yp['orientation'])
        err_orient = np.rad2deg(err_orient)
        if err_orient > 180:
            err_orient = 360 - err_orient
        err_orient = err_orient ** 2
        # Return weighted sum of area and orient errors:
        return w_area * err_area + w_orient * err_orient

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
        losses = p2pu.parfor(self._scores_props,
                             zip(y.iterrows(), y_pred.iterrows()),
                             engine=self.engine, scheduler=self.scheduler,
                             n_jobs=self.n_jobs)
        return np.mean(losses)


class SRDLossMixin(BaseModel):
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
        super(SRDLossMixin, self)._sets_default_params()
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
        params = super(SRDLossMixin, self).get_params(deep=deep)
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


class ModelA(SRDLossMixin, RetinalGridMixin, ScoreboardMixin):
    """Scoreboard model with SRD loss"""
    pass


class ModelB(SRDLossMixin, RetinalCoordTrafoMixin, ScoreboardMixin):
    """Scoreboard model with perspective transform and SRD loss"""
    pass


class ModelC(SRDLossMixin, RetinalGridMixin, AxonMapMixin):
    """Axon map model with SRD loss"""
    pass


class ModelD(SRDLossMixin, RetinalCoordTrafoMixin, AxonMapMixin):
    """Axon map model with perspective transform and SRD loss"""
    pass


class ModelE(ImageMomentsLossMixin, RetinalGridMixin, AxonMapMixin):
    """Axon map model with SRD loss"""
    pass


class ModelF(ImageMomentsLossMixin, RetinalCoordTrafoMixin, AxonMapMixin):
    """Axon map model with perspective transform and SRD loss"""
    pass
