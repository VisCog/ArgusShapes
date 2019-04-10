from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import six

import sklearn.exceptions as skle

import pulse2percept.implants as p2pi
import pulse2percept.retina as p2pr
from .. import models as m
from .. import utils

import numpy.testing as npt
import pytest


def get_dummy_data(nrows=3, img_in_shape=(10, 10), img_out_shape=(10, 10)):
    """Helper function for test suite"""
    # Choose from the following electrodes
    electrodes = ['A01', 'A2', 'A03', 'A3', 'A04', 'B01', 'B2']
    data = []
    for _ in range(nrows):
        img = np.random.rand(np.prod(img_in_shape)).reshape(img_in_shape)
        el = np.random.randint(len(electrodes))
        data.append({'electrode': electrodes[el], 'image': img,
                     'img_shape': img_out_shape})
    # Shuffle row indices
    idx = np.arange(nrows)
    np.random.shuffle(idx)
    # Create data frame
    X = pd.DataFrame(data, index=idx)
    return X


class ValidBaseModel(m.BaseModel):
    """A class that implements all abstract methods of BaseModel"""

    def _calcs_el_curr_map(self, electrode):
        return np.array([[0]])

    def _predicts_target_values(self, row):
        return row

    def _predicts(self, Xrow):
        """Returns the input (a DataFrame row, without its index)"""
        _, row = Xrow
        return self._predicts_target_values(row)

    def build_ganglion_cell_layer(self):
        self.xret = [[0]]
        self.yret = [[0]]

    def score(self, X, y, sample_weight=None):
        if not isinstance(X, pd.core.frame.DataFrame):
            raise TypeError("'X' must be a pandas DataFrame, not %s" % type(X))
        if not isinstance(y, pd.core.frame.DataFrame):
            raise TypeError("'y' must be a pandas DataFrame, not %s" % type(y))
        self.predict(X)
        return 0


class ValidScoreboardModel(m.ShapeLossMixin, m.RetinalGridMixin,
                           m.ScoreboardMixin, m.BaseModel):
    """A class that implements all abstract methods of BaseModel"""

    def _predicts_target_values(self, row):
        return row

    def _predicts(self, Xrow):
        """Returns the input (a DataFrame row, without its index)"""
        _, row = Xrow
        return self._predicts_target_values(row)


class ValidAxonMapModel(m.ShapeLossMixin, m.RetinalGridMixin, m.AxonMapMixin,
                        m.BaseModel):
    """A class that implements all abstract methods of AxonMapModel"""

    def build_optic_fiber_layer(self):
        self.axon_contrib = [np.zeros((1, 3))] * int(np.prod(self.xret.shape))

    def _predicts_target_values(self, row):
        return row

    def _predicts(self, Xrow):
        """Returns the input (a DataFrame row, without its index)"""
        _, row = Xrow
        return self._predicts_target_values(row)


class ValidRetinalCoordTrafo(m.ShapeLossMixin, m.RetinalCoordTrafoMixin,
                             m.BaseModel):
    """A class that implements all abstract methods of BaseModel"""

    def _calcs_el_curr_map(self, electrode):
        return np.array([[0]])

    def _predicts_target_values(self, row):
        return row

    def _predicts(self, Xrow):
        """Returns the input (a DataFrame row, without its index)"""
        _, row = Xrow
        return self._predicts_target_values(row)


class ValidRetinalGrid(m.ShapeLossMixin, m.RetinalGridMixin, m.BaseModel):
    """A class that implements all abstract methods of BaseModel"""

    def _calcs_el_curr_map(self, electrode):
        return np.array([[0]])

    def _predicts_target_values(self, row):
        return row

    def _predicts(self, Xrow):
        """Returns the input (a DataFrame row, without its index)"""
        _, row = Xrow
        return self._predicts_target_values(row)


def test_RetinalCoordTrafo():
    params = {'xrange': (-2, 2), 'yrange': (-1, 1), 'xystep': 1}
    trafo = ValidRetinalCoordTrafo()
    print(trafo.get_params())
    for p in params:
        npt.assert_equal(hasattr(trafo, p), True)

    # Two different ways to set the parameters:
    trafo.set_params(**params)
    for k, v in six.iteritems(params):
        npt.assert_equal(getattr(trafo, k), v)
    trafo = ValidRetinalCoordTrafo(**params)
    for k, v in six.iteritems(params):
        npt.assert_equal(getattr(trafo, k), v)


def test_RetinalCoordTrafo__watson_displacement():
    trafo = ValidRetinalCoordTrafo()
    with pytest.raises(ValueError):
        trafo._watson_displacement(0, meridian='invalid')
    npt.assert_almost_equal(trafo._watson_displacement(0), 0.4957506)
    npt.assert_almost_equal(trafo._watson_displacement(100), 0)

    # Check the max of the displacement function for the temporal meridian:
    radii = np.linspace(0, 30, 100)
    all_displace = trafo._watson_displacement(radii, meridian='temporal')
    npt.assert_almost_equal(np.max(all_displace), 2.153532)
    npt.assert_almost_equal(radii[np.argmax(all_displace)], 1.8181818)

    # Check the max of the displacement function for the nasal meridian:
    all_displace = trafo._watson_displacement(radii, meridian='nasal')
    npt.assert_almost_equal(np.max(all_displace), 1.9228664)
    npt.assert_almost_equal(radii[np.argmax(all_displace)], 2.1212121)


def test_RetinalCoordTrafo__displaces_rgc():
    trafo = ValidRetinalCoordTrafo()
    for xy in [1, [1, 2], np.zeros((10, 3))]:
        with pytest.raises(ValueError):
            trafo._displaces_rgc(xy)

    npt.assert_almost_equal(
        trafo._displaces_rgc(np.array([0, 0]).reshape((1, 2))),
        ([p2pr.dva2ret(trafo._watson_displacement(0))], [0]),
        decimal=1
    )


def test_RetinalCoordTrafo_build_ganglion_cell_layer():
    trafo = ValidRetinalCoordTrafo()
    trafo.xrange = (-2, 2)
    trafo.yrange = (-1, 1)
    trafo.xystep = 1
    trafo.build_ganglion_cell_layer()


def test_RetinalGrid():
    params = {'xrange': (-2, 2), 'yrange': (-1, 1), 'xystep': 1}
    trafo = ValidRetinalGrid()
    for p in params:
        npt.assert_equal(hasattr(trafo, p), True)

    # Two different ways to set the parameters:
    trafo.set_params(**params)
    for k, v in six.iteritems(params):
        npt.assert_equal(getattr(trafo, k), v)
    trafo = ValidRetinalGrid(**params)
    for k, v in six.iteritems(params):
        npt.assert_equal(getattr(trafo, k), v)

    trafo.build_ganglion_cell_layer()

    # Make sure shape is right
    npt.assert_equal(trafo.xret.shape, (3, 5))
    npt.assert_equal(trafo.xret.shape, (3, 5))

    # Make sure transformation is right
    npt.assert_almost_equal(trafo.xret[1, 2], 0)
    npt.assert_almost_equal(trafo.yret[1, 2], 0)


def test_BaseModel___init__():
    # We can overwrite default param values if they are in ``get_params``:
    model = ValidBaseModel()
    model_params = model.get_params()
    for key, value in six.iteritems(model_params):
        npt.assert_equal(getattr(model, key), value)
        set_param = {key: 1234}
        model.set_params(**set_param)
        npt.assert_equal(getattr(model, key), 1234)

        newmodel = ValidBaseModel(**set_param)
        npt.assert_equal(getattr(newmodel, key), 1234)

    # But setting parameters that are not in ``get_params`` is not allowed:
    for forbidden_key in ['_is_fitted', 'greater_is_better', 'invalid',
                          'xrange', 'w_scale']:
        print(forbidden_key)
        set_param = {forbidden_key: 1234}
        with pytest.raises(ValueError):
            model.set_params(**set_param)
        with pytest.raises(ValueError):
            ValidBaseModel(**set_param)

    # You could bypass this by using model.key = value, but this only works for
    # attributes that already exist. Adding new ones is prohibited after the
    # constructor:
    with pytest.raises(ValueError):
        model.newvariable = 1234


def test_BaseModel_fit():
    # Create a valid DataFrame
    X = get_dummy_data(nrows=3)

    # Model must be fitted first thing
    model = ValidBaseModel(engine='serial')
    npt.assert_equal(model._is_fitted, False)
    model.fit(X)
    npt.assert_equal(model._is_fitted, True)
    npt.assert_equal(hasattr(model, 'implant'), True)
    npt.assert_equal(hasattr(model, 'implant_type'), True)
    npt.assert_equal(isinstance(model.implant, model.implant_type), True)

    # `fit` only accepts DataFrame
    for XX in [42, [3.3, 1.1], np.array([[0]]), {'img': [[2]]}]:
        with pytest.raises(TypeError):
            model.fit(XX)
    yy = pd.DataFrame([{'img': [[2]]}, {'img': [[4]]}], index=[0, 1])
    for XX in [42, [3.3, 1.1], {'img': [[2]]}]:
        with pytest.raises(TypeError):
            model.fit(XX, y=yy)
    for yy in [42, [3.3, 1.1], {'img': [[2]]}]:
        with pytest.raises(TypeError):
            model.fit(X, y=yy)

    # We must pass an implant type, not an implant instance
    with pytest.raises(TypeError):
        model = ValidBaseModel(engine='serial', implant_type=p2pi.ArgusII())
        model.fit(X)
    with pytest.raises(TypeError):
        model = ValidBaseModel(engine='serial')
        model.set_params(implant_type=p2pi.ArgusII())
        model.fit(X)
    with pytest.raises(TypeError):
        model = ValidBaseModel(engine='serial')
        model.fit(X, implant_type=p2pi.ArgusII())

    # Implant rotation must be in radians:
    with pytest.raises(ValueError):
        model = ValidBaseModel(implant_rot=180)
        model.fit(X)

    # `fit_params` must take effect
    model = ValidBaseModel(engine='serial')
    model_params = model.get_params()
    for key, value in six.iteritems(model_params):
        npt.assert_equal(getattr(model, key), value)
        if isinstance(value, (int, float)):
            set_param = {key: 0.1234}
        elif isinstance(value, (list, set, tuple, np.ndarray)):
            set_param = {key: np.array([0, 0])}
        else:
            continue
        model.fit(X, **set_param)
        npt.assert_equal(getattr(model, key), set_param[key])


def test_BaseModel_calc_curr_map():
    X = get_dummy_data(nrows=10)
    model = ValidBaseModel(engine='serial')

    # Make sure we calculate only the current maps we need: We start with
    # nothing, but add a new current map step-by-step
    npt.assert_equal(model._curr_map, {})
    for idx in np.arange(1, len(X)):
        # Grab a slice from the data array
        slc = X.iloc[:idx, :]
        # Trim the zeros in the electrode names, e.g. 'A01' => 'A1':
        electrodes = set(['%s%d' % (e[0], int(e[1:])) for e in slc.electrode])
        # Fit the model on the data slice:
        model.fit(slc)
        npt.assert_equal(model._is_fitted, True)
        # Make sure all electrodes in the slice are also in the current map:
        npt.assert_equal(set(model._curr_map.keys()), electrodes)
        model._is_fitted = False

    # Additional calls to ``calc_curr_map`` should not make a difference:
    electrodes = set(['%s%d' % (e[0], int(e[1:])) for e in X.electrode])
    for _ in range(3):
        model.calc_curr_map(X)
        npt.assert_equal(set(model._curr_map.keys()), electrodes)


def test_BaseModel_predict():
    # Model must be fitted first
    model = ValidBaseModel(engine='serial')
    out_shape = (10, 18)
    X = get_dummy_data(nrows=3, img_out_shape=out_shape)
    with pytest.raises(skle.NotFittedError):
        model.predict(X)

    # But then must pass through ``predict`` just fine
    model.fit(X)
    y_pred = model.predict(X)
    npt.assert_equal(np.allclose(X.index, y_pred.index), True)
    # ValidBaseModel returns input, so that predict(X) == X
    npt.assert_equal(y_pred.equals(X), True)

    # `predict` only accepts DataFrame
    for XX in [42, [3.3, 1.1], np.array([[0]]), {'img': [[2]]}]:
        with pytest.raises(TypeError):
            model.predict(XX)

    # Instead of target values, we can predict images:
    for _, row in X.iterrows():
        el = model._ename(row['electrode'])
        npt.assert_almost_equal(model.predict_image(el), model._curr_map[el])
    with pytest.raises(TypeError):
        model.predict_image(X.loc[0, :])


def test_BaseModel_score():
    # Model must be fitted first
    model = ValidBaseModel(engine='serial')
    X = get_dummy_data(nrows=3)
    with pytest.raises(skle.NotFittedError):
        model.score(X, X)

    # But then must pass through ``score`` just fine: ValidBaseModel returns
    # input, so that predict(X) == X, and score(X, X) == 0
    model.fit(X)
    npt.assert_almost_equal(model.score(X, X), 0.0)


def test_ScoreboardModel():
    # ScoreboardModel automatically sets `rho`:
    X = get_dummy_data(nrows=10)
    model = ValidScoreboardModel(implant_type=p2pi.ArgusII, engine='serial')
    npt.assert_equal(hasattr(model, 'rho'), True)

    # ScoreboardModel uses the SRD loss, should have `greater_is_better` set to
    # False
    npt.assert_equal(hasattr(model, 'greater_is_better'), True)
    npt.assert_equal(model.greater_is_better, False)

    # User can set `rho`:
    model.set_params(rho=123)
    npt.assert_equal(model.rho, 123)
    model.fit(X, rho=987)
    npt.assert_equal(model.rho, 987)

    # Some electrodes in the test set might not be in the train set:
    X.loc[0, 'electrode'] = 'F09'
    model.predict(X)

    for electrode, cm in six.iteritems(model._curr_map):
        # Current maps must be in [0, 1]
        npt.assert_almost_equal(cm.min(), 0, decimal=3)
        npt.assert_almost_equal(cm.max(), 1, decimal=3)
        # All phosphenes are Gaussian blobs with the max under the electrode:
        r2 = (model.xret - model.implant[electrode].x_center) ** 2
        r2 += (model.yret - model.implant[electrode].y_center) ** 2
        npt.assert_almost_equal(cm.ravel()[np.argmin(r2)], 1, decimal=3)


def test_AxonMapModel():
    # AxonMapModel automatically sets a number of parameters:
    X = get_dummy_data(nrows=10)
    model = ValidAxonMapModel(implant_type=p2pi.ArgusII, xystep=2,
                              engine='serial')
    set_params = {'rho': 432, 'axlambda': 2, 'n_axons': 3, 'n_ax_segments': 4,
                  'loc_od_x': 5, 'loc_od_y': 6}
    for param in set_params:
        npt.assert_equal(hasattr(model, param), True)

    # AxonMapModel uses the SRD loss, should have `greater_is_better` set to
    # False
    npt.assert_equal(hasattr(model, 'greater_is_better'), True)
    npt.assert_equal(model.greater_is_better, False)

    # User can override default values
    for key, value in six.iteritems(set_params):
        model.set_params(**{key: value})
        npt.assert_equal(getattr(model, key), value)
    model = ValidAxonMapModel(implant_type=p2pi.ArgusII, xystep=2,
                              engine='serial')
    model.fit(X, **set_params)
    for key, value in six.iteritems(set_params):
        npt.assert_equal(getattr(model, key), value)

    # Some electrodes in the test set might not be in the train set:
    X.loc[0, 'electrode'] = 'F09'
    model.predict(X)

    for electrode, cm in six.iteritems(model._curr_map):
        # Current maps must be in [0, 1]
        npt.assert_almost_equal(cm.min(), 0, decimal=3)
        npt.assert_almost_equal(cm.max(), 0, decimal=3)  # FIXME
        # Phosphenes are shaped by axonal activation


def test_AxonMapModel__jansonius2009():
    # With `rho` starting at 0, all axons should originate in the optic disc
    # center
    for loc_od in [(15.0, 2.0), (-15.0, 2.0), (-4.2, -6.66)]:
        model = ValidAxonMapModel(loc_od_x=loc_od[0], loc_od_y=loc_od[1],
                                  xystep=2, engine='serial',
                                  ax_segments_range=(0, 45),
                                  n_ax_segments=100)
        for phi0 in [-135.0, 66.0, 128.0]:
            ax_pos = model._jansonius2009(phi0)
            npt.assert_almost_equal(ax_pos[0, 0], loc_od[0])
            npt.assert_almost_equal(ax_pos[0, 1], loc_od[1])

    # These axons should all end at the meridian
    for sign in [-1.0, 1.0]:
        for phi0 in [110.0, 135.0, 160.0]:
            model = ValidAxonMapModel(loc_od_x=15, loc_od_y=2,
                                      xystep=2, engine='serial',
                                      n_ax_segments=801,
                                      ax_segments_range=(0, 45))
            ax_pos = model._jansonius2009(sign * phi0)
            print(ax_pos[-1, :])
            npt.assert_almost_equal(ax_pos[-1, 1], 0.0, decimal=1)

    # `phi0` must be within [-180, 180]
    for phi0 in [-200.0, 181.0]:
        with pytest.raises(ValueError):
            ValidAxonMapModel(xystep=2, engine='serial')._jansonius2009(phi0)

    # `n_rho` must be >= 1
    for n_rho in [-1, 0]:
        with pytest.raises(ValueError):
            model = ValidAxonMapModel(n_ax_segments=n_rho, xystep=2,
                                      engine='serial')
            model._jansonius2009(0.0)

    # `ax_segments_range` must have min <= max
    for lorho in [-200.0, 90.0]:
        with pytest.raises(ValueError):
            model = ValidAxonMapModel(ax_segments_range=(lorho, 45), xystep=2,
                                      engine='serial')
            model._jansonius2009(0)
    for hirho in [-200.0, 40.0]:
        with pytest.raises(ValueError):
            model = ValidAxonMapModel(ax_segments_range=(45, hirho), xystep=2,
                                      engine='serial')
            model._jansonius2009(0)

    # A single axon fiber with `phi0`=0 should return a single pixel location
    # that corresponds to the optic disc
    for eye in ['LE', 'RE']:
        for loc_od in [(15.5, 1.5), (7.0, 3.0), (-2.0, -2.0)]:
            model = ValidAxonMapModel(loc_od_x=loc_od[0], loc_od_y=loc_od[1],
                                      xystep=2, engine='serial',
                                      ax_segments_range=(0, 0),
                                      n_ax_segments=1)
            single_fiber = model._jansonius2009(0)
            npt.assert_equal(len(single_fiber), 1)
            npt.assert_almost_equal(single_fiber[0], loc_od)


def test_AxonMapModel__grows_axon_bundles():
    for n_axons in [1, 2, 3, 5, 10]:
        model = ValidAxonMapModel(xystep=2, engine='serial', n_axons=n_axons,
                                  axons_range=(-20, 20))
        model.build_ganglion_cell_layer()
        bundles = model._grows_axon_bundles()
        npt.assert_equal(len(bundles), n_axons)


def test_AxonMapModel__finds_closest_axons():
    model = ValidAxonMapModel(xystep=1, engine='serial', n_axons=5,
                              axons_range=(-45, 45))
    model.build_ganglion_cell_layer()

    # Pretend there is an axon close to each point on the grid:
    bundles = [np.array([x + 0.001, y - 0.001]).reshape((1, 2))
               for x, y in zip(model.xret.ravel(), model.yret.ravel())]
    closest = model._finds_closest_axons(bundles)
    for ax1, ax2 in zip(bundles, closest):
        npt.assert_almost_equal(ax1[0, 0], ax2[0, 0])
        npt.assert_almost_equal(ax1[0, 1], ax2[0, 1])


def test_AxonMapModel__calcs_axon_contribution():
    model = ValidAxonMapModel(xystep=2, engine='serial', n_axons=10,
                              axons_range=(-30, 30))
    model.build_ganglion_cell_layer()
    xyret = np.column_stack((model.xret.ravel(), model.yret.ravel()))
    bundles = model._grows_axon_bundles()
    axons = model._finds_closest_axons(bundles)
    contrib = model._calcs_axon_contribution(axons)

    # Check lambda math:
    for ax, xy in zip(contrib, xyret):
        axon = np.insert(ax, 0, list(xy) + [0], axis=0)
        d2 = np.cumsum(np.diff(axon[:, 0], axis=0) ** 2 +
                       np.diff(axon[:, 1], axis=0) ** 2)
        sensitivity = np.exp(-d2 / (2.0 * model.axlambda ** 2))
        npt.assert_almost_equal(sensitivity, ax[:, 2])


def test_AxonMapModel__calc_bundle_tangent():
    model = m.ModelC(xystep=5, engine='serial', n_axons=500, n_ax_segments=500,
                     axons_range=(-180, 180), ax_segments_range=(3, 50))
    npt.assert_almost_equal(model.calc_bundle_tangent(0, 0), 0.4819, decimal=3)
    npt.assert_almost_equal(model.calc_bundle_tangent(0, 1000), -0.5532,
                            decimal=3)
