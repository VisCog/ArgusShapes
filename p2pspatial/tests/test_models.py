from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import six

import sklearn.exceptions as skle

import pulse2percept.implants as p2pi
import pulse2percept.retina as p2pr
from .. import models as m

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


class ValidBaseModel(m.SRDLoss, m.RetinalGrid, m.BaseModel):
    """A class that implements all abstract methods of BaseModel"""

    def _calcs_el_curr_map(self, electrode):
        return np.array([[0]])

    def _predicts_target_values(self, row):
        return row

    def _predicts(self, Xrow):
        """Returns the input (a DataFrame row, without its index)"""
        _, row = Xrow
        return self._predicts_target_values(row)


class ValidScoreboardModel(m.SRDLoss, m.RetinalGrid, m.ScoreboardModel):
    """A class that implements all abstract methods of BaseModel"""

    def _predicts_target_values(self, row):
        return row

    def _predicts(self, Xrow):
        """Returns the input (a DataFrame row, without its index)"""
        _, row = Xrow
        return self._predicts_target_values(row)


class ValidAxonMapModel(m.SRDLoss, m.RetinalGrid, m.AxonMapModel):
    """A class that implements all abstract methods of AxonMapModel"""

    def _predicts_target_values(self, row):
        return row

    def _predicts(self, Xrow):
        """Returns the input (a DataFrame row, without its index)"""
        _, row = Xrow
        return self._predicts_target_values(row)


def test_RetinalCoordTrafo__cart2pol():
    trafo = m.RetinalCoordTrafo()
    npt.assert_almost_equal(trafo._cart2pol(0, 0), (0, 0))
    npt.assert_almost_equal(trafo._cart2pol(10, 0), (0, 10))
    npt.assert_almost_equal(trafo._cart2pol(3, 4), (np.arctan(4 / 3.0), 5))
    npt.assert_almost_equal(trafo._cart2pol(4, 3), (np.arctan(3 / 4.0), 5))


def test_RetinalCoordTrafo__pol2cart():
    trafo = m.RetinalCoordTrafo()
    npt.assert_almost_equal(trafo._pol2cart(0, 0), (0, 0))
    npt.assert_almost_equal(trafo._pol2cart(0, 10), (10, 0))
    npt.assert_almost_equal(trafo._pol2cart(np.arctan(4 / 3.0), 5), (3, 4))
    npt.assert_almost_equal(trafo._pol2cart(np.arctan(3 / 4.0), 5), (4, 3))


def test_RetinalCoordTrafo__watson_displacement():
    trafo = m.RetinalCoordTrafo()
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
    trafo = m.RetinalCoordTrafo()
    for xy in [1, [1, 2], np.zeros((10, 3))]:
        with pytest.raises(ValueError):
            trafo._displaces_rgc(xy)

    npt.assert_almost_equal(
        trafo._displaces_rgc(np.array([0, 0]).reshape((1, 2))),
        ([p2pr.dva2ret(trafo._watson_displacement(0))], [0]),
        decimal=1
    )


def test_RetinalCoordTrafo_build_ganglion_cell_layer():
    trafo = m.RetinalCoordTrafo()
    trafo.xrange = (-2, 2)
    trafo.yrange = (-1, 1)
    trafo.xystep = 1
    trafo.build_ganglion_cell_layer()


def test_RetinalGrid():
    trafo = m.RetinalGrid()
    trafo.xrange = (-2, 2)
    trafo.yrange = (-1, 1)
    trafo.xystep = 1
    trafo.build_ganglion_cell_layer()

    # Make sure shape is right
    npt.assert_equal(trafo.xret.shape, (3, 5))
    npt.assert_equal(trafo.xret.shape, (3, 5))

    # Make sure transformation is right
    npt.assert_almost_equal(trafo.xret[1, 2], 0)
    npt.assert_almost_equal(trafo.yret[1, 2], 0)


def test_ImageMomentLoss():
    pass


def test_SRDLoss():
    pass


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
    for forbidden_key in ['_is_fitted', 'greater_is_better', 'invalid']:
        set_param = {forbidden_key: 1234}
        with pytest.raises(ValueError):
            model.set_params(**set_param)
        with pytest.raises(ValueError):
            ValidBaseModel(**set_param)


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

    # `fit_params` must take effect
    model = ValidBaseModel(engine='serial')
    model_params = model.get_params()
    for key, value in six.iteritems(model_params):
        npt.assert_equal(getattr(model, key), value)
        if isinstance(value, (int, float)):
            set_param = {key: 1234}
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


def test_BaseModel_score():
    # Model must be fitted first
    model = ValidBaseModel(engine='serial')
    X = get_dummy_data(nrows=3)
    with pytest.raises(skle.NotFittedError):
        model.score(X, X)

    # But then must pass through ``score`` just fine: ValidBaseModel returns input,
    # so that predict(X) == X, and score(X, X) == 0
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
    npt.assert_equal(hasattr(model, 'w_scale'), True)
    npt.assert_equal(hasattr(model, 'w_rot'), True)
    npt.assert_equal(hasattr(model, 'w_dice'), True)
    model.set_params(w_scale=34, w_rot=33, w_dice=34)

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
    model = ValidAxonMapModel(implant_type=p2pi.ArgusII, engine='serial')
    set_params = {'rho': 432, 'axlambda': 2, 'n_axons': 3, 'n_ax_segments': 4,
                  'loc_od_x': 5, 'loc_od_y': 6}
    for param in set_params:
        npt.assert_equal(hasattr(model, param), True)

    # AxonMapModel uses the SRD loss, should have `greater_is_better` set to
    # False
    npt.assert_equal(hasattr(model, 'greater_is_better'), True)
    npt.assert_equal(model.greater_is_better, False)
    npt.assert_equal(hasattr(model, 'w_scale'), True)
    npt.assert_equal(hasattr(model, 'w_rot'), True)
    npt.assert_equal(hasattr(model, 'w_dice'), True)
    model.set_params(w_scale=34, w_rot=33, w_dice=34)

    # User can override default values
    for key, value in six.iteritems(set_params):
        model.set_params(**{key: value})
        npt.assert_equal(getattr(model, key), value)
    model = ValidAxonMapModel(implant_type=p2pi.ArgusII, engine='serial')
    model.fit(X, **set_params)
    for key, value in six.iteritems(set_params):
        npt.assert_equal(getattr(model, key), value)

    # Some electrodes in the test set might not be in the train set:
    X.loc[0, 'electrode'] = 'F09'
    model.predict(X)

    for electrode, cm in six.iteritems(model._curr_map):
        # Current maps must be in [0, 1]
        npt.assert_almost_equal(cm.min(), 0, decimal=3)
        npt.assert_almost_equal(cm.max(), 1, decimal=3)
        # Phosphenes are shaped by axonal activation
