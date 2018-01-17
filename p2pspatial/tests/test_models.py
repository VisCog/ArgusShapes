from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import pandas as pd
import sklearn.exceptions as skle

import pulse2percept.implants as p2pi
from .. import models

import numpy.testing as npt
import pytest


class DummyModel(models.RetinalGridMixin, models.BaseModel):

    def _calc_curr_map(self, electrode):
        return electrode, 0

    def _predict(self, Xrow):
        """Returns the input (a DataFrame row, without its index)"""
        _, row = Xrow
        return row


def test_BaseModel():
    """Tests the BaseModel from which all real models derive"""
    # Must supply a valid electrode array
    for implant in [2, np.array([10]), p2pi.Electrode('epiretinal', 1, 0, 0)]:
        with pytest.raises(TypeError):
            model = DummyModel(implant)

    model = DummyModel(implant_type=p2pi.ArgusI)
    npt.assert_equal(hasattr(model, 'implant_type'), True)
    npt.assert_equal(isinstance(model.implant_type, type), True)

    # All these parameters should be settable in the constructor:
    model_params = model.get_params()
    DummyModel(**model_params)
    # But not others:
    with pytest.raises(ValueError):
        DummyModel(meow=0)

    # Create a valid DataFrame with two different images
    img1 = np.ones((10, 10))
    img1[3, 3] = 0
    img2 = np.ones((10, 10))
    img2[5, 5] = 0
    X = pd.DataFrame([['A01', img1], ['A2', img2]],
                     columns=['electrode', 'image'], index=[2, 182])

    # Must be fitted first:
    npt.assert_equal(model._is_fitted, False)
    with pytest.raises(skle.NotFittedError):
        model.predict(X)
    model.fit(X)
    npt.assert_equal(model._is_fitted, True)
    npt.assert_equal(hasattr(model, 'implant'), True)
    npt.assert_equal(isinstance(model.implant, p2pi.ArgusI), True)

    # Must pass correctly through ``predict`` and ``score``
    npt.assert_equal(X.equals(model.predict(X)), True)
    npt.assert_almost_equal(model.score(X, X), 0.0)

    # Only accepts pandas DataFrame
    for X in [42, [3.3, 1.1], {'img': [[2]]}]:
        with pytest.raises(TypeError):
            model.predict(X)
    y = pd.DataFrame([{'img': [[2]]}, {'img': [[4]]}], index=[0, 1])
    for X in [42, [3.3, 1.1], {'img': [[2]]}]:
        with pytest.raises(TypeError):
            model.score(X, y)
    X = pd.DataFrame([{'img': [[2]]}, {'img': [[4]]}], index=[0, 1])
    for y in [42, [3.3, 1.1], {'img': [[2]]}]:
        with pytest.raises(TypeError):
            model.score(X, y)


def test_ModelA():
    model = models.ModelA(implant_type=p2pi.ArgusII)
    npt.assert_equal(hasattr(model, 'implant_type'), True)
    npt.assert_equal(isinstance(model.implant_type, type), True)

    # All these parameters should be settable in the constructor:
    model_params = model.get_params()
    models.ModelA(**model_params)
    # But not others:
    with pytest.raises(ValueError):
        models.ModelA(implant_type=p2pi.ArgusI, meow=0)

    # Build a data matrix
    img = np.ones((10, 10))
    img[0, 0] = 0
    X = pd.DataFrame([['A01', img, img.shape],
                      ['B03', img, img.shape],
                      ['C09', img, img.shape]],
                     columns=['electrode', 'image', 'img_shape'],
                     index=[1, 3, 10])

    # Must be fitted first:
    npt.assert_equal(model._is_fitted, False)
    with pytest.raises(skle.NotFittedError):
        model.predict(pd.DataFrame())

    # Make sure we calculate only the current maps we need: We start with
    # nothing, but add a new current map step-by-step
    npt.assert_equal(model._curr_map, {})
    for idx in [1, 2, 3]:
        slc = X.iloc[:idx, :]
        model.fit(slc)
        npt.assert_equal(model._is_fitted, True)
        npt.assert_equal(set(model._curr_map.keys()), set(slc.electrode))
        model._is_fitted = False

    # Additional calls to ``calc_curr_map`` should not make a difference:
    model.calc_curr_map(X)
    npt.assert_equal(set(model._curr_map.keys()), set(X.electrode))
