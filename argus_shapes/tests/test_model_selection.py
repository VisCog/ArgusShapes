from __future__ import absolute_import, division, print_function
import numpy as np
import sklearn.base as sklb
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import ParameterGrid

from . import test_argus_shapes as shapes
from .. import model_selection as ms

import numpy.testing as npt
import pytest


class DummyPredictor(sklb.BaseEstimator):

    def __init__(self, dummy_var=1):
        self.dummy_var = dummy_var
        self.greater_is_better = False

    def fit(self, X, y=None, **fit_params):
        return self

    def predict(self, X):
        return X['feature1']

    def score(self, X, y, sample_weight=None):
        return np.sum((y['target'] - self.dummy_var * X['feature1']) ** 2)


def test_FunctionMinimizer():
    # DummyPredictor always predicts 'feature1'.
    # The best `dummy_var` value is 1.
    X, y = shapes.generate_dummy_data()
    fmin = ms.FunctionMinimizer(DummyPredictor(), {'dummy_var': (0.5, 2.5)})
    with pytest.raises(NotFittedError):
        fmin.predict(X)
    fmin.fit(X, y)
    npt.assert_almost_equal(fmin.estimator.dummy_var, 1.0)
    npt.assert_almost_equal(fmin.score(X, y), 0.0)


def test_GridSearchOptimizer():
    # DummyPredictor always predicts 'feature1'.
    # The best `dummy_var` value is 1.
    X, y = shapes.generate_dummy_data()
    search_params = {'dummy_var': np.linspace(0.5, 2, 4)}
    fmin = ms.GridSearchOptimizer(DummyPredictor(),
                                  ParameterGrid(search_params))
    with pytest.raises(NotFittedError):
        fmin.predict(X)
    fmin.fit(X, y)
    npt.assert_almost_equal(fmin.estimator.dummy_var, 1.0)
    npt.assert_almost_equal(fmin.score(X, y), 0.0)


def test_ParticleSwarmOptimizer():
    # DummyPredictor always predicts 'feature1'.
    # The best `dummy_var` value is 1.
    X, y = shapes.generate_dummy_data()
    fmin = ms.ParticleSwarmOptimizer(DummyPredictor(),
                                     {'dummy_var': (0.5, 2.5)},
                                     min_func=1e-6, min_step=1e-6)
    with pytest.raises(NotFittedError):
        fmin.predict(X)
    # Test both {} and None:
    fmin.fit(X, y, fit_params=None)
    fmin.fit(X, y, fit_params={})
    npt.assert_almost_equal(fmin.estimator.dummy_var, 1.0, decimal=2)
    npt.assert_almost_equal(fmin.score(X, y), 0.0, decimal=2)


def test_crossval_predict():
    X, y = shapes.generate_dummy_data()
    dummy = DummyPredictor()

    # Grouped by subject:
    y_true, y_pred, _, _, _ = ms.crossval_predict(
        dummy, X, y, groups='subject', verbose=False
    )
    npt.assert_equal(len(y_true), len(X.subject.unique()))
    for subject, yt in zip(X.subject.unique(), y_true):
        npt.assert_equal(np.allclose(np.where(X.subject == subject), yt.index),
                         True)


def test_crossval_score():
    X, y = shapes.generate_dummy_data()
    dummy = DummyPredictor()
    dummy.fit(X, y)
    npt.assert_almost_equal(ms.crossval_score([y.target], [dummy.predict(X)]),
                            [0.0])
