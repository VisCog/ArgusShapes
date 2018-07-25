from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import sklearn.base as sklb

from .. import model_selection as ms

import numpy.testing as npt
import pytest


class DummyPredictor(sklb.BaseEstimator):

    def fit(self, X, y=None, **fit_params):
        return self

    def predict(self, X):
        return X['feature1']

    def score(self, X, y):
        return np.sum((y['target'] - X['feature1']) ** 2)


def generate_dummy_data():
    X = pd.DataFrame()
    X['subject'] = pd.Series(['S1', 'S1', 'S2', 'S2', 'S3', 'S3'])
    X['feature1'] = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    X['feature2'] = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    y = pd.DataFrame()
    y['subject'] = pd.Series(['S1', 'S1', 'S2', 'S2', 'S3', 'S3'],
                             index=X.index)
    y['target'] = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                            index=X.index)
    return X, y


def test_crossval_predict():
    X, y = generate_dummy_data()
    dummy = DummyPredictor()

    # Grouped by subject:
    y_true, y_pred, best_params, best_train, best_test = ms.crossval_predict(
        dummy, X, y, groups='subject', verbose=False
    )
    npt.assert_equal(len(y_true), len(X.subject.unique()))
    for subject, yt in zip(X.subject.unique(), y_true):
        npt.assert_equal(np.allclose(np.where(X.subject == subject), yt.index),
                         True)
