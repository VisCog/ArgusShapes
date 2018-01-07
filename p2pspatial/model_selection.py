from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import six
import pyswarm
import sklearn.base as sklb
import sklearn.metrics as sklm
import sklearn.utils.validation as skluv


class ParticleSwarmOptimizer(sklb.BaseEstimator, sklb.RegressorMixin):

    def __init__(self, estimator, search_params, swarm_size=None, max_iter=100,
                 min_func=1e-4, verbose=True):
        """Performs particle swarm optimization

        Parameters
        ----------
        estimator :
            A scikit-learn estimator. Make sure its scoring function has
            greater equals better.
        search_params : dict of tupels (lower bound, upper bound)
            Search parameters
        swarm_size : int, optional, default: 10 * number of search params
            The number of particles in the swarm.
        max_iter : int, optional, default: 100
            Maximum number of iterations for the swarm to search.
        min_func : float, optional, default: 1e-4
            The minimum change of swarm's best objective value before the
            search terminates.
        verbose : bool, optional, default: True
            Flag whether to print more stuff
        """
        if swarm_size is None:
            swarm_size = 10 * len(search_params)
        self.estimator = estimator
        self.search_params = search_params
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.min_func = min_func
        self.verbose = verbose

    def swarm_error(self, search_vals, X, y, fit_params={}):
        """Calculates the particle swarm error

        The error is calculated using the estimator's scoring function (assumes
        a true scoring function, i.e. greater == better).
        """
        # pyswarm provides values for all search parameters in a list:
        # Need to pair these values with the names of the search params
        # to build a dict
        search_params = {}
        for k, v in zip(list(self.search_params.keys()), search_vals):
            search_params[k] = v

        # Clone the estimator to make sure we have a clean slate
        estimator = sklb.clone(self.estimator)
        estimator.set_params(**search_params)
        estimator.fit(X, y=y, **fit_params)

        # Scoring function: greater is better, so invert to get an
        # error function
        return -estimator.score(X, y)

    def fit(self, X, y, fit_params={}):
        # Run particle swarm optimization
        lb = [v[0] for v in self.search_params.values()]
        ub = [v[1] for v in self.search_params.values()]
        best_vals, best_err = pyswarm.pso(
            self.swarm_error, lb, ub, swarmsize=self.swarm_size,
            maxiter=self.max_iter, minfunc=self.min_func, debug=self.verbose,
            args=[X, y], kwargs={'fit_params': fit_params}
        )

        # Pair values of best params with their names to build a dict
        self.best_params_ = {}
        for k, v in zip(list(self.search_params.keys()), best_vals):
            self.best_params_[k] = v
        print('Best err:', best_err, 'Best params:', self.best_params_)

        # Fit the class attribute with best params
        self.estimator.set_params(**self.best_params_)
        self.estimator.fit(X, y=y, **fit_params)

    def predict(self, X):
        msg = "Estimator, %(name)s, must be fitted before predicting."
        skluv.check_is_fitted(self, "best_params_", msg=msg)
        return self.estimator.predict(X)


def crossval_predict(estimator, X, y, fit_params={}, n_folds=5):
    """Performs cross-validation

    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface. Must
        possess a ``fit``, ``predict``, and ``score`` method.
    X : pd.core.data.DataFrame, shape = [n_samples, n_features]
        Training matrix, where `n_samples` is the number of samples and
        `n_features` is the number of features.
    y : pd.core.data.DataFrame, shape = [n_samples] or [n_samples, n_output]
        Target relative to `X` for classification or regression
    n_folds : int, optional, default: 2
        Number of cross-validation folds.
    """
    assert isinstance(X, pd.core.frame.DataFrame)
    assert isinstance(y, pd.core.frame.DataFrame)
    # Manual partitioning of X
    all_idx = np.arange(len(X))
    groups = np.array_split(all_idx, n_folds)

    y_true = []
    y_pred = []
    best_params = []
    for test_idx in groups:
        train_idx = np.delete(all_idx, test_idx)
        est = sklb.clone(estimator)
        est.fit(X.iloc[train_idx, :], y.iloc[train_idx], fit_params=fit_params)
        if hasattr(est, 'best_params_'):
            best_params.append(est.best_params_)
        else:
            best_params.append(None)
        y_true.append(y.iloc[test_idx])
        y_pred.append(est.predict(X.iloc[test_idx, :]))
    return y_true, y_pred, best_params


def crossval_score(y_true, y_pred, metric='mse'):
    score_funcs = {'mse': sklm.mean_squared_error,
                   'mae': sklm.mean_absolute_error,
                   'msle': sklm.mean_squared_log_error,
                   'var_explained': sklm.explained_variance_score,
                   'r2': sklm.r2_score}
    assert metric in score_funcs.keys()
    return [score_funcs[metric](yt, yp) for yt, yp in zip(y_true, y_pred)]
