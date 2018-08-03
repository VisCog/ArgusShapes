from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import six
import scipy.optimize as spo
import pyswarm
import sklearn.base as sklb
import sklearn.metrics as sklm
import sklearn.utils.validation as skluv


class FunctionMinimizer(sklb.BaseEstimator):

    def __init__(self, estimator, search_params, search_params_init=None,
                 method='L-BFGS-B', max_iter=50, print_iter=1, min_step=1e-5,
                 verbose=True):
        """Performs function minimization

        Parameters
        ----------
        estimator :
            A scikit-learn estimator. Make sure its scoring function has
            greater equals better.
        search_params : dict of tupels (lower bound, upper bound)
            Search parameters
        search_params_init : dict of floats, optional, default: None
            Initial values of all search parameters. If None, initialize to
            midpoint between lower and upper bounds
        method : str, optional, default: 'L-BFGS-B'
            Solving method to use (e.g., 'Nelder-Mead', 'Powell', 'L-BFGS-B')
        max_iter : int, optional, default: 100
            Maximum number of iterations for the swarm to search.
        print_iter : int, optional, default: 10
            Print status message every x iterations
        min_step : float, optional, default: 0.1
            Minimum gradient change before termination.
        verbose : bool, optional, default: True
            Flag whether to print more stuff
        """
        self.estimator = estimator
        assert hasattr(estimator, 'greater_is_better')
        self.search_params = search_params
        if search_params_init is None:
            search_params_init = {}
            for k, v in six.iteritems(self.search_params):
                search_params_init[k] = (v[1] - v[0]) / 2.0
        self.search_params_init = search_params_init
        self.method = method
        self.max_iter = max_iter
        self.print_iter = print_iter
        self.min_step = min_step
        self.verbose = verbose

    def calc_error(self, search_vals, X, y, fit_params={}):
        """Calculates the estimator's error

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

        # Loss function: if `greater_is_better`, the estimator's ``score``
        # method is a true scoring function => invert to get an error function
        loss = estimator.score(X, y)
        loss = -loss if estimator.greater_is_better else loss
        if np.mod(self.iter, self.print_iter) == 0:
            print("Iter %d: Loss=%f, %s" % (
                self.iter, loss, ', '.join(['%s: %f' % (k, v)
                                            for k, v
                                            in six.iteritems(search_params)])))
        self.iter += 1
        return loss

    def fit(self, X, y, fit_params={}):
        """Runs the optimizer"""
        self.iter = 0
        # (lower, upper) bounds for every parameter
        bounds = [v for v in self.search_params.values()]
        init = [v for v in self.search_params_init.values()]
        options = {'maxfun': self.max_iter, 'gtol': self.min_step, 'eps': 100}
        res = spo.minimize(self.calc_error, init, args=(X, y, fit_params),
                           bounds=bounds, options=options)
        if not res['success']:
            print('Optimization unsucessful:')
            print(res)

        # Pair values of best params with their names to build a dict
        self.best_params_ = {}
        for k, v in zip(list(self.search_params.keys()), res['x']):
            self.best_params_[k] = v
        self.best_train_score_ = res['fun']
        print('Best err:', res['fun'], 'Best params:', self.best_params_)

        # Fit the class attribute with best params
        self.estimator.set_params(**self.best_params_)
        self.estimator.fit(X, y=y, **fit_params)

    def predict(self, X):
        msg = "Estimator, %(name)s, must be fitted before predicting."
        skluv.check_is_fitted(self, "best_params_", msg=msg)
        return self.estimator.predict(X)

    def score(self, X, y, sample_weight=None):
        return self.estimator.score(X, y, sample_weight=None)


class GridSearchOptimizer(sklb.BaseEstimator):

    def __init__(self, estimator, search_params, verbose=True):
        self.estimator = estimator
        assert hasattr(estimator, 'greater_is_better')
        self.search_params = search_params
        self.verbose = verbose

    def fit(self, X, y, fit_params={}):
        best_params = {}
        best_loss = np.inf
        for params in self.search_params:
            estimator = sklb.clone(self.estimator)
            estimator.set_params(**params)
            estimator.fit(X, y=y, **fit_params)
            loss = estimator.score(X, y)
            loss = -loss if estimator.greater_is_better else loss
            if loss < best_loss:
                best_loss = loss
                best_params = params
        self.best_params_ = best_params
        print('Best err:', best_loss, 'Best params:', self.best_params_)

        self.estimator.set_params(**self.best_params_)
        self.estimator.fit(X, y=y, **fit_params)
        return self

    def predict(self, X):
        msg = "Estimator, %(name)s, must be fitted before predicting."
        skluv.check_is_fitted(self, "best_params_", msg=msg)
        return self.estimator.predict(X)

    def score(self, X, y, sample_weight=None):
        return self.estimator.score(X, y, sample_weight=None)


class ParticleSwarmOptimizer(sklb.BaseEstimator):

    def __init__(self, estimator, search_params, swarm_size=None, max_iter=50,
                 min_func=0.01, min_step=0.01, verbose=True):
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
        min_func : float, optional, default: 0.01
            The minimum change of swarm's best objective value before the
            search terminates.
        min_step : float, optional, default: 0.01
            The minimum step size of swarm's best objective value before
            the search terminates.
        verbose : bool, optional, default: True
            Flag whether to print more stuff
        """
        if swarm_size is None:
            swarm_size = 10 * len(search_params)
        self.estimator = estimator
        assert hasattr(estimator, 'greater_is_better')
        self.search_params = search_params
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.min_func = min_func
        self.min_step = min_step
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

        # Loss function: if `greater_is_better`, the estimator's ``score``
        # method is a true scoring function => invert to get an error function
        loss = estimator.score(X, y)
        loss = -loss if estimator.greater_is_better else loss
        return loss

    def fit(self, X, y, fit_params={}):
        # Run particle swarm optimization
        lb = [v[0] for v in self.search_params.values()]
        ub = [v[1] for v in self.search_params.values()]
        best_vals, best_err = pyswarm.pso(
            self.swarm_error, lb, ub, swarmsize=self.swarm_size,
            maxiter=self.max_iter, minfunc=self.min_func, minstep=self.min_step,
            debug=self.verbose, args=[X, y], kwargs={'fit_params': fit_params}
        )

        # Pair values of best params with their names to build a dict
        self.best_params_ = {}
        for k, v in zip(list(self.search_params.keys()), best_vals):
            self.best_params_[k] = v
        self.best_train_score_ = best_err
        print('Best err:', best_err, 'Best params:', self.best_params_)

        # Fit the class attribute with best params
        self.estimator.set_params(**self.best_params_)
        self.estimator.fit(X, y=y, **fit_params)

    def predict(self, X):
        msg = "Estimator, %(name)s, must be fitted before predicting."
        skluv.check_is_fitted(self, "best_params_", msg=msg)
        return self.estimator.predict(X)

    def score(self, X, y, sample_weight=None):
        return self.estimator.score(X, y, sample_weight=None)


def crossval_predict(estimator, X, y, fit_params={}, n_folds=5, idx_fold=-1,
                     groups=None, verbose=True):
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
    groups : str
        Column name of `X` to be used as groups. If `groups` is given,
        `n_folds` will be ignored, and the result is leave-one-group-out
        cross-validation.

    Returns
    -------
    y_true : list
    y_pred : list
    best_params = dict
    """
    assert isinstance(X, pd.core.frame.DataFrame)
    assert isinstance(y, pd.core.frame.DataFrame)
    assert n_folds > 1
    assert idx_fold >= -1 and idx_fold < n_folds
    # Manual partitioning of X
    all_idx = np.arange(len(X))
    if groups is None:
        # No groups given: manually partition
        groups = np.array_split(all_idx, n_folds)
    else:
        # `groups` must be a column of `X`
        assert isinstance(groups, six.string_types)
        assert groups in X.columns
        # Transform into a list of folds, each of which has an array of
        # data sample indices, thus mimicking np.array split; i.e. from
        # ['S1', 'S1', 'S2, 'S2', 'S3', 'S3']
        # to
        # [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]:
        groups = [np.where(X[groups] == i)[0] for i in np.unique(X[groups])]
        n_folds = len(groups)
        assert idx_fold < n_folds

    y_true = []
    y_pred = []
    best_params = []
    best_train_score = []
    best_test_score = []
    for i, test_idx in enumerate(groups):
        if idx_fold != -1 and idx_fold != i:
            # Process only one fold, not all
            continue
        if verbose:
            print('Fold %d / %d' % (i + 1, n_folds))
        train_idx = np.delete(all_idx, test_idx)
        est = sklb.clone(estimator)
        est.fit(X.iloc[train_idx, :], y.iloc[train_idx, :], fit_params=fit_params)
        if hasattr(est, 'best_params_'):
            best_params.append(est.best_params_)
        else:
            best_params.append(None)
        if hasattr(est, 'best_train_score_'):
            best_train_score.append(est.best_train_score_)
        else:
            best_train_score.append(None)
        y_true.append(y.iloc[test_idx, :])
        y_pred.append(est.predict(X.iloc[test_idx, :]))
        best_test_score.append(est.score(X.iloc[test_idx, :], y.iloc[test_idx, :]))
    return y_true, y_pred, best_params, best_train_score, best_test_score


def crossval_score(y_true, y_pred, metric='mse', key='all', weights=None):
    score_funcs = {'mse': sklm.mean_squared_error,
                   'mae': sklm.mean_absolute_error,
                   'msle': sklm.mean_squared_log_error,
                   'var_explained': sklm.explained_variance_score,
                   'r2': sklm.r2_score}
    assert metric in score_funcs.keys()
    scores = []
    for yt, yp in zip(y_true, y_pred):
        if key is not None and key != 'all':
            scores.append(score_funcs[metric](yt.loc[:, key], yp.loc[:, key]))
        else:
            scores.append(score_funcs[metric](yt, yp, multioutput=weights))
    return scores
