from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import six
import sklearn.base as sklb
import scipy.stats as spst


def crossval_predict(estimator, X, y, n_folds=5):
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

    X_test = []
    y_true = []
    y_pred = []
    for test_idx in groups:
        train_idx = np.delete(all_idx, test_idx)
        est = sklb.clone(estimator)
        est.fit(X.iloc[train_idx, :], y.iloc[train_idx])
        X_test.append(X.iloc[test_idx, :])
        y_true.append(y.iloc[test_idx])
        y_pred.append(est.predict(X.iloc[test_idx, :]))
    return X_test, y_true, y_pred


def crossval_score(X_test, y_true, y_pred, metric='mse', tasks=None):
    score_funcs = {'mse': sklm.mean_squared_error,
                   'mae': sklm.mean_absolute_error,
                   'msle': sklm.mean_squared_log_error,
                   'var_explained': sklm.explained_variance_score,
                   'r2': sklm.r2_score}
    if tasks is None or len(tasks) == 0:
        tasks = ['all']
        X_tasks = tasks
    else:
        X_tasks = np.unique([xt['task'].unique() for xt in X_test])

    t_mu = []
    t_std = []
    for task in tasks:
        if task != 'all' and task not in X_tasks:
            t_mu.append(np.nan)
            t_std.append(np.nan)
            continue

        fold_mse = []
        for xt, yt, yp in zip(X_test, y_true, y_pred):
            assert len(xt) == len(yt)
            assert len(yt) == len(yp)
            if task == 'all':
                idx = np.ones(len(xt), dtype=np.bool)
            else:
                idx = np.array(xt['task'] == task, dtype=np.bool)
            assert len(idx) == len(yt)
            yt = np.array(yt)[idx]
            yp = np.array(yp)[idx]
            score = score_funcs[metric](yt, yp)
            fold_mse.append(score)
        t_mu.append(np.mean(fold_mse))
        t_std.append(np.std(fold_mse))
    return t_mu, t_std


class NestedCV(sklb.BaseEstimator):

    def __init__(self, pipe, search_params, fit_params={}, n_jobs=-1,
                 cv=5, cvmethod='grid', cvsample='uniform', cvparams={},
                 return_train_score=False):
        """Hyperparameter selection using nested cross-validation

        The outer loop of the model validation procedure:
        - Data is split into development and evaluation sets.
        - The development set is split again in the `fit` method to perform
          grid search with cross-validation on it.
        - The model's performance is then evaluated on the evaluation set.
        """
        self.pipe = pipe
        self.cv = cv
        self.cvmethod = cvmethod
        self.cvsample = cvsample
        self.cvparams = cvparams
        self.return_train_score = return_train_score
        self.search_params = search_params
        self.fit_params = fit_params
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        if deep:
            get_pipe = clone(self.pipe)
        else:
            get_pipe = self.pipe

        return {'pipe': get_pipe,
                'search_params': self.search_params,
                'fit_params': self.fit_params,
                'n_jobs': self.n_jobs,
                'cv': self.cv,
                'cvmethod': self.cvmethod,
                'cvsample': self.cvsample,
                'cvparams': self.cvparams,
                'return_train_score': self.return_train_score}

    def set_params(self, **params):
        for param, value in six.iteritems(params):
            setattr(self, param, value)

    def predict(self, X):
        return self.pipe.predict(X)

    def fit(self, X, y=None, **fit_params):
        """Perform search on parameter space with cross-validation"""
        if self.cvmethod.lower() == 'grid':
            # Grid search: Generate parameter grid from value ranges
            if self.cvsample.lower() == 'uniform':
                for key, valrange in six.iteritems(self.search_params):
                    assert isinstance(self.search_params[key], (tuple, list))
                    self.search_params[key] = np.linspace(*valrange)
            else:
                raise NotImplementedError
            # Run grid search
            search = GridSearchCV(self.pipe, self.search_params, verbose=1,
                                  cv=self.cv, n_jobs=self.n_jobs,
                                  return_train_score=self.return_train_score,
                                  **self.cvparams)

        elif self.cvmethod.lower() == 'random':
            # Randomized grid search
            if self.cvsample.lower() == 'uniform':
                for key, valrange in six.iteritems(self.search_params):
                    assert isinstance(self.search_params[key], (tuple, list))
                    self.search_params[key] = scst.uniform(
                        loc=valrange[0], scale=valrange[1] - valrange[0]
                    )
            else:
                raise NotImplementedError

            # Run randomized grid search
            search = RandomizedSearchCV(
                self.pipe, self.search_params, verbose=1, cv=self.cv,
                n_jobs=self.n_jobs, return_train_score=self.return_train_score,
                **self.cvparams
            )

        elif self.cvmethod.lower() == 'swarm':
            # Particle swarm optimization
            search = ParticleSwarmCV(self.pipe, self.search_params, verbose=1,
                                     cv=self.cv, n_jobs=self.n_jobs,
                                     **self.cv_params)
            raise NotImplementedError
        else:
            raise ValueError('Unknown `cvmethod` "%s"' % self.cvmethod)

        search.fit(X, y, **self.fit_params)
        self.pipe = search.best_estimator_
        self.best_params = search.best_params_
        self.cv_result = search.cv_results_
        print(search.best_params_)
        return self

    def score(self, X, y):
        return self.pipe.score(X, y)
