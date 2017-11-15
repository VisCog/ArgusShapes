import numpy as np
import os
import pickle
from time import time
from datetime import datetime

import skimage.io as skio
import skimage.transform as skit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from sklearn.metrics import r2_score, explained_variance_score

import pulse2percept as p2p
import p2pmodelselect
import p2pspatial


subject = None
electrodes = None
sensitivity_rule = 'decay'
scaling = 8
img_shape = (41, 61)
n_jobs = 1
n_folds = 5
rootfolder = '/home/mbeyeler/data/secondsight/shape/52-001'

now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = 'crossval-spatial_%s_%s.pickle' % (sensitivity_rule, now)

X, y = p2pspatial.load_data(rootfolder, subject=subject, electrodes=electrodes,
                            scaling=scaling, img_shape=img_shape)
print(X.shape, y.shape)
assert len(X) == len(y) and len(X) > 0

search_params = {'reg__decay_const': (1, 20, 20)}
fit_params = {'reg__sampling': 200,
              'reg__sensitivity_rule': sensitivity_rule,
              'reg__loc_od': (15.609559078040428, 2.2381648328706558),
              'reg__implant_x': -1657.11040863,
              'reg__implant_y': 196.93351877,
              'reg__implant_rot': -0.43376793904131516}
orig_pipe = Pipeline([('reg', p2pspatial.SpatialModelRegressor())])
validator = p2pmodelselect.ModelValidator(orig_pipe, search_params,
                                          fit_params=fit_params,
                                          n_jobs=n_jobs)
X_test, y_true, y_pred, cv_results = p2pmodelselect.utils.crossval_predict(
    validator, X, y, n_folds=n_folds
)
sim_time = time() - t0
print("done in %0.3fs" % sim_time)
specifics = {'subject': subject,
             'electrodes': electrodes,
             'ecmethod': ecmethod,
             'validator': validator,
             'datafiles': datafiles,
             'X': X,
             'y': y,
             'n_folds': n_folds,
             'now': now,
             'sim_time': sim_time,
             'random_state': random_state}
pickle.dump((X_test, y_true, y_pred, cv_results, specifics),
            open(filename, 'wb'))
print('Dumped data to %s' % filename)


print('performing grid search')
t0 = time()
grid = ParameterGrid(search_params)
scores = p2p.utils.parfor(gridscore, grid, func_args=[orig_pipe, X, y],
                          func_kwargs={'model_params': fit_params})

best_score = np.max(scores)
print('best score:', best_score)

best_params = grid[np.argmax(scores)]
print('best params:', best_params)

pipe = clone(orig_pipe)
pipe.set_params(**best_params)
pipe.fit(X, **model_params)
print('score:', pipe.score(X, y))

print('explained var:', explained_variance_score(y, pipe.predict(X)))

print('r2:', r2_score(y, pipe.predict(X)))

pickle.dump((X, y, best_params), open(filename, 'wb'))
print('Dumped data to %s' % filename)
print(time() - t0)
