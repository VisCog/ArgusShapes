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
random_state = 42
rootfolder = '/home/mbeyeler/data/secondsight/shape/52-001'

now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = 'crossval-spatial_%s_%s.pickle' % (sensitivity_rule, now)

t0 = time()
X, y = p2pspatial.load_data(rootfolder, subject=subject, electrodes=electrodes,
                            scaling=scaling, img_shape=img_shape,
                            random_state=random_state)
print(X.shape, y.shape)
assert len(X) == len(y) and len(X) > 0

search_params = {'reg__decay_const': (1, 20, 20)}
fit_params = {'reg__sampling': 200,
              'reg__sensitivity_rule': sensitivity_rule,
              'reg__loc_od': (15.609559078040428, 2.2381648328706558),
              'reg__implant_x': -1657.11040863,
              'reg__implant_y': 196.93351877,
              'reg__implant_rot': -0.43376793904131516,
              'reg__thresh': 'min'}
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
             'validator': validator,
             'X': X,
             'y': y,
             'n_folds': n_folds,
             'now': now,
             'random_state': random_state}
pickle.dump((X_test, y_true, y_pred, cv_results, specifics),
            open(filename, 'wb'))
print('Dumped data to %s' % filename)
