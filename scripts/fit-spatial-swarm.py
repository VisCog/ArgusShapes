import numpy as np
import os
import pickle
import pyswarm
from time import time
from datetime import datetime

import skimage.io as skio
import skimage.transform as skit
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from sklearn.metrics import r2_score, explained_variance_score

import pulse2percept as p2p
import p2pmodelselect
import p2pspatial


def swarm_error(search_vals, regressor, XX, yy, search_keys, fit_params={}):
    # Rebuild dictionary from keys and values in list
    search_params = {}
    for k, v in zip(search_keys, search_vals):
        search_params[k] = v
    reg = clone(regressor)
    reg.set_params(**search_params)
    reg.fit(XX, **fit_params)
    return np.mean(reg.rmse(XX, yy))


now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = 'fit-spatial-swarm_%s.pickle' % now
rootfolder = os.path.join(os.environ['SECOND_SIGHT_DATA'], 'shape', '52-001')
subject = None
electrodes = None
X, y = p2pspatial.load_data(rootfolder, subject=subject, electrodes=electrodes,
                            single_stim=True, verbose=True)
print(X.shape, y.shape)
if len(X) == 0:
    raise ValueError('No data found in %s' % rootfolder)

sensitivity_rule = 'decay'
search_params = {'decay_const': (1, 100),
                 'cswidth': (50, 1000),
                 'implant_x': (-1500, 1500),
                 'implant_y': (-500, 500),
                 'implant_rot': (0, 2 * np.pi),
                 'thresh': (0.1, 1.2)}
fit_params = {'sampling': 200,
              'loc_od': (13.13228137, 2.163527405),
              'csmode': 'gaussian',
              'sensitivity_rule': sensitivity_rule}
regressor = p2pspatial.SpatialModelRegressor()

print('performing swarm optimization')
t0 = time()
lb = [v[0] for v in search_params.values()]
ub = [v[1] for v in search_params.values()]
xopt, fopt = pyswarm.pso(swarm_error, lb, ub,
                         args=[regressor, X, y, list(search_params.keys())],
                         kwargs={'fit_params': fit_params}, debug=True)

pickle.dump((X, y, xopt, fopt, regressor, search_params, fit_params),
            open(filename, 'wb'))
print('Dumped data to %s' % filename)
print(time() - t0)
