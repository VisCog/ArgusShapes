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


def gridscore(g, ppipe, XX, yy, model_params={}):
    pipe = clone(ppipe)
    pipe.set_params(**g)
    pipe.fit(XX, **model_params)
    return pipe.score(XX, yy)


filename = 'fit-spatial.pickle'
rootfolder = 'C:/Users/mbeyeler/data/secondsight/shape/52-001'
subject = None
electrodes = 'A05'
scaling = 8
img_shape = (41, 61)
X, y = p2pspatial.load_data(rootfolder, subject=subject, electrodes=electrodes,
                            scaling=scaling, img_shape=img_shape)

sensitivity_rule = 'decay'
search_params = {'reg__decay_const': (1, 10, 10),
                 'reg__implant_x': (-1000, 1000, 10),
                 'reg__implant_y': (-1000, 1000, 10),
                 'reg__implant_rot': (0, 2 * np.pi, 8)}
fit_params = {'reg__sampling': 200,
              'reg__sensitivity_rule': sensitivity_rule}
orig_pipe = Pipeline([('reg', p2pspatial.SpatialModelRegressor())])

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
