
# coding: utf-8

# In[1]:

import numpy as np
import os
import pickle
from time import time
from datetime import datetime


# In[2]:

import pulse2percept as p2p
import p2pspatial


# In[3]:

subject = '12-005'
amplitude = 2.0
electrodes = None
random_state = 42
n_folds = 5


# In[4]:

t_start = time()
now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = 'crossval-swarm_%s_%s.pickle' % (subject, now)
print(filename)


# In[ ]:

rootfolder = os.path.join(os.environ['SECOND_SIGHT_DATA'], 'shape')
X, y = p2pspatial.load_data(rootfolder, subject=subject, electrodes=electrodes,
                            amplitude=amplitude, random_state=random_state,
                            single_stim=True, verbose=True)
print(X.shape, y.shape)
if len(X) == 0:
    raise ValueError('no data found')


# In[5]:

model_params = {'sampling': 200,
                'csmode': 'gaussian',
                'sensitivity_rule': 'decay',
                'thresh': 1.0}
regressor = p2pspatial.SpatialModelRegressor(**model_params)


# In[6]:

search_params = {'decay_const': (0.001, 10),
                 'cswidth': (1, 10000),
                 'implant_x': (-1500, 1500),
                 'implant_y': (-1000, 1000),
                 'loc_od_x': (13, 17),
                 'loc_od_y': (0, 4),
                 'implant_rot': np.deg2rad((-75, -15))}
pso_options = {'max_iter': 100,
               'min_func': 1e-4}
pso = p2pspatial.model_selection.ParticleSwarmOptimizer(
    regressor, search_params, **pso_options
)


# In[7]:

fit_params = {'scoring_weights': {'orientation': 100.0,
                                  'major_axis_length': 1.0,
                                  'minor_axis_length': 1.0}}
y_test, y_pred, best_params = p2pspatial.model_selection.crossval_predict(
    pso, X, y, fit_params=fit_params, n_folds=n_folds)


# In[10]:

print("Done in %.3fs" % (time() - t_start))


# In[ ]:

specifics = {'subject': subject,
             'amplitude': amplitude,
             'electrodes': electrodes,
             'n_folds': n_folds,
             'regressor': regressor,
             'optimizer': pso,
             'model_params': model_params,
             'search_params': search_params,
             'fit_params': fit_params,
             'now': now,
             'random_state': random_state}
pickle.dump((y_test, y_pred, best_params, specifics), open(filename, 'wb'))
print('Dumped data to %s' % filename)
