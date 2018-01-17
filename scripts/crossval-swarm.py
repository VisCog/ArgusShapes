
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
modelname = ['B', p2pspatial.models.ModelB]
amplitude = 2.0
electrodes = None
random_state = 42
n_folds = 10


# In[4]:

t_start = time()
now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = '%s-crossval-swarm_%s_%s.pickle' % (modelname[0], subject, now)
print(filename)


# In[5]:

rootfolder = os.path.join(os.environ['SECOND_SIGHT_DATA'], 'shape')
X, y = p2pspatial.load_data(rootfolder, subject=subject, electrodes=electrodes,
                            amplitude=amplitude, random_state=random_state,
                            single_stim=True, verbose=False)
print(X.shape, y.shape)
if len(X) == 0:
    raise ValueError('no data found')


# In[6]:

model_params = {}
regressor = modelname[1](**model_params)


# In[ ]:

search_params = {'rho': (20, 1000),
                 'implant_x': (-2000, 1000),
                 'implant_y': (-2000, 2000),
                 'implant_rot': (np.deg2rad(-45), 0)}
pso_options = {'max_iter': 100,
               'min_func': 0.1}
pso = p2pspatial.model_selection.ParticleSwarmOptimizer(
    regressor, search_params, **pso_options
)


# In[ ]:

fit_params = {}
y_test, y_pred, best_params = p2pspatial.model_selection.crossval_predict(
    pso, X, y, fit_params=fit_params, n_folds=n_folds)


# In[ ]:

print("Done in %.3fs" % (time() - t_start))


# In[ ]:

specifics = {'subject': subject,
             'modelname': modelname,
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


# In[ ]:
