# coding: utf-8
import sys
import os
import getopt

import numpy as np
import pickle
from time import time
from datetime import datetime

import pulse2percept as p2p
import p2pspatial

# All available models with their corresponding objects to call:
models = {
    'A': p2pspatial.models.ModelA,
    'B': p2pspatial.models.ModelB,
    # 'C': p2pspatial.models.ModelC,
    # 'D': p2pspatial.models.ModelD
}

# All search parameters for each individual model:
models_search_params = {
    # Model A: Scoreboard model:
    'A': {
        'rho': (20, 1000)
    },
    # Model B: Scoreboard model with perspective transform:
    'B': {
        'rho': (20, 1000),
        'implant_x': (-2000, 1000),
        'implant_y': (-2000, 2000),
        'implant_rot': (np.deg2rad(-45), 0)
    },
}


def main():
    # Default values
    amplitude = 2.0
    n_folds = 5
    n_jobs = -1
    w_scale = 34
    w_rot = 33
    w_dice = 34

    # Parse input arguments
    assert len(sys.argv) >= 3
    modelname = sys.argv[1]
    assert modelname in models
    subject = sys.argv[2]
    assert subject in ['12-005', '51-009', '52-001']
    try:
        longopts = ["n_folds=", "n_jobs=", "amplitude=",
                    "w_scale=", "w_rot=", "w_dice="]
        opts, args = getopt.getopt(sys.argv[3:], "", longopts=longopts)
    except getopt.GetoptError as err:
        raise RuntimeError(err)
    for o, a in opts:
        if o == "--n_folds":
            n_folds = int(a)
        elif o == "--n_jobs":
            n_jobs = int(a)
        elif o == "--amplitude":
            amplitude = float(a)
        elif o == "--w_scale":
            w_scale = float(a)
        elif o == "--w_rot":
            w_rot = float(a)
        elif o == "--w_dice":
            w_dice = float(a)
        else:
            raise ValueError("Unknown option '%s'='%s'" % (o, a))

    # Generate filename
    t_start = time()
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = '%s-crossval-swarm_%s_%s.pickle' % (modelname, subject, now)
    print("")
    print(filename)
    print("------------------------------------------------------------------")
    print("Subject: %s" % subject)
    print("Model: %s" % modelname)
    print("Amplitude: %.2fx Th" % amplitude)
    print("%d-fold cross-validation (n_jobs=%d)" % (n_folds, n_jobs))
    print("w_scale=%.2f w_rot=%.2f w_dice=%.2f" % (w_scale, w_rot, w_dice))

    # Load data
    rootfolder = os.path.join(os.environ['SECOND_SIGHT_DATA'], 'shape')
    X, y = p2pspatial.load_data(rootfolder, subject=subject, electrodes=None,
                                amplitude=amplitude, random_state=42,
                                single_stim=True, verbose=False)
    print('Data loaded:', X.shape, y.shape)
    if len(X) == 0:
        raise ValueError('No data found. Abort.')

    # Instantiate model
    model_params = {'engine': 'joblib', 'scheduler': 'threading',
                    'n_jobs': n_jobs,
                    'w_scale': w_scale, 'w_rot': w_rot, 'w_dice': w_dice}
    regressor = models[modelname](**model_params)

    # Set up particle swarm
    search_params = models_search_params[modelname]
    pso_options = {'max_iter': 100,
                   'min_func': 0.1}
    pso = p2pspatial.model_selection.ParticleSwarmOptimizer(
        regressor, search_params, **pso_options
    )

    # Launch cross-validation
    fit_params = {}
    y_test, y_pred, best_params = p2pspatial.model_selection.crossval_predict(
        pso, X, y, fit_params=fit_params, n_folds=n_folds)

    print("Done in %.3fs" % (time() - t_start))

    # Store results
    specifics = {'subject': subject,
                 'modelname': modelname,
                 'amplitude': amplitude,
                 'electrodes': None,
                 'n_folds': n_folds,
                 'w_scale': w_scale,
                 'w_rot': w_rot,
                 'w_dice': w_dice,
                 'regressor': regressor,
                 'optimizer': pso,
                 'model_params': model_params,
                 'search_params': search_params,
                 'fit_params': fit_params,
                 'now': now,
                 'random_state': 42}
    pickle.dump((y_test, y_pred, best_params, specifics), open(filename, 'wb'))
    print('Dumped data to %s' % filename)


if __name__ == "__main__":
    main()
