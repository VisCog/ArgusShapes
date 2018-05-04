# coding: utf-8
import sys
import os
import getopt

import numpy as np
import pandas as pd
import pickle
from time import time
from datetime import datetime

import pulse2percept.implants as p2pi
import argus_shapes


# All available models with their corresponding function calls, search
# parameters and model parameters
models = {
    'A': {  # Scoreboard model
        'object': argus_shapes.models.ModelA,
        'search_params': ['rho'],
        'subject_params': ['implant_type',
                           'implant_x', 'implant_y', 'implant_rot',
                           'xrange', 'yrange']
    },
    'B': {  # Scoreboard model with perspective transform
        'object': argus_shapes.models.ModelB,
        'search_params': ['rho'],
        'subject_params': ['implant_type', 'implant_x', 'implant_y', 'implant_rot',
                           'xrange', 'yrange']
    },
    'C': {  # Axon map model: search OD location
        'object': argus_shapes.models.ModelC,
        'search_params': ['rho', 'axlambda'],
        'subject_params': ['implant_type', 'xrange', 'yrange',
                           'loc_od_x', 'loc_od_y',
                           'implant_x', 'implant_y', 'implant_rot']
    },
    'C2': {  # Axon map model: search OD location
        'object': argus_shapes.models.ModelC,
        'search_params': ['rho', 'axlambda', 'loc_od_x', 'loc_od_y',
                          'implant_x', 'implant_y', 'implant_rot'],
        'subject_params': ['implant_type', 'xrange', 'yrange']
    },
    'D': {  # Axon map model with perspective transform + predict area/orient
        'object': argus_shapes.models.ModelD,
        'search_params': ['rho', 'axlambda'],
        'subject_params': ['implant_type', 'xrange', 'yrange',
                           'loc_od_x', 'loc_od_y',
                           'implant_x', 'implant_y', 'implant_rot']
    }
}

search_param_ranges = {
    'rho': (50, 3000),
    'axlambda': (10, 3000),
    'loc_od_x': (13, 17),
    'loc_od_y': (0, 5),
    'implant_x': (-2000, 500),
    'implant_y': (-1500, 500),
    'implant_rot': (-np.deg2rad(65), -np.deg2rad(25))
}


def main():
    # Default values
    amplitude = 2.0
    idx_fold = -1
    n_folds = 5
    n_jobs = -1
    avg_img = False

    datafolder = os.path.join(os.environ['DATA_ROOT'], 'argus_shapes')
    subjects = argus_shapes.load_subjects(os.path.join(datafolder,
                                                       'subjects.csv'))

    # Parse input arguments
    assert len(sys.argv) >= 3
    modelname = sys.argv[1]
    assert modelname in models
    subject = sys.argv[2]
    assert subject in subjects.index
    try:
        longopts = ["n_folds=", "idx_fold=", "n_jobs=", "amplitude=",
                    "avg_img"]
        opts, args = getopt.getopt(sys.argv[3:], "", longopts=longopts)
    except getopt.GetoptError as err:
        raise RuntimeError(err)
    for o, a in opts:
        if o == "--n_folds":
            n_folds = int(a)
        elif o == "--idx_fold":
            idx_fold = int(a)
        elif o == "--n_jobs":
            n_jobs = int(a)
        elif o == "--amplitude":
            amplitude = float(a)
        elif o == "--avg_img":
            avg_img = True
        else:
            raise ValueError("Unknown option '%s'='%s'" % (o, a))

    # Generate filename
    t_start = time()
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = '%s_%s_%s-swarm_%s.pickle' % (
        subject, modelname,
        ("shape8fit" if n_folds == 1
         else ("shape8cv%s%s" % (str(n_folds) if n_folds > 0 else "LOO",
                                 ("-" + str(idx_fold)) if idx_fold > -1 else ""))),
        now
    )
    print("")
    print(filename)
    print("------------------------------------------------------------------")
    print("Subject: %s" % subject)
    print("Model: %s" % modelname)
    print("Amplitude: %.2fx Th" % amplitude)
    print("Processing: %s (n_jobs=%d)" % ("serial" if n_jobs == 1 else "parallel", n_jobs))
    print("Average image: %s" % ("on" if avg_img else "off"))
    if n_folds == -1:
        print("Leave-one-out cross-validation (idx_fold=%d)" % idx_fold)
    elif n_folds == 1:
        print("Fit all data (n_jobs=%d)" % n_jobs)
    else:
        print("%d-fold cross-validation (idx_fold=%d)" % (n_folds, idx_fold))

    # Load data
    X, y = argus_shapes.load_data(os.path.join(datafolder,
                                               'drawings_single.csv'),
                                  subject=subject, electrodes=None,
                                  amp=amplitude, random_state=42)
    if len(X) == 0:
        raise ValueError('No data found. Abort.')

    # Calculate mean images:
    if avg_img:
        X, y = argus_shapes.calc_mean_images(X, y, max_area=1.5)

    print('Data extracted:', X.shape, y.shape)
    print(X.electrode.unique())

    if n_folds == -1:
        n_folds = len(X)
        print('Leave-one-out cross-validation: n_folds=%d' % n_folds)
    if idx_fold != -1:
        print('Processing fold ID %d' % idx_fold)

    # Instantiate model
    model = models[modelname]
    model_params = {'engine': 'cython', 'scheduler': 'threading',
                    'n_jobs': n_jobs, 'xystep': 0.25}
    if 'C' in modelname or 'D' in modelname:
        model_params.update({'axon_pickle': 'axons-%s.pickle' % now})
    for key in model['subject_params']:
        value = subjects.loc[subject, key]
        model_params.update({key: value})
    regressor = model['object'](**model_params)
    print('regressor:', regressor)

    # Set up particle swarm
    search_params = {}
    for key in model['search_params']:
        search_params.update({key: search_param_ranges[key]})
    print('search_params:', search_params)
    pso_options = {'max_iter': 50,
                   'min_func': 0.1,
                   'min_step': 0.1}
    pso = argus_shapes.model_selection.ParticleSwarmOptimizer(
        regressor, search_params, **pso_options
    )

    # Launch cross-validation
    fit_params = {}
    if n_folds > 1:
        result = argus_shapes.model_selection.crossval_predict(
            pso, X, y, fit_params=fit_params, n_folds=n_folds,
            idx_fold=idx_fold,
        )
        y_test, y_pred, best_params, best_train_score, best_test_score = result
    else:
        pso.fit(X, y, fit_params=fit_params)
        best_params = pso.best_params_
        y_pred = pso.predict(X)
        y_test = y
        best_train_score = pso.score(X, y)
        best_test_score = None

    t_end = time()
    print("Done in %.3fs" % (t_end - t_start))

    # Store results
    specifics = {'subject': subject,
                 'modelname': modelname,
                 'amplitude': amplitude,
                 'electrodes': None,
                 'n_folds': n_folds,
                 'idx_fold': idx_fold,
                 'regressor': regressor,
                 'optimizer': pso,
                 'optimizer_options': pso_options,
                 'best_train_score': best_train_score,
                 'best_test_score': best_test_score,
                 'model_params': model_params,
                 'search_params': search_params,
                 'fit_params': fit_params,
                 'drawing': drawing[subject],
                 'now': now,
                 'avg_img': avg_img,
                 'exetime': t_end - t_start,
                 'random_state': 42}
    pickle.dump((y_test, y_pred, best_params, specifics), open(filename, 'wb'))
    print('Dumped data to %s' % filename)

    if 'axon_pickle' in model_params:
        if os.path.isfile(model_params['axon_pickle']):
            os.remove(model_params['axon_pickle'])


if __name__ == "__main__":
    main()
