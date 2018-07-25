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
    'C': {  # Axon map model with shape loss
        'object': argus_shapes.models.ModelC,
        'search_params': ['rho', 'axlambda'],
        'subject_params': ['implant_type', 'xrange', 'yrange',
                           'loc_od_x', 'loc_od_y',
                           'implant_x', 'implant_y', 'implant_rot']
    },
    'C2': {  # Axon map model with RD loss
        'object': argus_shapes.models.ModelC2,
        'search_params': ['rho', 'axlambda'],
        'subject_params': ['implant_type', 'xrange', 'yrange',
                           'loc_od_x', 'loc_od_y',
                           'implant_x', 'implant_y', 'implant_rot']
    },
    'C3': {  # Axon map model with shape loss and flexible array loc
        'object': argus_shapes.models.ModelC,
        'search_params': ['rho', 'axlambda', 'loc_od_x', 'loc_od_y',
                          'implant_x', 'implant_y', 'implant_rot'],
        'subject_params': ['implant_type', 'xrange', 'yrange',
                           'loc_od_x', 'loc_od_y',
                           'implant_x', 'implant_y', 'implant_rot']
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
    'loc_od_x': (14, 18),
    'loc_od_y': (0.5, 3),
    'implant_x': (-2000, 0),
    'implant_y': (-1500, 500),
    'implant_rot': (-np.deg2rad(50), -np.deg2rad(20))
}

opt_methods = {
    'swarm': argus_shapes.model_selection.ParticleSwarmOptimizer,
    'fmin': argus_shapes.model_selection.FunctionMinimizer
}


def main():
    # Default values
    amplitude = 2.0
    idx_fold = -1
    n_folds = 5
    n_jobs = -1
    groups = None
    avg_img = False
    method = "swarm"

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
                    "method=", "avg_img"]
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
        elif o == "--method":
            assert a in opt_methods.keys()
            method = a
        elif o == "--avg_img":
            avg_img = True
        else:
            raise ValueError("Unknown option '%s'='%s'" % (o, a))

    # Generate filename
    t_start = time()
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = '%s_%s_%s-%s_%s.pickle' % (
        subject, modelname,
        ("trial1fit" if n_folds == 1
         else ("trial1cv%s%s" % (str(n_folds) if n_folds > 0 else "LOO",
                                 ("-" + str(idx_fold)) if idx_fold > -1 else ""))),
        method,
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
    print('Electrodes:', X.electrode.unique())

    if n_folds == -1:
        if not avg_img:
            # Grouped CV
            groups = 'electrode'
            n_folds = len(X.electrode.unique())
        print('Leave-one-electrode-out cross-validation: n_folds=%d' % n_folds)
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
    opt_options = {'max_iter': 50}
    opt = opt_methods[method](regressor, search_params, **opt_options)

    # Launch cross-validation
    fit_params = {}
    if n_folds > 1:
        result = argus_shapes.model_selection.crossval_predict(
            opt, X, y, fit_params=fit_params, n_folds=n_folds,
            idx_fold=idx_fold, groups=groups
        )
        y_test, y_pred, best_params, best_train_score, best_test_score = result
    else:
        opt.fit(X, y, fit_params=fit_params)
        best_params = opt.best_params_
        y_pred = opt.predict(X)
        y_test = y
        best_train_score = opt.score(X, y)
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
                 'optimizer': opt,
                 'optimizer_options': opt_options,
                 'best_train_score': best_train_score,
                 'best_test_score': best_test_score,
                 'model_params': model_params,
                 'search_params': search_params,
                 'fit_params': fit_params,
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
