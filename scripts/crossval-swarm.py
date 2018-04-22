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
import p2pspatial



# All available models with their corresponding function calls, search
# parameters and model parameters
models = {
    'A': {  # Scoreboard model
        'object': p2pspatial.models.ModelA,
        'search_params': ['rho'],
        'subject_params': ['implant_type',
                           'implant_x', 'implant_y', 'implant_rot',
                           'xrange', 'yrange']
    },
    'B': {  # Scoreboard model with perspective transform
        'object': p2pspatial.models.ModelB,
        'search_params': ['rho'],
        'subject_params': ['implant_type', 'implant_x', 'implant_y', 'implant_rot',
                           'xrange', 'yrange']
    },
    'C': {  # Axon map model: search OD location
        'object': p2pspatial.models.ModelC,
        'search_params': ['rho', 'axlambda'],
        'subject_params': ['implant_type', 'xrange', 'yrange',
                           'loc_od_x', 'loc_od_y',
                           'implant_x', 'implant_y', 'implant_rot']
    },
    'D': {  # Axon map model with perspective transform + predict area/orient
        'object': p2pspatial.models.ModelD,
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
    'implant_x': (-2000, 1000),
    'implant_y': (-1000, 1000),
    'implant_rot': (-np.deg2rad(50), 0)
}

subject_params = {
    'TB': { 
        'implant_type': p2pi.ArgusI,
        'implant_x': -168.699,
        'implant_y': -579.2926,
        'implant_rot': -0.72437198,
        'loc_od_x': 19.84,
        'loc_od_y': 2.04,
        'xrange': (-36.9, 36.9),
        'yrange': (-36.9, 36.9)
    },
    '12-005': {
        'implant_type': p2pi.ArgusII,
        'implant_x': -1462.56,
        'implant_y': -1153.17,
        'implant_rot': -0.485286613,
        'loc_od_x': 17.0,
        'loc_od_y': 0.321,
        'xrange': (-30, 30),
        'yrange': (-22.5, 22.5)
    },
    '51-009': { 
        'implant_type': p2pi.ArgusII,
        'implant_x': 586.0497,
        'implant_y': 93.1473147,
        'implant_rot': -0.6689511,
        'loc_od_x': 16.68115,
        'loc_od_y': 2.9778827,
        'xrange': (-32.5, 32.5),
        'yrange': (-24.4, 24.4)
    },
    '52-001': {
        'implant_type': p2pi.ArgusII,
        'implant_x': -1572.360776,
        'implant_y': 148.6334026,
        'implant_rot': -0.385777021,
        'loc_od_x': 16.1,
        'loc_od_y': 2.5,
        'xrange': (-32, 32),
        'yrange': (-24, 24)
    }
}

drawing = {
    'TB': {
        'major': (1 / 1.34, 1 / 0.939),
        'minor': (1 / 1.19, 1 / 1.62),
        'orient': -9
    },
    '12-005': {
        'major': (1 / 0.632, 1 / 0.686),
        'minor': (1 / 0.704, 1 / 1.35),
        'orient': -16
    },
    '51-009': {
        'major': (1 / 1.38, 1 / 1.34),
        'minor': (1 / 1.06, 1 / 1.94),
        'orient': 4
    },
    '52-001': {
        'major': (1 / 1.39, 1 / 1.47),
        'minor': (1 / 1.76, 1 / 1.61),
        'orient': -14
    }
}


def main():
    # Default values
    amplitude = 2.0
    idx_fold = -1
    n_folds = 5
    n_jobs = -1
    avg_img = False
    adjust_bias = False

    # Parse input arguments
    assert len(sys.argv) >= 3
    modelname = sys.argv[1]
    assert modelname in models
    subject = sys.argv[2]
    assert subject in subject_params
    try:
        longopts = ["n_folds=", "idx_fold=", "n_jobs=", "amplitude=", "avg_img",
                    "adjust_bias"]
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
        elif o == "--adjust_bias":
            adjust_bias = True
        else:
            raise ValueError("Unknown option '%s'='%s'" % (o, a))

    # Generate filename
    t_start = time()
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = '%s_%s_%s%s-swarm_%s.pickle' % (
        subject, modelname, ("adjust_" if adjust_bias else "_"),
        ("shape4fit" if n_folds == 1
         else ("shape4cv%s%s" % (str(n_folds) if n_folds > 0 else "LOO",
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
    print("Adjust bias: %s" % ("on" if adjust_bias else "off"))
    if n_folds == -1:
        print("Leave-one-out cross-validation (idx_fold=%d)" % idx_fold)
    elif n_folds == 1:
        print("Fit all data (n_jobs=%d)" % n_jobs)
    else:
        print("%d-fold cross-validation (idx_fold=%d)" % (n_folds, idx_fold))

    # Load data
    rootfolder = os.path.join(os.environ['SECOND_SIGHT_DATA'], 'shape')
    X, y = p2pspatial.load_data(rootfolder, subject=subject, electrodes=None,
                                amplitude=amplitude, random_state=42,
                                n_jobs=n_jobs, verbose=False)

    # Exclude bistable percepts:
    X, y = p2pspatial.exclude_bistables(X, y)

    # Adjust for drawing bias:
    if adjust_bias:
        y = p2pspatial.adjust_drawing_bias(X, y,
                                           scale_major=drawing[subject]['major'],
                                           scale_minor=drawing[subject]['minor'],
                                           rotate=drawing[subject]['orient']) 
        print('Adjusted for drawing bias:', X.shape, y.shape)
    if len(X) == 0:
        raise ValueError('No data found. Abort.')
    if avg_img:
        X, y = p2pspatial.calc_mean_images(X, y)

    print('Data extracted:', X.shape, y.shape)

    if n_folds == -1:
        n_folds = len(X)
        print('Leave-one-out cross-validation: n_folds=%d' % n_folds)
    if idx_fold != -1:
        print('Processing fold ID %d' % idx_fold)

    # Instantiate model
    model = models[modelname]
    model_params = {'engine': 'cython', 'scheduler': 'threading',
                    'n_jobs': n_jobs, 'xystep': 0.35}
    if 'C' in modelname or 'D' in modelname:
        model_params.update({'axon_pickle': 'axons-%s.pickle' % now})
    for key in model['subject_params']:
        model_params.update({key: subject_params[subject][key]})
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
    pso = p2pspatial.model_selection.ParticleSwarmOptimizer(
        regressor, search_params, **pso_options
    )

    # Launch cross-validation
    fit_params = {}
    if n_folds > 1:
        result = p2pspatial.model_selection.crossval_predict(
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
                 'adjust_bias': adjust_bias,
                 'exetime': t_end - t_start,
                 'random_state': 42}
    pickle.dump((y_test, y_pred, best_params, specifics), open(filename, 'wb'))
    print('Dumped data to %s' % filename)

    if 'axon_pickle' in model_params:
        if os.path.isfile(model_params['axon_pickle']):
            os.remove(model_params['axon_pickle'])


if __name__ == "__main__":
    main()
