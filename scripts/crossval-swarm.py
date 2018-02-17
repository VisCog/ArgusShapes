# coding: utf-8
import sys
import os
import getopt

import numpy as np
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
        'subject_params': ['implant_type', 'implant_x', 'implant_y',
                           'implant_rot']
    },
    'B': {  # Scoreboard model with perspective transform
        'object': p2pspatial.models.ModelB,
        'search_params': ['rho'],
        'subject_params': ['implant_type', 'implant_x', 'implant_y',
                           'implant_rot']
    },
    'C': {  # Axon map model
        'object': p2pspatial.models.ModelC,
        'search_params': ['rho', 'axlambda'],
        'subject_params': ['implant_type', 'implant_x', 'implant_y',
                           'implant_rot', 'loc_od_x', 'loc_od_y']
    },
    'D': {  # Axon map model with perspective transform
        'object': p2pspatial.models.ModelD,
        'search_params': ['rho', 'axlambda'],
        'subject_params': ['implant_type', 'implant_x', 'implant_y',
                           'implant_rot', 'loc_od_x', 'loc_od_y']
    },
    'D2': {  # Axon map model with perspective transform
        'object': p2pspatial.models.ModelD,
        'search_params': ['rho', 'axlambda', 'loc_od_x', 'loc_od_y',
                          'implant_x', 'implant_y', 'implant_rot'],
        'subject_params': ['implant_type']
    },
    'E2': {  # Axon map model with perspective transform
        'object': p2pspatial.models.ModelE,
        'search_params': ['rho', 'axlambda', 'loc_od_x', 'loc_od_y',
                          'implant_x', 'implant_y', 'implant_rot'],
        'subject_params': ['implant_type']
    },
    'F2': {  # Axon map model with perspective transform
        'object': p2pspatial.models.ModelF,
        'search_params': ['rho', 'axlambda', 'loc_od_x', 'loc_od_y',
                          'implant_x', 'implant_y', 'implant_rot'],
        'subject_params': ['implant_type']
    }
}

search_param_ranges = {
    'rho': (10, 3000),
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
        'implant_x': -700,
        'implant_y': 0,
        'implant_rot': -0.700177748,
        'loc_od_x': 15.6,
        'loc_od_y': 0.6
    },
    '12-005': {
        'implant_type': p2pi.ArgusII,
        'implant_x': -1344.36597,
        'implant_y': 537.7463881,  # or should this be minus?
        'implant_rot': -0.664813628,
        'loc_od_x': 15.5,
        'loc_od_y': 1.2
    },
    '51-009': {
        'implant_type': p2pi.ArgusII,
        'implant_x': 398.514982,
        'implant_y': -540.8417613,
        'implant_rot': -0.526951314,
        'loc_od_x': 14.8,
        'loc_od_y': 4.7
    },
    '52-001': {
        'implant_type': p2pi.ArgusII,
        'implant_x': -1147.132944,
        'implant_y': -369.1922119,
        'implant_rot': -0.342307766,
        'loc_od_x': 14.9,
        'loc_od_y': 4.3
    }
}


def main():
    # Default values
    amplitude = 2.0
    n_folds = 5
    n_jobs = -1
    w_scale = 34
    w_rot = 33
    w_dice = 34
    avg_img = False

    # Parse input arguments
    assert len(sys.argv) >= 3
    modelname = sys.argv[1]
    assert modelname in models
    subject = sys.argv[2]
    assert subject in subject_params
    try:
        longopts = ["n_folds=", "n_jobs=", "amplitude=",
                    "w_scale=", "w_rot=", "w_dice=", "avg_img"]
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
        elif o == "--avg_img":
            avg_img = True
        else:
            raise ValueError("Unknown option '%s'='%s'" % (o, a))

    # Generate filename
    t_start = time()
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = '%s-%s-swarm_%s_%s.pickle' % (
        modelname, ("crossval" if n_folds > 1 else "fit"), subject, now
    )
    print("")
    print(filename)
    print("------------------------------------------------------------------")
    print("Subject: %s" % subject)
    print("Model: %s" % modelname)
    print("Amplitude: %.2fx Th" % amplitude)
    if n_folds == -1:
        print("Leave-one-out cross-validation (n_jobs=%d)" % n_jobs)
    elif n_folds == 1:
        print("Fit all data (n_jobs=%d)" % n_jobs)
    else:
        print("%d-fold cross-validation (n_jobs=%d)" % (n_folds, n_jobs))
    print("w_scale=%.2f w_rot=%.2f w_dice=%.2f" % (w_scale, w_rot, w_dice))

    # Load data
    rootfolder = os.path.join(os.environ['SECOND_SIGHT_DATA'], 'shape',
                              subject)
    X, y = p2pspatial.load_data(rootfolder, subject=subject, electrodes=None,
                                amplitude=amplitude, random_state=42,
                                single_stim=True, verbose=False)
    print('Data loaded:', X.shape, y.shape)
    if len(X) == 0:
        raise ValueError('No data found. Abort.')
    if avg_img:
        X, y = p2pspatial.transform_mean_images(X, y)
        print('Data transformed:', X.shape, y.shape)

    if n_folds == -1:
        n_folds = len(X)
        print('Leave-one-out cross-validation: n_folds=%d' % n_folds)

    # Instantiate model
    model = models[modelname]
    model_params = {'engine': 'joblib', 'scheduler': 'threading',
                    'n_jobs': n_jobs,
                    'xystep': 0.5, 'img_thresh': 1.0 / np.sqrt(np.e)}
    # 'w_scale': w_scale, 'w_rot': w_rot, 'w_dice': w_dice}
    for key in model['subject_params']:
        model_params.update({key: subject_params[subject][key]})
    regressor = model['object'](**model_params)
    print('regressor:', regressor)

    # Set up particle swarm
    search_params = {}
    for key in model['search_params']:
        search_params.update({key: search_param_ranges[key]})
    print('search_params:', search_params)
    pso_options = {'max_iter': 100,
                   'min_func': 0.1}
    pso = p2pspatial.model_selection.ParticleSwarmOptimizer(
        regressor, search_params, **pso_options
    )

    # Launch cross-validation
    fit_params = {}
    if n_folds > 1:
        result = p2pspatial.model_selection.crossval_predict(
            pso, X, y, fit_params=fit_params, n_folds=n_folds
        )
        y_test, y_pred, best_params = result
    else:
        pso.fit(X, y, fit_params=fit_params)
        best_params = pso.best_params_
        y_pred = []
        y_test = []

    t_end = time()
    print("Done in %.3fs" % (t_end - t_start))

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
                 'exetime': t_end - t_start,
                 'random_state': 42}
    pickle.dump((y_test, y_pred, best_params, specifics), open(filename, 'wb'))
    print('Dumped data to %s' % filename)


if __name__ == "__main__":
    main()
