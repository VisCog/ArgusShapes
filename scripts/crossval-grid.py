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

from sklearn.model_selection import ParameterGrid


class ValidShapeLoss(p2pspatial.models.ShapeLossMixin):

    def _calcs_el_curr_map(self, electrode):
        return 0

    def build_ganglion_cell_layer(self):
        pass


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
        'search_params': ['rho', 'implant_x', 'implant_y', 'implant_rot'],
        'subject_params': ['implant_type', 'xrange', 'yrange']
    },
    'C': {  # Axon map model: search OD location
        'object': p2pspatial.models.ModelC,
        'search_params': ['rho', 'axlambda', 'loc_od_x', 'loc_od_y',
                          'implant_x', 'implant_y', 'implant_rot'],
        'subject_params': ['implant_type', 'xrange', 'yrange']
    },
    'D': {  # Axon map model with perspective transform + predict area/orient
        'object': p2pspatial.models.ModelD,
        'search_params': ['rho', 'axlambda', 'loc_od_x', 'loc_od_y',
                          'implant_x', 'implant_y', 'implant_rot'],
        'subject_params': ['implant_type', 'xrange', 'yrange']
    }
}

search_params = {
    'rho': (50, 3000, 500),
    'axlambda': (10, 3000, 500),
    'loc_od_x': (13, 17, 2),
    'loc_od_y': (0, 5, 2),
    'implant_x': (-2000, 1000, 1000),
    'implant_y': (-1000, 1000, 1000),
    'implant_rot': (-np.deg2rad(50), 0, 10)
}

subject_params = {
    'TB': {
        'implant_type': p2pi.ArgusI,
        'implant_x': -700,
        'implant_y': 0,
        'implant_rot': -0.700177748,
        'loc_od_x': 15.6,
        'loc_od_y': 0.6,
        'xrange': (-36.9, 36.9),
        'yrange': (-36.9, 36.9)
    },
    '12-005': {
        'implant_type': p2pi.ArgusII,
        'implant_x': -1344.36597,
        'implant_y': 537.7463881,  # or should this be minus?
        'implant_rot': -0.664813628,
        'loc_od_x': 15.5,
        'loc_od_y': 1.2,
        'xrange': (-30, 30),
        'yrange': (-22.5, 22.5)
    },
    '51-009': {
        'implant_type': p2pi.ArgusII,
        'implant_x': 398.514982,
        'implant_y': -540.8417613,
        'implant_rot': -0.526951314,
        'loc_od_x': 14.8,
        'loc_od_y': 4.7,
        'xrange': (-32.5, 32.5),
        'yrange': (-24.4, 24.4)
    },
    '52-001': {
        'implant_type': p2pi.ArgusII,
        'implant_x': -1147.132944,
        'implant_y': -369.1922119,
        'implant_rot': -0.342307766,
        'loc_od_x': 14.9,
        'loc_od_y': 4.3,
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
        longopts = ["n_folds=", "n_jobs=", "amplitude=", "avg_img",
                    "adjust_bias"]
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
        elif o == "--avg_img":
            avg_img = True
        elif o == "--adjust_bias":
            adjust_bias = True
        else:
            raise ValueError("Unknown option '%s'='%s'" % (o, a))

    # Generate filename
    t_start = time()
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = '%s-%s-%s-grid_%s.pickle' % (
        subject, modelname, ("shapefit" if n_folds == 1 else "shapecv"), now
    )
    print("")
    print(filename)
    print("------------------------------------------------------------------")
    print("Subject: %s" % subject)
    print("Model: %s" % modelname)
    print("Amplitude: %.2fx Th" % amplitude)
    print("Average image: %s" % ("on" if avg_img else "off"))
    print("Adjust bias: %s" % ("on" if adjust_bias else "off"))
    if n_folds == -1:
        print("Leave-one-out cross-validation (n_jobs=%d)" % n_jobs)
    elif n_folds == 1:
        print("Fit all data (n_jobs=%d)" % n_jobs)
    else:
        print("%d-fold cross-validation (n_jobs=%d)" % (n_folds, n_jobs))

    # Load data
    rootfolder = os.path.join(os.environ['SECOND_SIGHT_DATA'], 'shape')
    X, y = p2pspatial.load_data(rootfolder, subject=subject, electrodes=None,
                                amplitude=amplitude, random_state=42,
                                verbose=False)
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
        print('Images averaged:', X.shape, y.shape)

    shapeloss = ValidShapeLoss()
    y = pd.DataFrame([shapeloss._predicts_target_values(row['electrode'],
                                                        row['image'])
                      for _, row in y.iterrows()], index=X.index)
    assert 'eccentricity' in y.columns
    print('Image props extracted:', X.shape, y.shape)

    if n_folds == -1:
        n_folds = len(X)
        print('Leave-one-out cross-validation: n_folds=%d' % n_folds)

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

    # Set up grid search
    param_expand = {}
    for key in model['search_params']:
        value = search_params[key]
        n_steps = np.ceil((value[1] - value[0]) / value[2])
        param_expand.update({key: np.linspace(*value[:2], num=n_steps)})
    n_iter = [len(v) for _, v in param_expand.items()]
    print('Running %s=%d iters' % ('x'.join([str(n) for n in n_iter]),
                                   np.prod(n_iter)))

    grid = p2pspatial.model_selection.GridSearchOptimizer(
        regressor, ParameterGrid(param_expand)
    )

    # Launch cross-validation
    fit_params = {}
    if n_folds > 1:
        result = p2pspatial.model_selection.crossval_predict(
            grid, X, y, fit_params=fit_params, n_folds=n_folds
        )
        y_test, y_pred, best_params, best_score = result
    else:
        grid.fit(X, y, fit_params=fit_params)
        best_params = grid.best_params_
        y_pred = grid.predict(X)
        y_test = y

    t_end = time()
    print("Done in %.3fs" % (t_end - t_start))

    # Store results
    specifics = {'subject': subject,
                 'modelname': modelname,
                 'amplitude': amplitude,
                 'electrodes': None,
                 'n_folds': n_folds,
                 'regressor': regressor,
                 'optimizer': grid,
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
