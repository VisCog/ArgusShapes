from __future__ import absolute_import, division, print_function
import os
import glob

import numpy as np
import pandas as pd

import pulse2percept as p2p

import skimage.io as skio
import skimage.transform as skit
import skimage.measure as skim
import sklearn.base as sklb

from .due import due, Doi

__all__ = ["load_data", "DataPreprocessor", "SpatialSimulation",
           "get_region_props"]


# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi("10.1167/13.9.30"),
         description="Template project for small scientific Python projects",
         tags=["reference-implementation"],
         path='p2pspatial')


def get_region_props(img):
    regions = skim.regionprops(skim.label(img))
    return regions[0] if len(regions) == 1 else regions


def load_data(folder, random_state=None):
    search_pattern = os.path.join(folder, '**', '*_rawDataFileList_*')
    dfs = []
    for fname in glob.iglob(search_pattern, recursive=True):
        tmp = pd.read_csv(fname)
        tmp['Folder'] = os.path.dirname(fname)
        dfs.append(tmp)
    df = pd.concat(dfs)
    if random_state is not None:
        df = shuffle(df, random_state=random_state)
    return df


class DataPreprocessor(sklb.TransformerMixin):

    def __init__(self, subject=None, electrodes=None, date=None, verbose=True):
        self.subject = subject
        self.electrodes = electrodes
        self.date = date
        self.verbose = verbose

    def get_params(self, deep=True):
        return {'subject': self.subject,
                'electrodes': self.electrodes,
                'date': self.date,
                'verbose': self.verbose}

    def set_params(self, **params):
        for param, value in six.iteritems(params):
            setattr(self, param, value)

    def fit(self, *_):
        return self

    def transform(self, X, *_):
        assert isinstance(X, pd.core.frame.DataFrame)

        features = []
        targets = []
        for _, row in X.iterrows():
            # Split the data strings to extract subject, electrode, etc.
            fname = row['Filename']
            date = fname.split('_')[0]
            params = row['Params'].split(' ')
            stim = params[0].split('_')
            if len(params) < 2 or len(stim) < 2:
                if self.verbose:
                    print('Could not parse row:', row['Filename'],
                          row['Params'])
                continue
            if self.subject is not None and stim[0] != self.subject:
                continue
            if self.electrodes is not None:
                if params[1] not in self.electrodes:
                    continue
            if self.date is not None and date != self.date:
                continue

            # Find the Hu momemnts of the image: Calculate area in deg^2, but
            # operate on image larger than 1px = 1deg so that thin lines
            # are still visible
            sc_fact = 4
            img = skio.imread(os.path.join(
                row['Folder'], row['Filename']), as_grey=True)
            img = skit.resize(img, (41 * sc_fact, 61 * sc_fact))
            props = get_region_props(img)
            if isinstance(props, list):
                if len(props) == 0:
                    if self.verbose:
                        print('Found empty props:', row['Folder'],
                              row['Filename'])
                    continue

                areas = np.array([p.area for p in props])
                idx = np.argmax(areas)
                props = props[idx]
                if self.verbose:
                    print('Found multiple props:', row['Folder'],
                          row['Filename'])
                    print('Chose props[%d] with area %f' % (idx, props.area))

            # Assemble all feature values in a dict
            feat = {'filename': fname,
                    'folder': row['Folder'],
                    'param_str': row['Params'],
                    'subject': stim[0],
                    'electrode': params[1],
                    'stim_class': stim[1],
                    'date': date,
                    'area': props.area / sc_fact ** 2,
                    'orientation': props.orientation,
                    'major_axis_length': props.major_axis_length / sc_fact,
                    'minor_axis_length': props.minor_axis_length / sc_fact}
            features.append(feat)
            targets.append(props.moments_hu)
        return features, targets


class SpatialSimulation(p2p.Simulation):

    def set_ganglion_cell_layer(self):
        pass

    def pulse2percept(self, electrode):
        pass
