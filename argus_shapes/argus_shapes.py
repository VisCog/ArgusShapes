from __future__ import absolute_import, division, print_function

from . import imgproc

import os
import glob
import logging
import pickle

import numpy as np
import pandas as pd

import requests
from posixpath import join as urljoin
import zipfile

import pulse2percept as p2p

import skimage
import skimage.io as skio

import sklearn.utils as sklu

try:
    FileNotFoundError
except NameError:
    # Python 2
    FileNotFoundError = IOError

p2p.console.setLevel(logging.ERROR)

__all__ = ["download_file", "fetch_data", "load_data", "load_subjects",
           "calc_mean_images", "is_singlestim_dataframe",
           "extract_best_pickle_files"]


def download_file(url, fname):
    """Downloads a file from the web

    Parameters
    ----------
    url : str
        The URL of the file to be downloaded.
    fname : str
        Local file path where the downloaded content should be stored.
    """
    with open(fname, 'wb') as file:
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(e)
            return
        file.write(response.content)
        print('Successfully created file "%s".' % fname)


def fetch_data(osf_zip_url='https://osf.io/yad7x', save_path=None):
    """Fetches the dataset from the web

    You can view the dataset online at the Open Science Framework (OSF).

    To automatically fetch and unzip the data, click on 'argus_shapes.zip' in
    the 'Files' tab and pass the URL to this function.

    osf_zip_url : str
        The URL to view the zip file at OSF.
    save_path : str or None
        Local file path where to store and unzip the data. If None, will look
        for an environment variable named 'ARGUS_SHAPES_DATA' and store it
        there.
    """
    if save_path is None:
        # Look for environment variable
        if not hasattr(os.environ, 'ARGUS_SHAPES_DATA'):
            raise ValueError(('No such environment variable: '
                              '"ARGUS_SHAPES_DATA". Please explicitly '
                              'specify a path.'))
        save_path = os.environ['ARGUS_SHAPES_DATA']

    # Create save path if necessary:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('Successfully created path %s' % save_path)

    # Save to this zip file:
    fzipname = os.path.join(save_path, 'argus_shapes.zip')

    # Construct a download URL using forward slashes (via posixpath join)
    # and download the data to a local file:
    download_file(urljoin(osf_zip_url, 'download'), fzipname)

    # Unzip the file:
    fzip = zipfile.ZipFile(fzipname, 'r')
    fzip.extractall(save_path)
    fzip.close()
    print('Successfully unzipped file "%s".' % fzipname)


def load_data(fname, subject=None, electrodes=None, amp=None, add_cols=[],
              random_state=42):
    """Loads shuffled shape data

    Shape data is supposed to live in a .csv file with the following columns:
    - `PTS_AMP`: Current amplitude as multiple of threshold current
    - `PTS_ELECTRODE`: Name of the electrode (e.g., 'A1')
    - `PTS_FILE`: Name of the drawing file
    - `PTS_FREQ`: Stimulation frequency (Hz)
    - `PTS_PULSE_DUR`: Stimulation pulse duration (ms)
    - `date`: Date that data was recorded
    - `stim_class`: 'SingleElectrode' or 'MultiElectrode'
    - `subject_id`: must match the subject data .csv (e.g., 'S1')

    If there

    Parameters
    ----------
    fname : str
        Path to .csv file.
    subject : str or None, default: None
        Only load data from a particular subject. Set to None to load data from
        all subjects.
    electrodes : list or None, default: None
        Only load data from a particular set of electrodes. Set to None to load
        data from all electrodes
    amp : float or None, default: None
        Only load data with a particular current amplitude. Set to None to load
        data with all current amplitudes.
    add_cols : list, optional, default: []
        List specifying additional columns you want to extract.
    random_state : int or None, default: 42
        Seed for the random number generator. Set to None to prevent shuffling.

    Returns
    -------
    df : pd.DataFrame
        The parsed .csv file loaded as a DataFrame, optionally with shuffled
        rows.

    """
    # Read data and make sure it's a single-stim file:
    data = pd.read_csv(fname)
    is_singlestim = is_singlestim_dataframe(data)

    # Make sure .csv file has all necessary columns:
    has_cols = set(data.columns)
    needs_cols = set(['PTS_AMP', 'PTS_FILE', 'PTS_FREQ', 'PTS_PULSE_DUR',
                      'date', 'stim_class', 'subject_id'] + add_cols)
    if bool(needs_cols - has_cols):
        err = "The following required columns are missing: "
        err += ", ".join(needs_cols - has_cols)
        raise ValueError(err)

    # Only load data from a particular subject:
    if subject is not None:
        data = data[data.subject_id == subject]

    # Only load data from a particular set of electrodes:
    if electrodes is not None:
        if not isinstance(electrodes, (list, np.ndarray)):
            raise ValueError("`electrodes` must be a list or NumPy array")
        idx = np.zeros(len(data), dtype=np.bool)
        if is_singlestim:
            for e in electrodes:
                idx = np.logical_or(idx, data.PTS_ELECTRODE == e)
        else:
            for e in electrodes:
                idx = np.logical_or(idx, data.PTS_ELECTRODE1 == e)
                idx = np.logical_or(idx, data.PTS_ELECTRODE2 == e)
        data = data[idx]

    # Only load data with a particular current amplitude:
    if amp is not None:
        data = data[np.isclose(data.PTS_AMP, amp)]

    # Shuffle data if random seed is set:
    if random_state is not None:
        data = sklu.shuffle(data, random_state=random_state)

    # Build data matrix:
    rows = []
    for idx, row in data.iterrows():
        # Extract shape descriptors from phosphene drawing:
        if pd.isnull(row['PTS_FILE']):
            if is_singlestim:
                e_s = "ID %d %s: 'PTS_FILE' is empty" % (idx,
                                                         row['PTS_ELECTRODE'])
            else:
                e_s = ("ID %d %s, %s: 'PTS_FILE' is "
                       "empty") % (idx, row['PTS_ELECTRODE1'],
                                   row['PTS_ELECTRODE2'])
            raise FileNotFoundError(e_s)
        else:
            try:
                img = skio.imread(os.path.join(os.path.dirname(fname),
                                               row['PTS_FILE']), as_grey=True)
                img = skimage.img_as_float(img)
            except FileNotFoundError:
                try:
                    img = skio.imread(row['PTS_FILE'], as_grey=True)
                except FileNotFoundError:
                    s = ('Column "PTS_FILE" must either specify an absolute '
                         'path or a relative path that starts in the '
                         'directory of `fname`.')
                    raise FileNotFoundError(s)
        columns = {
            'subject': row['subject_id'],
            'filename': row['PTS_FILE'],
            'image': img,
            'img_shape': img.shape,
            'stim_class': row['stim_class'],
            'amp': row['PTS_AMP'],
            'freq': row['PTS_FREQ'],
            'pdur': row['PTS_PULSE_DUR'],
            'date': row['date']
        }
        if is_singlestim:
            columns.update({'electrode': row['PTS_ELECTRODE']})
        else:
            columns.update({'electrode1': row['PTS_ELECTRODE1'],
                            'electrode2': row['PTS_ELECTRODE2']})
        # Add shape descriptors:
        props = imgproc.calc_shape_descriptors(img)
        columns.update(props)
        # Add additional columns:
        for col in add_cols:
            columns.update({col: row[col]})
        rows.append(columns)
    return pd.DataFrame(rows, index=data.index)


def load_subjects(fname):
    """Loads subject data

    Subject data is supposed to live in a .csv file with the following columns:
    - `subject_id`: must match the shape data .csv (e.g., 'S1')
    - `second_sight_id`: corresponding identifier (e.g., '11-001')
    - `implant_type_str`: either 'ArgusI' or 'ArgusII'
    - (`implant_x`, `implant_y`): x, y coordinates of array center (um)
    - (`loc_od_x`, `loc_od_y`): x, y coordinates of the optic disc center (deg)
    - (`xmin`, `xmax`): screen width at arm's length (dva)
    - (`ymin`, `ymax`): screen height at arm's length (dva)

    Parameters
    ----------
    fname : str
        Path to .csv file.

    Returns
    -------
    df : pd.DataFrame
        The parsed .csv file loaded as a DataFrame.

    """
    # Make sure all required columns are present:
    df = pd.read_csv(fname, index_col='subject_id')
    has_cols = set(df.columns)
    needs_cols = set(['implant_type_str', 'implant_x', 'implant_y', 'loc_od_x',
                      'loc_od_y', 'xmin', 'xmax', 'ymin', 'ymax'])
    if bool(needs_cols - has_cols):
        err = "The following required columns are missing: "
        err += ", ".join(needs_cols - has_cols)
        raise ValueError(err)

    # Make sure array types are valid:
    if bool(set(df.implant_type_str.unique()) - set(['ArgusI', 'ArgusII'])):
        raise ValueError(("'implant_type_str' must be either 'ArgusI' or "
                          "'ArgusII' for all subjects."))

    # Calculate screen ranges from (xmin, xmax), (ymin, ymax):
    df['xrange'] = pd.Series([(a, b) for a, b in zip(df['xmin'], df['xmax'])],
                             index=df.index)
    df['yrange'] = pd.Series([(a, b) for a, b in zip(df['ymin'], df['ymax'])],
                             index=df.index)

    # Load array type from pulse2percept:
    df['implant_type'] = pd.Series([(p2p.implants.ArgusI if i == 'ArgusI'
                                     else p2p.implants.ArgusII)
                                    for i in df['implant_type_str']],
                                   index=df.index)
    return df.drop(columns=['xmin', 'xmax', 'ymin', 'ymax',
                            'implant_type_str'])


def is_singlestim_dataframe(data):
    """Determines whether a DataFrame contains single or multi electrode stim

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the shape data

    Returns
    -------
    is_singlestim : bool
        Whether DataFrame contains single stim (True) or not (False).
    """
    if not np.any([c in data.columns for c in ['PTS_ELECTRODE', 'electrode',
                                               'PTS_ELECTRODE1', 'electrode1',
                                               'PTS_ELECTRODE2',
                                               'electrode2']]):
        raise ValueError(('Incompatible DataFrame. Must contain one of '
                          'these columns: PTS_ELECTRODE, PTS_ELECTRODE1, '
                          'PTS_ELECTRODE2, electrode, electrode1, electrode2'))
    is_singlestim = (('PTS_ELECTRODE' in data.columns or
                      'electrode' in data.columns) and
                     ('PTS_ELECTRODE1' not in data.columns or
                      'electrode1' in data.columns) and
                     ('PTS_ELECTRODE2' not in data.columns or
                      'electrode2' in data.columns))
    return is_singlestim


def _calcs_mean_image(Xy, groupcols, thresh=True, max_area=np.inf):
    """Private helper function to calculate a mean image"""
    for col in groupcols:
        assert len(Xy[col].unique()) == 1

    # Calculate mean image
    images = Xy.image
    img_avg = None

    for img in images:
        if img_avg is None:
            img_avg = np.zeros_like(img, dtype=float)
        img_avg += imgproc.center_phosphene(img)

    # Adjust to [0, 1]
    if img_avg.max() > 0:
        img_avg /= img_avg.max()
    # Threshold if required:
    if thresh:
        img_avg = imgproc.get_thresholded_image(img_avg, thresh='otsu')
    # Move back to its original position:
    img_avg = imgproc.center_phosphene(img_avg, center=(np.mean(Xy.y_center),
                                                        np.mean(Xy.x_center)))

    # Remove ambiguous (trial-related) parameters:
    columns = {
        'image': img_avg,
        'img_shape': img_avg.shape
    }
    for col in groupcols:
        columns.update({col: Xy[col].unique()[0]})

    # Calculate shape descriptors:
    descriptors = imgproc.calc_shape_descriptors(img_avg)
    columns.update(descriptors)

    # Compare area of mean image to the mean of trial images: If smaller than
    # some fraction, skip:
    if descriptors['area'] > max_area * np.mean(Xy.area):
        return None

    return columns


def calc_mean_images(Xy, groupby=['subject', 'amp', 'electrode'], thresh=True,
                     max_area=np.inf):
    """Extract mean images on an electrode from all raw trial drawings

    Parameters
    ----------
    Xy: pd.DataFrame
        Data matrix, raw trial data
    groupby : list
        List of columns by which to group data matrix
    thresh: bool, optional, default: True
        Whether to binarize the averaged image.
    max_area: float, optional, default: inf
        Skip if mean image has area larger than a factor `max_area`
        of the mean of the individual images. A large area of the mean
        image indicates poor averaging: instead of maintaining area,
        individual nonoverlapping images are added.

    Returns
    =======
    Xymu: pd.DataFrame
        Data matrix, single entry per electrode
    """
    Xymu = []
    for _, data in Xy.groupby(groupby):
        row = _calcs_mean_image(
            data, groupby, thresh=thresh, max_area=max_area)
        if row is not None:
            Xymu.append(row)

    return pd.DataFrame(Xymu)


def _extracts_score_from_pickle(file, col_score, col_groupby):
    """Private helper function to extract the score from a pickle file"""
    _, _, _, specifics = pickle.load(open(file, 'rb'))
    assert np.all([g in specifics for g in col_groupby])
    assert col_score in specifics
    params = specifics['optimizer'].get_params()
    # TODO: make this work for n_folds > 1
    row = {
        'file': file,
        'greater_is_better': params['estimator__greater_is_better'],
        col_score: specifics[col_score][0]
    }
    for g in col_groupby:
        row.update({g: specifics[g]})
    return row


def extract_best_pickle_files(results_dir, col_score, col_groupby):
    """Finds the fitted models with the best scores

    For all pickle files in a directory (supposedly containing the results of
    different parameter fits), this function returns a list of pickle files
    that have the best score.

    The `col_groupby` argument can be used to find the best scores for each
    cross-validation fold (e.g., group by ['electrode', 'idx_fold']).

    Parameters
    ----------
    results_dir : str
        Path to results directory.
    col_score : str
        Name of the DataFrame column that contains the score.
    col_groupby : list
        List of columns by which to group the DataFrame
        (e.g., ['electrode', 'idx_fold']).

    Returns
    -------
    files : list
        A list of pickle files with the best scores.

    """
    # Extract relevant info from pickle files:
    pickle_files = np.sort(glob.glob(os.path.join(results_dir, '*.pickle')))
    if len(pickle_files) == 0:
        raise FileNotFoundError("No pickle files found in %s" % results_dir)
    data = p2p.utils.parfor(_extracts_score_from_pickle, pickle_files,
                            func_args=[col_score, col_groupby])
    # Convert to DataFrame:
    df = pd.DataFrame(data)
    # Make sure all estimator use the same scoring logic:
    assert np.isclose(np.var(df.greater_is_better), 0)
    # Find the rows that minimize/maximize the score:
    if df.loc[0, 'greater_is_better']:
        # greater score is better: maximize
        res = df.loc[df.groupby(col_groupby)[col_score].idxmax()]
    else:
        # greater is worse: minimize
        res = df.loc[df.groupby(col_groupby)[col_score].idxmin()]
    # Return list of files:
    return res.file.tolist()
