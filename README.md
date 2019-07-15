[![DOI](https://zenodo.org/badge/109446024.svg)](https://zenodo.org/badge/latestdoi/109446024)
[![license](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/uwescience/pulse2percept/blob/master/LICENSE)
[![Data](https://img.shields.io/badge/data-osf.io-lightgrey.svg)](https://osf.io/dw9nz)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/a2827a9c61ba41a8a9cf90ef21a956c0)](https://app.codacy.com/app/mbeyeler/ArgusShapes?utm_source=github.com&utm_medium=referral&utm_content=VisCog/ArgusShapes&utm_campaign=Badge_Grade_Dashboard)
[![Build Status](https://travis-ci.org/VisCog/ArgusShapes.svg?branch=master)](https://travis-ci.org/VisCog/ArgusShapes)
[![Coverage Status](https://coveralls.io/repos/github/VisCog/ArgusShapes/badge.svg?branch=master)](https://coveralls.io/github/VisCog/ArgusShapes?branch=master)

# ArgusShapes

`ArgusShapes` is a package used to predict phosphene shape in epiretinal prostheses.

Please cite as:

> M. Beyeler, D. Nanduri, J. D. Weiland, A. Rokem, G. M. Boynton, I. Fine (2019).
> A model of ganglion axon pathways accounts for percepts
> elicited by retinal implants. *Scientific Reports* 9(1):9199, doi:[10.1038/s41598-019-45416-4](https://doi.org/10.1038/s41598-019-45416-4).

This code is based on [pulse2percept](https://github.com/uwescience/pulse2percept),
a Python-based simulation framework for bionic vision
[(Beyeler et al. 2017)](https://doi.org/10.25080/shinma-7f4c6e7-00c).

Data is available on the [Open Science Framework](https://osf.io/dw9nz/).
You can either download and extract the data yourself
or have the scripts under "figures/" do it for you.

## Installation

Make sure you are running Python 3!

Before you get started, make sure you have NumPy and Cython installed:

```bash
$ pip3 install numpy==1.11
$ pip3 install cython==0.27
```

Then install all packages listed in `requirements.txt`:

```bash
$ pip3 install -r requirements.txt
```

These packages all have their recommended version numbers - these are the tested versions that I used to run the code for the paper. It's likely that other versions (e.g., of NumPy) might work just as well. The only problem I encountered was that phosphenes aren't centered correctly with `scikit-image` 0.14.

After that, you are ready to install the main package, `argus_shapes`:

```bash
$ pip3 install -e .
```

If you want to make sure that everything works as expect, you can run the test suite:

```bash
$ pip3 install pytest
$ pytest argus_shapes
```

## Figures

The code to reproduce figures in the paper can be found in the "figures/" folder:

-   [fig2-phosphene-shape.ipynb](https://github.com/VisCog/ArgusShapes/blob/master/figures/fig2-phosphene-shape.ipynb): Phosphene drawings vary across electrodes.

-   [fig3-shape-descriptors.ipynb](https://github.com/VisCog/ArgusShapes/blob/master/figures/fig3-shape-descriptors.ipynb): Shape descriptors used to measure phosphene variability.

-   [fig5-axon-map-orientation.ipynb](https://github.com/VisCog/ArgusShapes/blob/master/figures/fig5-axon-map-orientation.ipynb): Phosphene orientation is aligned with retinal nerve
  fiber bundles.

-   [fig6-model-shapes.ipynb](https://github.com/VisCog/ArgusShapes/blob/master/figures/fig6-model-shapes.ipynb): Cross-validated phosphene shape predictions.

-   [fig6-inset-models.ipynb](https://github.com/VisCog/ArgusShapes/blob/master/figures/fig6-inset-models.ipynb): Scoreboard and axon map model schematics.

-   [fig7-model-scatter.ipynb](https://github.com/VisCog/ArgusShapes/blob/master/figures/fig7-model-scatter.ipynb): Cross-validated shape descriptor predictions.

-   [fig8-model-phosphenes.ipynb](https://github.com/VisCog/ArgusShapes/blob/master/fig8-model-phosphenes.ipynb): Predicted phosphene shape as a function of electrode-retina distance.

These notebooks assume that the data live in a directory `${DATA_ROOT}/argus_shapes`,
where `DATA_ROOT` is an environment variable.
On Unix, make sure to add `DATA_ROOT` to your `~/.bashrc`:

```bash
$ echo 'export DATA_ROOT=/home/username/data' >> ~/.bashrc
$ source ~/.bashrc
```

You can either download and extract the data from OSF yourself, or have
the notebooks automatically do it for you. In the above case,
the data will end up in "/home/username/data/argus_shapes".

## Loading your own data

In order to load your own data, you will need two .csv files:

`subjects.csv` should have the following columns:

-   `subject_id`: subject ID, has to be the same as in `drawings.csv` (e.g., S1)
-   `implant_type`: currently supported are either 'ArgusI' or 'ArgusII'
-   `implant_eye`: either 'LE' for left eye or 'RE' for right eye
-   `implant_x` / `implant_y`: (x,y)-coordinates of array center in microns, assuming the fovea is at (0, 0)
-   `implant_rot`: array rotation in radians (positive: counter-clockwise rotation)
-   `loc_od_x` / `loc_od_y`: (x,y)-coordinates of optic disc center of this subject in microns
-   `xmin` / `xmax`: x-extent (horizontal) of touch screen in degrees of visual angle (e.g., xmin=-36, xmax=36)
-   `ymin` / `ymax`: y-extent (vertical) of touch screen in degrees of visual angle (e.g., ymin=-24, ymax=24)

`drawings.csv` should have the following columns:

-   `subject_id`: subject ID, has to be the same as in `subjects.csv` (e.g., S1)
-   `stim_class`: currently supported is 'SingleElectrode'
-   `PTS_ELECTRODE`: electrode name
-   `PTS_FILE`: path to image file
-   `PTS_AMP`: applied current amplitude in micro-Amps
-   `PTS_FREQ`: applied pulse frequency in Hz
-   `date`: date of data collection

Then the data can be loaded as Pandas DataFrames using the following Python recipe:

```python
>>> import argus_shapes as shapes
>>> df_subjects = shapes.load_subjects('subjects.csv')
>>> df_drawings = shapes.load_data('drawings.csv')
```

## Submodules

-   `argus_shapes`: Main module.

    -   `fetch_data`: Download data from the web.

    -   `load_data`: Load shape data from a local .csv file.

    -   `load_subjects`: Load subject data from a local .csv file.

    -   `extract_best_pickle_files`: Return a list of pickle files with lowest train
        scores.

-   `models`: Code to run various versions of the scoreboard and axon map models.

    -   `ScoreboardModel`: Scoreboard model with shape descriptor loss

    -   `AxonMapModel`: Axon map model with shape descriptor loss

-   `model_selection`:

    -   `FunctionMinimizer`: Perform function minimization.

    -   `GridSearchOptimizer`: Perform grid search optimization.

    -   `ParticleSwarmOptimizer`: Perform particle swarm optimization.

    -   `crossval_predict`: Predict data using k-fold cross-validation.

    -   `crossval_score`: Score a model using k-fold cross-validation.

-   `imgproc`: Various image processing routines.

    -   `get_thresholded_image`: Apply a threshold to a grayscale image.

    -   `get_region_props`: Calculate region properties of a binary image
        (area, center of mass, orientation, etc.)

    -   `calc_shape_descriptors`: Calculate area, orientation, elongation
        of a phosphene.

    -   `center_phosphene`: Center a phosphene in an image.

    -   `scale_phosphene`: Apply a scaling factor to a phosphene.

    -   `rotate_phosphene`: Rotate a phosphene by a certain angle.

    -   `dice_coeff`: Calculate the dice coefficient between two phosphenes.

-   `utils`: Various utility functions.

    -   `ret2dva`: Convert retinal to visual field coordinates.

    -   `dva2ret`: Convert visual field to retinal coordinates.

    -   `cart2pol`: Convert from Cartesian to polar coordinates.

    -   `pol2cart`: Convert from polar to Cartesian coordinates.

    -   `angle_diff`: Calculate the signed difference between two angles.

-   `viz`: Some visualization functions.

    -   `scatter_correlation`: Scatter plots some data points and fits a
        regression curve.

    -   `plot_phosphenes_on_array`: Plots mean phosphenes on a schematic of
        the implant.

    -   `plot_fundus`: Plots the implant on top of a simulated axon map.

## Miscellaneous

-   `minimal-example.ipynb`: A minimal usage example.

-   `run_fit.sh`: Bash script to fit the models to all subject data.

-   `run_crossval.sh`: Bas script to run leave-one-electrode-out cross-validation.

-   `crossval_swarm.py`: Python file running the model fitting / cross-validation.
