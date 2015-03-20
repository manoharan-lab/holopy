# Copyright 2011-2013, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang
#
# This file is part of HoloPy.
#
# HoloPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HoloPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HoloPy.  If not, see <http://www.gnu.org/licenses/>.
"""
Routine for fitting a time series of holograms to an exact solution

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>

"""
from __future__ import division

from tempfile import NamedTemporaryFile
import pandas as pd
from holopy.fitting import fit as fit_single
from holopy.fitting import FitResult
import os
import numpy as np
from holopy.core.process import normalize
from holopy.core import subimage
import yaml

#default preprocessing function
def div_normalize(holo, bg, df, model):
    if df is None:
        df = np.zeros_like(holo)
    if bg is not None:
        imagetofit = normalize((holo-df)/(bg-df))
    else:
        imagetofit = normalize(holo)
    return imagetofit

# Alternate preprocessing function which fits to a smaller subimage centered
# around the scatterer
def scatterer_centered_subimage(size, recenter_at_edge=False):
    def preprocess(holo, bg, df, model):
        center = np.array(model.scatterer.guess.center[:2])/holo.spacing
        try:
            return normalize(subimage(holo/bg, center, size))
        except IndexError:
            if not recenter_at_edge:
                raise
            new_center = np.array(model.scatterer.guess.center[:2])
            new_center -= np.clip(new_center-np.array(size)/2, -np.inf, 0)
            new_center += np.clip(holo.shape[:2]-(new_center + np.array(size)/2), -np.inf, 0)
            return normalize(subimage(holo/bg, new_center, size))

    return preprocess

#default updating function
def update_all(model, fitted_result):
    for p in model.parameters:
        name = p.name
        p.guess = fitted_result.parameters[name]
    return model

class SeriesResult(object):
    dataset = 'fit_result'
    def __init__(self, file=None, initial_model=None):
        self.file = file
        if file is None:
            file = NamedTemporaryFile
            self.file = file.name
        self.rows = 0

        self.store = pd.HDFStore(file)
        if initial_model is None:
            # See if we can get a initial_model from the data
            try:
                initial_model = self.storer.get_storer(dataset).attrs['initial_model']
            except:
                # No worries if we can't, just don't have one
                pass
        self.initial_model = initial_model


    # __enter__ and __exit__ let SeriesResult behave properly in a
    # context manager and make sure the HDFStore gets properly flushed
    # and closed
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.store.flush()
        self.store.close()

    def append(self, result, flush=False):
        self.store.append(dataset, pd.DataFrame([result.summary()], [self.rows]))
        self.rows += 1
        # the dataset does not exist until we have written at least
        # one result, so we have to add the initial_model to its
        # metadata afterwards. The if ensures that we only add it
        # once. It would be nice if there was a cleaner way to do
        # this.
        if not hasattr(self.store.get_storer(dataset).attrs, 'initial_model'):
            self.store.get_storer(dataset).attrs.initial_model = yaml.dump(self.initial_model)
        if flush:
            self.store.flush()

    def __getitem__(self, v):
        return self.store.select(dataset).iloc[v]

    def save(self, name):
        self.store.flush()
        if self.file != name:
            os.renames(self.file, name)

#def load_series_result(filename):



def fit(model, data, bg=None, df=None, output_file=None,
        preprocess_func=div_normalize, update_func=update_all, restart=False,
        **kwargs):
    """
    fit a model to a timeseries of data.

    Parameters
    ----------
    model : :class:`~holopy.fitting.model.Model` object
        A model describing the scattering system which leads to your data and
        the parameters to vary to fit it to the data
    data : :class:`~holopy.core.marray.ImageSequence` object
        The data to fit
    bg : :class:`~holopy.core.marray.Image` object
        A background image to apply to each frame before fitting
    df : :class:`~holopy.core.marray.Image` object
        A darkfield image to apply to each frame before fitting
    preprocess_func : function(frame, bg, df, model)
        A function to preprocess the data before fitting each frame. It will get
        as aguments the raw frame, the bg, df, and the model. The default is
        to do basic background/darkfield division and then normalize
    update_func : function(model, result)
        A function to produce the initial guess for the next frame. It will get
        as arguments the model and result from the last frame.
    **kwargs : various
        Keyword arguments to pass through to the minimizer
    Returns
    -------
    result : :class:`SeriesResult`
        an object containing the best fit parameters and information about the fit

    """
    if (output_file is not None and
        os.path.splitext(output_file)[1] not in ['.h5', '.hdf5']):
        output_file += '.h5'
    with SeriesResult(output_file) as results:
        if isinstance(bg, basestring):
            bg = load(bg, spacing=data_spacing, optics=data_optis)
        if isinstance(df, basestring):
            df = load(df, spacing=data_spacing, optics=data_optics)

        for i, frame in enumerate(data):
            imagetofit = preprocess_func(frame, bg, df, model)
            result = fit_single(model, imagetofit, **kwargs)
            results.append(fit_single(model, imagetofit, **kwargs), True)
            print("Fit frame {}, rsq={}".format(i, result.rsq))
            model = update_func(model, result)

        return results
