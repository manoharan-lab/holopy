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
.. moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>

"""


import warnings

import os
import types
import numpy as np
from holopy.core.process import normalize
from holopy.core import subimage, Image
from holopy.core.helpers import mkdir_p
from holopy.core.io import load, save
from holopy.fitting import fit

#default preprocessing function
def div_normalize(holo, bg, df, model):
    if df is None:
        df = np.zeros_like(holo)
    if bg is not None:
        imagetofit = normalize((holo-df)/(bg-df))
    else:
        imagetofit = normalize(holo)
    return imagetofit

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

def _get_first(x):
    if isinstance(x, types.GeneratorType):
        return next(x)
    if isinstance(x, list):
        return x[0]
    else:
        return x

def fit_series(model, data, data_optics=None, data_spacing=None,
               bg=None, df=None, outfilenames=None,
               preprocess_func=div_normalize, update_func=update_all,
               restart=False, **kwargs):
    """
    fit a model to each frame of data in a time series

    Parameters
    ----------
    model : :class:`.Model` object
        A model describing the scattering system which leads
        to your data and the parameters to vary to fit it
        to the data
    data : list(filenames) or list(:class:`.Image`)
        List of Image objects to fit, or full paths of images to load
    data_optics : :class:`.Optics` (optional)
        Optics information (only required if loading image files without
        optical information)
    data_spacing : float or np.array
        Pixel spacing for data. (Only required if loading image files without
        spacing information)
    bg : :class:`.Image` object or path
        Optional background image to be used for cleaning up
        the raw data images
    df : :class:`.Image` object or path
        Optional darkfield image to be used for cleaning up
        the raw data images
    outfilenames : list
        Full paths to save output for each image, if not
        included, nothing saved
    preprocess_func : function
        Handles pre-processing images before fitting the model
        to them
    update_func : function
        Updates the model (typically just the paramter guess)
        for the next frame
    restart : Bool
        Pick up a series fit that was interrupted. For any frame, if outfilename
        already exists load it instead of doing a fit
    kwargs : varies
        additional arguments to pass to fit for each frame

    Returns
    -------
    allresults : :list:`
        List of all the result objects (one per frame)
    """

    allresults = []

    if isinstance(bg, str):
        bg = load(bg, spacing=data_spacing, optics=data_optis)

    #to allow running without saving output
    if outfilenames is None:
        outfilenames = ['']*len(data)

    for frame, outf in zip(data, outfilenames):
        if restart and os.path.exists(outf):
            result = load(outf)
        else:
            if outf != '':
                mkdir_p(os.path.split(outf)[0])
            if not isinstance(frame, Image):
                frame = load(frame, spacing=data_spacing, optics=data_optics)
            imagetofit = preprocess_func(frame, bg, df, model)

            result = fit(model, imagetofit, **kwargs)
            allresults.append(result)
            if outf != '':
                save(outf, result)

        model = update_func(model, result)

    return allresults

def series_guess(model, data, data_optics=None, data_spacing=None,
                 bg=None, df=None, preprocess_func=div_normalize,
                 **kwargs):
    """See the guess that would be used in a series fit

    This function intentionally takes the same arguments as series_fit
    so that you can call series_guess and compare to data[0] before starting a fit

    Parameters
    ----------
    model : :class:`.Model` object
        A model describing the scattering system which leads
        to your data and the parameters to vary to fit it
        to the data
    data : list(filenames) or list(:class:`.Image`)
        List of Image objects to fit, or full paths of images to load
    data_optics : :class:`.Optics` (optional)
        Optics information (only required if loading image files without
        optical information)
    data_spacing : float or np.array
        Pixel spacing for data. (Only required if loading image files without
        spacing information)
    bg : :class:`.Image` object or path
        Optional background image to be used for cleaning up
        the raw data images
    df : :class:`.Image` object or path
        Optional darkfield image to be used for cleaning up
        the raw data images
    preprocess_func : function
        Handles pre-processing images before fitting the model
        to them
    kwargs : varies
        additional arguments to pass to fit for each frame

    Returns
    -------
    guess : marray (like data[0])
        The initial guess that would be used for fitting data[0] in fit_series
    """
    if isinstance(bg, str):
        bg = load(bg, spacing=data_spacing, optics=data_optics)
        if not isinstance(frame, Image):
            frame = load(frame, spacing=data_spacing, optics=data_optics)
        imagetofit = preprocess_func(frame, bg, df, model)

        result = fit(model, imagetofit, **kwargs)
        allresults.append(result)
        if outf != '':
            save(outf, result)

        model = update_func(model, result)

    return allresults

def series_guess(model, data, data_optics=None, data_spacing=None,
                 bg=None, df=None, preprocess_func=div_normalize,
                 **kwargs):
    """See the guess that would be used in a series fit

    This function intentionally takes the same arguments as series_fit
    so that you can call series_guess and compare to data[0] before starting a fit

    Parameters
    ----------
    model : :class:`.Model` object
        A model describing the scattering system which leads
        to your data and the parameters to vary to fit it
        to the data
    data : list(filenames) or list(:class:`.Image`)
        List of Image objects to fit, or full paths of images to load
    data_optics : :class:`.Optics` (optional)
        Optics information (only required if loading image files without
        optical information)
    data_spacing : float or np.array
        Pixel spacing for data. (Only required if loading image files without
        spacing information)
    bg : :class:`.Image` object or path
        Optional background image to be used for cleaning up
        the raw data images
    df : :class:`.Image` object or path
        Optional darkfield image to be used for cleaning up
        the raw data images
    preprocess_func : function
        Handles pre-processing images before fitting the model
        to them
    kwargs : varies
        additional arguments to pass to fit for each frame

    Returns
    -------
    guess : marray (like data[0])
        The initial guess that would be used for fitting data[0] in fit_series
    """
    imagetofit = series_preprocess_data(model, data, data_optics,
                                 data_spacing, bg, df, preprocess_func)

    return model.guess_holo(imagetofit)

def series_preprocess_data(model, data, data_optics=None, data_spacing=None,
                 bg=None, df=None, preprocess_func=div_normalize,
                 **kwargs):
    """See the data that a series fit will see

    This function intentionally takes the same arguments as series_fit
    so that you can call series_guess and compare to data[0] before starting a fit

    Parameters
    ----------
    model : :class:`.Model` object
        A model describing the scattering system which leads
        to your data and the parameters to vary to fit it
        to the data
    data : list(filenames) or list(:class:`.Image`)
        List of Image objects to fit, or full paths of images to load
    data_optics : :class:`.Optics` (optional)
        Optics information (only required if loading image files without
        optical information)
    data_spacing : float or np.array
        Pixel spacing for data. (Only required if loading image files without
        spacing information)
    bg : :class:`.Image` object or path
        Optional background image to be used for cleaning up
        the raw data images
    df : :class:`.Image` object or path
        Optional darkfield image to be used for cleaning up
        the raw data images
    preprocess_func : function
        Handles pre-processing images before fitting the model
        to them
    kwargs : varies
        additional arguments to pass to fit for each frame

    Returns
    -------
    guess : marray (like data[0])
        The initial guess that would be used for fitting data[0] in fit_series
    """
    if isinstance(bg, str):
        bg = load(bg, spacing=data_spacing, optics=data_optics)
    frame = _get_first(data)
    if not isinstance(frame, Image):
        frame = load(frame, data_spacing, data_optics)
    return preprocess_func(frame, bg, df, model)
