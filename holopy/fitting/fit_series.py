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
from __future__ import division

import warnings

from ..core.holopy_object import HoloPyObject
from ..core.process import normalize
from ..core.io import load, save
from ..core.metadata import Optics
from . import fit

#default preprocessing function
def div_normalize(holo, bg):
    if bg is not None:
        imagetofit = normalize(holo/bg)
    else:
        imagetofit = normalize(holo)   
    return imagetofit
    
#default updating function
def update_all(model, fitted_result):
    for p in model.parameters:
        name = p.name
        p.guess = fitted_result.parameters[name]
    return model
        
def fitseries(model, infilenames, opticsinfo, px_size, bg=None, 
    outfilenames=None, preprocess_func=div_normalize, 
    update_func=update_all):
    """
    fit a model to each frame of data in a time series

    Parameters
    ----------
    model : :class:`~holopy.fitting.model.Model` object
        A model describing the scattering system which leads 
        to your data and the parameters to vary to fit it 
        to the data
    infilenames : :list:`
        Full paths to the data to fit
    opticsinfo : :object:'
        A HoloPy optics object
    px_size : :float:' or :np.array:'
        The size of a pixel-- single float implies square pixels
    bg : :image:' or :path:'
        Optional background image to be used for cleaning up 
        the raw data images
    outfilenames : :list:`
        Full paths to save output for each image, if not 
        included, nothing saved
    preprocess_func : :
        Handles pre-processing images before fitting the model 
        to them
    update_func : : 
        Updates the model (typically just the paramter guess) 
        for the next frame

    Returns
    -------
    allresults : :list:`
        List of all the result objects (one per frame)
    """    
    
    allresults = []
    
    if isinstance(bg, basestring):
        bg = load(bg, px_size=px_size, optics=opticsinfo)

    #to allow running without saving output   
    if outfilenames is None:
        outfilenames = ['']*len(infilenames)
        
    for frame, outpath in zip(infilenames, outfilenames):
    
        holo = load(frame, spacing=px_size, optics=opticsinfo)
        imagetofit = preprocess_func(holo, bg)
        
        result = fit(model, imagetofit)
        allresults.append(result)
        if outpath!='':
            save(outpath, result)
        
        model = update_all(model, result)

    return allresults
