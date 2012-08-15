# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca W. Perry,
# Jerome Fung, and Ryan McGorty
#
# This file is part of Holopy.
#
# Holopy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Holopy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Holopy.  If not, see <http://www.gnu.org/licenses/>.
"""
Routines for fitting a hologram to an exact solution

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <jfung@physics.harvard.edu>
.. moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>

"""
from __future__ import division

import warnings
import time

from ..core.holopy_object import HolopyObject
from .errors import MinimizerConvergenceFailed
from .minimizer import Minimizer, Nmpfit

class FitResult(HolopyObject):
    def __init__(self, parameters, scatterer, chisq, rsq, converged, time, model,
                 minimizer, minimization_details):
        self.parameters = parameters
        self.scatterer = scatterer
        self.chisq = chisq
        self.rsq = rsq
        self.converged = converged
        self.time = time
        self.model = model
        self.minimizer = minimizer
        self.minimization_details = minimization_details



def fit(model, data, minimizer=Nmpfit):
    time_start = time.time()

    if issubclass(minimizer, Minimizer):
        minimizer = minimizer()
    
    try:
        fitted_pars, minimizer_info = minimizer.minimize(model.parameters,
                                                         model.cost_func(data))
        converged = True
    except MinimizerConvergenceFailed as cf:
        warnings.warn("Minimizer Convergence Failed, your results may not be "
                      "correct")
        # we still return the data even if the minimizer fails to converge
        # because often the data is of some value, and may in fact be what the
        # user wants if they have set low iteration limits for a "rough fit"
        fitted_pars, minimizer_info  = cf.result, cf.details
        converged = False

    # compute goodness of fit parameters
    fitted_scatterer = model.scatterer.make_from(fitted_pars)
    fitted_holo = model.theory(fitted_scatterer, model.get_target(data),
                               scaling = model.get_alpha(fitted_pars)) 
    chisq = float((((fitted_holo-data))**2).sum() / fitted_holo.size)
    rsq = float(1 - ((data - fitted_holo)**2).sum()/((data - data.mean())**2).sum())

    time_stop = time.time()

    return FitResult(fitted_pars, fitted_scatterer, chisq, rsq, converged,
                     time_stop - time_start, model, minimizer, minimizer_info)

