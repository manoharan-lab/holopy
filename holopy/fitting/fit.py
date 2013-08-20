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
Routines for fitting a hologram to an exact solution

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <jfung@physics.harvard.edu>
.. moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>

"""
from __future__ import division

import warnings
import time

from ..core.holopy_object import HoloPyObject
from .errors import MinimizerConvergenceFailed, InvalidMinimizer
from .minimizer import Minimizer, Nmpfit

from copy import copy

class FitResult(HoloPyObject):
    """
    The results of a fit.

    You should not make objects of this class directly, they will be given to
    you by :func:`fit`

    Parameters
    ----------
    parameters : array(float)
        The fitted values for each parameter
    scatterer : :mod:`.scatterer`
        The best fit scatterer
    chisq : float
        The :math:`\chi^2` goodness of fit
    rsq : float
        The :math:`R^2` goodness of fit
    converged : bool
        Did the minimizer converge
    time : float
        Time in seconds the fit took
    minimizer : :class:`.Minimizer`
        Them minimizer used in the fit
    minimization_details : object
        Additional information returned by the minimizer about the minimization
    """
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

    @property
    def alpha(self):
        return self.model.get_alpha(self.parameters)

    def fitted_holo(self, schema):
        return self.model.theory(self.scatterer, schema, self.alpha)

    def summary(self):
        """
        Put just the essential components of a fit result in a dictionary
        """
        d = copy(self.parameters)
        for par in self.summary_misc:
            d[par] = getattr(self, par)
        return d

    @classmethod
    def from_summary(cls, summary, scatterer_cls):
        """
        Build a FitResult from a summary.

        The returned FitResult will be incomplete because summaries do
        not contain all of the minimizer information that full results do

        Parameters
        ----------
        summary : dict
            A dict as from cls.summary containing information about a fit.
        """
        summary = copy(summary)
        misc = {}
        for key in cls.summary_misc[:-1]:
            misc[key] = summary.pop(key, None)
        scatterer = scatterer_cls.from_parameters(summary)
        return cls(scatterer.parameters, scatterer, model=None, minimizer=None,
                   minimization_details=None, **misc)

    summary_misc = ['rsq', 'chisq', 'time', 'converged', 'niter']

    def niter(self):
        # TODO: have this correctly pull number of iterations from
        # non-nmpfit minimizers.
        return self.minimization_details.niter




def chisq(fit, data):
    return float((((fit-data))**2).sum() / fit.size)

def rsq(fit, data):
    return float(1 - ((data - fit)**2).sum()/((data - data.mean())**2).sum())


def fit(model, data, minimizer=Nmpfit):
    """
    fit a model to some data

    Parameters
    ----------
    model : :class:`~holopy.fitting.model.Model` object
        A model describing the scattering system which leads to your data and
        the parameters to vary to fit it to the data
    data : :class:`~holopy.core.marray.Marray` object
        The data to fit
    minimizer : (optional) :class:`~holopy.fitting.minimizer.Minimizer`
        The minimizer to use to do the fit

    Returns
    -------
    result : :class:`FitResult`
        an object containing the best fit paramters and information about the fit
    """
    time_start = time.time()

    if not isinstance(minimizer, Minimizer):
        if issubclass(minimizer, Minimizer):
            minimizer = minimizer()
        else:
            raise InvalidMinimizer("Object supplied as a minimizer could not be"
                                   "interpreted as a minimizer")

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
    fitted_holo = model.theory(fitted_scatterer, model.get_schema(data),
                               scaling = model.get_alpha(fitted_pars))
    if fitted_holo.use_random_fraction is not None and fitted_holo.use_random_fraction != 1:
        print("Using subset to compute statistics")
        sel = fitted_holo.selection
        fitted_holo = fitted_holo[sel]
        data = data[sel]

    time_stop = time.time()

    return FitResult(fitted_pars, fitted_scatterer, chisq(fitted_holo, data),
                     rsq(fitted_holo, data), converged, time_stop - time_start,
                     model, minimizer, minimizer_info)
