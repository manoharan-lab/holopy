# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, Anna Wang, Solomon Barkley
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

from copy import copy, deepcopy

from holopy.core.holopy_object import HoloPyObject


class FitResult(HoloPyObject):
    """
    The results of a fit.

    You should not make objects of this class directly, they will be given to
    you by :func:`fit`

    Parameters
    ----------
    parameters : array(float)
        The fitted values for each parameter
    scatterer : :class:`.Scatterer`
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
        return self.model.get_parameter(self.parameters)

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

    def next_model(self):
        """
        Construct a model to fit the next frame in a time series
        """
        nextmodel = deepcopy(self.model)

        for p in nextmodel._parameters:
            name = p.name
            p.guess = self.parameters[name]
        return nextmodel

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
