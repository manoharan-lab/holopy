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
Interfaces to minimizers.  Classes here provide a common interface to a variety
of third party minimizers.

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""

import time
import warnings

import numpy as np

from holopy.core.holopy_object import HoloPyObject
from holopy.core.metadata import flat, make_subset_data
from holopy.core.math import chisq, rsq
from holopy.inference.third_party import nmpfit
from holopy.scattering.errors import (
    ParameterSpecificationError, MissingParameter)
from holopy.inference.result import FitResult, UncertainValue


# FIXME the errors from NmpfitStrategy seem to be incorrect! They do
# not agree with those from scipy least squares


class NmpfitStrategy(HoloPyObject):
    """
    Levenberg-Marquardt minimizer, from Numpy/Python translation of Craig
    Markwardt's mpfit.pro.

    Parameters
    ----------
    npixels: None
        Fit only a randomly selected fraction of the data points in data
    quiet: Boolean
        If False, print output on minimizer convergence. Default is True
    ftol: float
        Convergence criterion for minimizer: converges if actual and predicted
        relative reductions in chi squared <= ftol
    xtol: float
        Convergence criterion for minimizer: converges if relative error between
        two Levenberg-Marquardt iterations is <= xtol
    gtol: float
        Convergence criterion for minimizer: converges if absolute value of
        cosine of angle between vector of cost function evaluated at current
        solution for minimized parameters and any column of the Jacobian is
        <= gtol
    damp: float
        If nonzero, residuals larger than damp will be replaced by tanh. See
        nmpfit documentation.
    maxiter: int
        Maximum number of Levenberg-Marquardt iterations to be performed.

    Notes
    -----

    See nmpfit documentation for further details. Not all functionalities of
    nmpfit are implemented here: in particular, we do not allow analytical
    derivatives of the residual function, which is impractical and/or
    impossible to calculate for holograms. If you want to weight the residuals,
    you need to supply a custom residual function.

    """
    def __init__(self, npixels=None, quiet=True, ftol=1e-10, xtol=1e-10,
                 gtol=1e-10, damp=0, maxiter=100, seed=None):
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.damp = 0
        self.maxiter = maxiter
        self.quiet = quiet
        self.npixels = npixels
        self.seed = seed

    def unscale_pars_from_minimizer(self, values):
        assert len(values) == len(self._parameters)
        return [val.unscale(value) for val, value in zip(self._parameters,
                                                         values)]

    def fit(self, model, data):
        """
        fit a model to some data

        Parameters
        ----------
        model : :class:`~holopy.inference.model.Model` object
            A model describing the scattering system which leads to your
            data and the parameters to vary to fit it to the data
        data : xarray.DataArray
            The data to fit

        Returns
        -------
        result : :class:`FitResult`
            an object containing the best fit parameters and information
            about the fit
        """
        time_start = time.time()
        self.initialize_fit(model, data)
        fitted_pars, self._minimizer_info = self.minimize(self._parameters,
                                                          self.calc_residuals)
        intervals = self.get_errors_from_minimizer(fitted_pars)
        d_time = time.time() - time_start
        self.cleanup_from_fit()
        return FitResult(data, model, self, d_time,
                         {'intervals': intervals,
                          'mpfit_details': self._minimizer_info})

    def initialize_fit(self, model, data):
        self._model = model
        self._parameters = model._parameters
        if len(self._parameters) == 0:
            raise MissingParameter('at least one parameter to fit')
        if self.npixels is not None:
            data = make_subset_data(data, pixels=self.npixels, seed=self.seed)
        self._data = data
        self._guess_lnpriors = np.array([par.lnprob(par.guess)
                                        for par in self._parameters])

    def calc_residuals(self, par_vals):
        noise = self._model._find_noise(par_vals, self._data)
        data_residuals = self._model._residuals(par_vals, self._data, noise)
        data_residuals = data_residuals.flatten()
        current_lnpriors = np.array([par.lnprob(val) for par, val in
                                     zip(self._parameters, par_vals)])
        prior_residuals = np.sqrt(self._guess_lnpriors - current_lnpriors)
        residuals = np.append(data_residuals, prior_residuals)
        return residuals

    def get_errors_from_minimizer(self, fitted_pars):
        if self._minimizer_info.status == 5:
            setattr(self._minimizer_info, 'converged', False)
            warnings.warn("Minimizer Convergence Failed, your results \
                                may not be correct.")
        else:
            setattr(self._minimizer_info, 'converged', True)
        errors_rescaled = self._minimizer_info.perror
        if errors_rescaled is None:
            errors_rescaled = [0] * len(self._parameters)
        errors = self.unscale_pars_from_minimizer(errors_rescaled)
        intervals = [UncertainValue(par, err, name=name) for par, err, name in
                     zip(fitted_pars, errors, self._model._parameter_names)]
        return intervals

    def cleanup_from_fit(self):
        del self._model
        del self._parameters
        del self._data
        del self._guess_lnpriors

    def minimize(self, parameters, obj_func):
        if not hasattr(self, "_parameters"):
            self._parameters = parameters
        nmp_pars = []
        for par in parameters:
            d = {'parname': par.name, 'value': par.scale(par.guess),
                 'limited': [False, False], 'limits': [np.NaN, np.NaN]}
            if hasattr(par, "lower_bound") and par.lower_bound > -np.inf:
                d['limited'][0] = True
                d['limits'][0] = par.scale(par.lower_bound)
            if hasattr(par, "upper_bound") and par.upper_bound < np.inf:
                d['limited'][1] = True
                d['limits'][1] = par.scale(par.upper_bound)
            nmp_pars.append(d)

        def resid_wrapper(parameters, fjac=None):
            status = 0
            out = obj_func(self.unscale_pars_from_minimizer(parameters))
            return [status, out]

        # now fit it
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fitresult = nmpfit.mpfit(
                resid_wrapper, parinfo=nmp_pars, ftol = self.ftol,
                xtol = self.xtol, gtol = self.gtol, damp = self.damp,
                maxiter = self.maxiter, quiet = self.quiet)

        result_pars = self.unscale_pars_from_minimizer(fitresult.params)

        return result_pars, fitresult
