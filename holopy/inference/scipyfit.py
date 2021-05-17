"""
.. moduleauthor:: Brian D Leahy <bleahy@seas.harvard.edu>
"""

import time
import warnings

import numpy as np
from scipy.optimize import least_squares

from holopy.core.holopy_object import HoloPyObject
from holopy.core.metadata import flat, make_subset_data
from holopy.scattering.errors import  MissingParameter
from holopy.inference.result import FitResult, UncertainValue


class LeastSquaresScipyStrategy(HoloPyObject):
    def __init__(self, ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=None,
                 npixels=None):
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.max_nfev = max_nfev
        self.npixels = npixels
        self._optimizer_kwargs = {
            'ftol': self.ftol,
            'xtol': self.xtol,
            'gtol': self.gtol,
            'max_nfev': self.max_nfev,
            'jac': '2-point',
            'method': 'lm',
            'loss': 'linear',
            }

    def unscale_pars_from_minimizer(self, parameters, values):
        assert len(parameters) == len(values)
        return [val.unscale(value)
                for val, value in zip(parameters, values)]

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
            Contains the best fit parameters and information about the fit
        """
        # timing decorator...
        time_start = time.time()

        parameters = model._parameters
        if len(parameters) == 0:
            raise MissingParameter('at least one parameter to fit')

        if self.npixels is None:
            data = flat(data)
        else:
            data = make_subset_data(data, pixels=self.npixels)
        guess_lnprior = model.lnprior(model.initial_guess)

        def residual(rescaled_values):
            unscaled_values = self.unscale_pars_from_minimizer(
                parameters, rescaled_values)
            noise = model._find_noise(unscaled_values, data)
            residuals = model._residuals(unscaled_values, data, noise)
            ln_prior = model._lnprior(unscaled_values) - guess_lnprior
            zscore_prior = np.sqrt(2 * -ln_prior)
            np.append(residuals, zscore_prior)
            return residuals

        # The only work here
        fitted_pars, minimizer_info = self.minimize(parameters, residual)

        if not minimizer_info.success:
            warnings.warn("Minimizer Convergence Failed, your results \
                                may not be correct.")

        unit_errors = self._calculate_unit_noise_errors_from_fit(minimizer_info)
        noise = model._find_noise(fitted_pars, data)
        errors_scaled = noise * unit_errors
        errors = self.unscale_pars_from_minimizer(parameters, errors_scaled)
        intervals = [UncertainValue(par, err, name=name)
                     for par, err, name in
                     zip(fitted_pars, errors, model._parameter_names)]

        # timing decorator...
        d_time = time.time() - time_start
        kwargs = {'intervals': intervals, 'minimizer_info': minimizer_info}
        return FitResult(data, model, self, d_time, kwargs)

    def minimize(self, parameters, residuals_function):
        initial_parameter_guess = [par.scale(par.guess) for par in parameters]
        fitresult = least_squares(residuals_function, initial_parameter_guess,
                                  **self._optimizer_kwargs)
        result_pars = self.unscale_pars_from_minimizer(parameters, fitresult.x)
        return result_pars, fitresult

    @classmethod
    def _calculate_unit_noise_errors_from_fit(cls, minimizer_info):
        jacobian = minimizer_info.jac
        jtj = np.dot(jacobian.T, jacobian)
        jtjinv = np.linalg.inv(jtj)
        return np.sqrt(np.diag(jtjinv))

