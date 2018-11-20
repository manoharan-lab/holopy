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


class LevenbergMarquardtStrategy(HoloPyObject):
    def __init__(self, ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=None,
                 random_subset=None):
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.max_nfev = max_nfev
        self.random_subset = random_subset
        self._optimizer_kwargs = {
            'ftol': self.ftol,
            'xtol': self.xtol,
            'gtol': self.gtol,
            'max_nfev': self.max_nfev,
            'jac': '2-point',
            'method': 'lm',  # FIXME
            'loss': 'linear',
            }
        pass

    def pars_from_minimizer(self, parameters, values):
        assert len(parameters) == len(values)
        return {par.name: par.unscale(value) for par, value in zip(parameters, values)}

    def fit(self, model, data):
        """
        fit a model to some data

        Parameters
        ----------
        model : :class:`~holopy.fitting.model.Model` object
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

        if self.random_subset is None:
            data = flat(data)
        else:
            data = make_subset_data(data, self.random_subset)

        guess_prior = model.lnprior({par.name:par.guess for par in parameters})
        def residual(par_vals):
            pars, noise = model._prep_pars(par_vals, data)
            residuals = model._residuals(par_vals, data, noise)
            prior = np.sqrt(guess_prior - model.lnprior(par_vals))
            np.append(residuals, prior)
            return residuals

        # The only work here
        fitted_pars, minimizer_info = self.minimize(parameters, residual)
        # ~~~
        if not minimizer_info.success:
            warnings.warn("Minimizer Convergence Failed, your results \
                                may not be correct.")

        perrors = self._estimate_error_from_fit(minimizer_info, data.size)
        assert len(parameters) == perrors.size
        intervals = [UncertainValue(fitted_pars[par.name], err, name=par.name)
                     for err, par in zip(perrors, parameters)]
        # timing decorator...
        d_time = time.time() - time_start
        return FitResult(data, model, self, intervals, d_time, minimizer_info)

    def minimize(self, parameters, obj_func):
        initial_parameter_guess = [par.scale(par.guess) for par in parameters]

        def resid_wrapper(p):
            return obj_func(self.pars_from_minimizer(parameters, p))

        # now fit it
        fitresult = least_squares(
            resid_wrapper, initial_parameter_guess, **self._optimizer_kwargs)
        result_pars = self.pars_from_minimizer(parameters, fitresult.x)
        return result_pars, fitresult

    def _estimate_error_from_fit(self, minimizer_info, n_data_points):
        # Estimates 1-sigma gaussian errors
        jacobian = minimizer_info.jac
        jtj = np.dot(jacobian.T, jacobian)
        jtjinv = np.linalg.inv(jtj)
        noise_estimate = np.sqrt(minimizer_info.cost / n_data_points)
        parameter_uncertainties = np.diag(jtjinv) * noise_estimate
        return parameter_uncertainties
