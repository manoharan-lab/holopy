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
from holopy.inference.prior import Uniform
from holopy.scattering.errors import ParameterSpecificationError
from holopy.fitting import FitResult


class NmpfitStrategy(HoloPyObject):
    """
    Levenberg-Marquardt minimizer, from Numpy/Python translation of Craig
    Markwardt's mpfit.pro.

    Parameters
    ----------
    quiet: Boolean
        If True, suppress output on minimizer convergence.
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
    random_subset : float
        Fit only a randomly selected fraction of the data points in data

    Notes
    -----

    See nmpfit documentation for further details. Not all functionalities of
    nmpfit are implemented here: in particular, we do not allow analytical
    derivatives of the residual function, which is impractical and/or
    impossible to calculate for holograms. If you want to weight the residuals,
    you need to supply a custom residual function.

    """
    def __init__(self, quiet = False, ftol = 1e-10, xtol = 1e-10, gtol = 1e-10,
                 damp = 0, maxiter = 100, random_subset=None):
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.damp = 0
        self.maxiter = maxiter
        self.quiet = quiet
        self.random_subset = random_subset

    def pars_from_minimizer(self, parameters, values):
        assert len(parameters) == len(values)
        return {par.name: par.unscale(value) for par, value in zip(parameters, values)}

    def fit(self, model, data):
        """
        fit a model to some data

        Parameters
        ----------
        model : :class:`~holopy.fitting.model.Model` object
            A model describing the scattering system which leads to your data and
            the parameters to vary to fit it to the data
        data : xarray.DataArray
            The data to fit

        Returns
        -------
        result : :class:`FitResult`
            an object containing the best fit parameters and information about the fit
        """
        time_start = time.time()
        parameters = model._parameters

        if self.random_subset is None:
            data = flat(data)
        else:
            data = make_subset_data(data, self.random_subset)

      # marshall the parameters into a dict of the form nmpfit wants
        nmp_pars = []
        for par  in parameters:
            d = {'parname': par.name, 'value':par.scale(par.guess)}
            d['limited'] = [False, False]
            if hasattr(par, 'lower_bound') and par.lower_bound > -np.inf:
                d['limited'][0] = True
                d['limits'] = [par.scale(par.lower_bound)]
            if hasattr(par, 'upper_bound') and par.upper_bound < np.inf:
                d['limited'][1] = True
                if 'limits' not in d.keys():
                    d['limits']=[]
                d['limits'].append(par.scale(par.upper_bound))

            # Check for other allowed parinfo keys here: see nmpfit docs
            allowed_keys = ['step', 'mpside', 'mpmaxstep']
            if hasattr(par, 'kwargs'):
                for key, value in par.kwargs.items():
                    if key in allowed_keys:
                        if key == 'mpside':
                            d[key] = value
                        else:
                            d[key] = par.scale(value)
                    else:
                        raise ParameterSpecificationError("Parameter " + par.name +
                                                      " contains kwargs that" +
                                                      " are not supported by" +
                                                      " nmpfit")
            nmp_pars.append(d)

        ignore_prior = np.all([isinstance(par, Uniform) for par in parameters])
        ignore_prior = False
        if not ignore_prior:
            guess_prior = 10 * model.lnprior([par.guess for par in parameters])

        def resid_wrapper(p, fjac=None):
            status = 0
            scaled_pars = self.pars_from_minimizer(parameters, p)
            scaled_pars, noise = model._prep_pars(scaled_pars, data)
            residuals = model._residuals(scaled_pars, data, noise)
            if not ignore_prior:
                prior = model.lnprior(scaled_pars)
                if prior > 0:
                    prior = np.sqrt(prior)
                elif prior < guess_prior:
                    prior = 0
                else:
                    prior = np.sqrt(prior - 10 * guess_prior)
                np.append(residuals, prior)
            return [status, residuals]

        # now fit it
        fitresult = nmpfit.mpfit(resid_wrapper, parinfo=nmp_pars, ftol = self.ftol,
                                 xtol = self.xtol, gtol = self.gtol, damp = self.damp,
                                 maxiter = self.maxiter, quiet = self.quiet)

        result_pars = self.pars_from_minimizer(parameters, fitresult.params)

        if fitresult.status == 5:
            converged = False
            warnings.warn("Minimizer Convergence Failed, your results \
                                may not be correct.")
        else:
            converged = True

        fitted_scatterer = model.scatterer.from_parameters(result_pars)
        time_stop = time.time()
        fitted = model._calc(result_pars, data)
        return FitResult(result_pars, fitted_scatterer, chisq(fitted, data),
                     rsq(fitted, data), converged, time_stop - time_start,
                     model, self, fitresult)
