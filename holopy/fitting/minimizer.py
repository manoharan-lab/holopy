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
Interfaces to minimizers.  Classes here provide a common interface to a variety
of third party minimizers.

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""

from __future__ import division

import numpy as np
from ..core.holopy_object import HoloPyObject
from ..core.helpers import OrderedDict
from .errors import ParameterSpecificationError, MinimizerConvergenceFailed
from ..scattering.errors import ScattererDefinitionError
from .third_party import nmpfit


class Minimizer(HoloPyObject):
    """
    Common interface to all minimizers holopy supports
    """
    def minimize(self, parameters, cost_func):
        """
        Find the best solution to an optimization problem

        Parameters
        ----------
        parameters : list of :class:`.Parameter` objects
            Parameters to vary in the model
        cost_func : function
            A function taking parameters as arguments that returns the residual
            for the minimization problem
        """
        raise NotImplementedError() # pragma: nocover

    # if minimizers do any parameter rescaling, they are responsible for putting
    # the parameters back before handing them off to the model.
    def pars_from_minimizer(self, parameters, values):
        pars = OrderedDict()
        for par, value in zip(parameters, values):
            pars[par.name] = par.unscale(value)

        return pars


class Nmpfit(Minimizer):
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

    Notes
    -----

    See nmpfit documentation for further details. Not all functionalities of
    nmpfit are implemented here: in particular, we do not allow analytical
    derivatives of the residual function, which is impractical and/or
    impossible to calculate for holograms. If you want to weight the residuals,
    you need to supply a custom residual function.

    """
    def __init__(self, quiet = False, ftol = 1e-10, xtol = 1e-10, gtol = 1e-10,
                 damp = 0, maxiter = 100):
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.damp = 0
        self.maxiter = maxiter
        self.quiet = quiet

    def minimize(self, parameters, cost_func, debug = False):
        # marshall the paramters into a dict of the form nmpfit wants
        nmp_pars = []
        for par in parameters:
            d = {'parname': par.name}
            if par.limit is not None:
                d['limited'] = [par.scale(l) is not None for l in par.limit]
                d['limits'] = par.scale(np.array(par.limit))
            else:
                d['limited'] = [False, False]
            if par.guess is not None:
                d['value'] = par.scale(par.guess)
            else:
                raise ParameterSpecificationError("nmpfit requires an "
                                                    "initial guess for all "
                                                    "parameters")
            # Check for other allowed parinfo keys here: see nmpfit docs
            allowed_keys = ['step', 'mpside', 'mpmaxstep']
            for key, value in par.kwargs.iteritems():
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

        def resid_wrapper(p, fjac=None):
            status = 0
            return [status, cost_func(self.pars_from_minimizer(parameters, p))]

        # now fit it
        fitresult = nmpfit.mpfit(resid_wrapper, parinfo=nmp_pars, ftol = self.ftol,
                                 xtol = self.xtol, gtol = self.gtol, damp = self.damp,
                                 maxiter = self.maxiter, quiet = self.quiet)

        result_pars = self.pars_from_minimizer(parameters, fitresult.params)

        if fitresult.status == 5:
            raise MinimizerConvergenceFailed(result_pars, fitresult)

        if debug == True:
            return result_pars, fitresult, nmp_pars
        else:
            return result_pars, fitresult

    minimize.__doc__ = Minimizer.minimize.__doc__

class OpenOpt(Minimizer):
    def __init__(self, algorithm = 'ralg', quiet = False, plot = False):
        self.algorithm = algorithm
        self.quiet = quiet
        self.plot = plot
        import openopt
        openopt_nllsq = ['scipy_leastsq']
        # scipy_leastsq cannot handle bounds
        openopt_nlp = ['ralg', 'scipy_lbfgsb', 'scipy_slsqp']
        openopt_global = ['galileo']
        if algorithm in openopt_nlp:
            self.problem_type = openopt.NLP
        elif algorithm in openopt_global:
            self.problem_type = openopt.GLP
        elif algorithm in openopt_nllsq:
            self.problem_type = openopt.NLLSP

    def minimize(self, parameters, cost_func):
        lb = []
        ub = []
        for p in parameters:
            if p.limit is not None:
                lb.append(p.scale(p.limit[0]))
                ub.append(p.scale(p.limit[1]))
            else:
                lb.append(-np.inf)
                ub.append(np.inf)
        guess = [p.scale(p.guess) for p in parameters]
        def resid_wrapper(p):
            try:
                resid = cost_func(self.pars_from_minimizer(parameters, p))
                return np.dot(resid, resid)
            except ScattererDefinitionError:
                return np.inf
        problem = self.problem_type(resid_wrapper, guess, lb=lb, ub=ub, iprint = self.quiet,
                                    plot = self.plot)
        r = problem.solve(self.algorithm)
        result_pars = self.pars_from_minimizer(parameters, r.xf)
        return result_pars, r

    minimize.__doc__ = Minimizer.minimize.__doc__
