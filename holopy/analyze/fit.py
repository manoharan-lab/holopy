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
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

import inspect
import warnings
import time

import numpy as np

import scatterpy
from scatterpy.io import SerializeByConstructor

from holopy.third_party import nmpfit


from holopy.utility.errors import (ParameterSpecficationError,
                                   GuessOutOfBoundsError, MinimizerConvergenceFailed)


class FitResult(SerializeByConstructor):
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

class Parameterization(SerializeByConstructor):
    def __init__(self, pars_to_target, parameters):
        self.parameters = []
        self._fixed_params = {}
        for par in parameters:
            def add_par(p, name = None):
                if name != None:
                    p.name = name
                if not p.fixed:
                    self.parameters.append(p)
                else:
                    self._fixed_params[p.name] = p.guess

            if isinstance(par, ComplexParameter):
                add_par(par.real, par.name+'.real')
                add_par(par.imag, par.name+'.imag')
            elif isinstance(par, Parameter):
                add_par(par)
        self.pars_to_target = pars_to_target

    def make_from(self, parameters):
        # parameters is an ordered dictionary
        for_target = {}
        for arg in inspect.getargspec(self.pars_to_target).args:
            if (arg + '.real') in parameters and (arg + '.imag') in parameters:
                for_target[arg] = (parameters[arg + '.real'] + 1.j *
                                   parameters[arg + '.imag'])
            elif (arg + '.real') in self._fixed_params and \
                    (arg + '.imag') in parameters:
                for_target[arg] = (self._fixed_params[arg + '.real'] + 1.j * 
                                   parameters[arg + '.imag'])
            elif (arg + '.real') in parameters and (arg + '.imag') in \
                    self._fixed_params:
                for_target[arg] = (parameters[arg + '.real'] + 1.j * 
                                   self._fixed_params[arg + '.imag'])
            else:
                for_target[arg] = parameters[arg] 
        return self.pars_to_target(**for_target)

    @property
    def guess(self):
        guess_pars = {}
        for par in self.parameters:
            guess_pars[par.name] = par.guess
        return self.make_from(guess_pars)

    
class ParameterizedTarget(Parameterization):
    def __init__(self, target):
        self.target = target

        # find all the Parameter's in the target
        parameters = []
        for name, par in target.parameters.iteritems():
            def add_par(p, name):
                p.name = name
                if not p.fixed:
                    parameters.append(p)
            if isinstance(par, ComplexParameter):
                add_par(par.real, name+'.real')
                add_par(par.imag, name+'.imag')
            elif isinstance(par, Parameter):
                add_par(par, name)

        self.parameters = parameters

    def make_from(self, parameters):
        target_pars = {}
        
        for name, item in self.target.parameters.iteritems():
            def get_val(par, name):
                if par.fixed:
                    return par.limit
                else:
                    return parameters[name]
                
            if isinstance(item, ComplexParameter):
                target_pars[name] = (get_val(item.real, name+'.real') + 1.0j *
                                     get_val(item.imag, name+'.imag'))
            elif isinstance(item, Parameter):
                target_pars[name] = get_val(item, name)
            else:
                target_pars[name] = item

        return self.target.from_parameters(target_pars)

class Model(SerializeByConstructor):
    """
    Representation of a model to fit to data

    Parameters
    ----------
    parameters: list(:class:`Parameter`)
        The parameters which can be varied in this model.  This list can include
        a scatterer object containing parameters as variable values.  In this
        case that scatterer will be used as a template for make_scatterer, and a
        make_scatterer function does not need to be provided
    theory: :class:`scatterpy.theory.ScatteringTheory`
        The theory that should be used to compute holograms
    make_scatterer: function(par_values) -> :class:`scatterpy.scatterer.AbstractScatterer`
        Function that returns a scatterer given parameters
    selection : float or array of integers (optional)
        Fraction of pixels to compare, or an array with 1's in the locations of
        pixels where you want to calculate the field.  

    Notes
    -----
    Any arbitrary parameterization can be used by simply specifying a
    make_scatterer function which can turn the parameters into a scatterer
    
    """
    def __init__(self, scatterer, theory, metadata=None, selection=None,
                 alpha = None):
        if not isinstance(scatterer, Parameterization):
            scatterer = ParameterizedTarget(scatterer)
        self.scatterer = scatterer

        self.theory = theory

        if metadata is not None and not isinstance(metadata, Parameterization):
            metadata = ParameterizedTarget(metadata)
        self.metadata = metadata

        self.selection = selection
        self._selection = None

        if isinstance(alpha, Parameter) and alpha.name is None:
            alpha.name = 'alpha'
        self.alpha = alpha
        
        self.parameters = []
        for parameterizition in (self.scatterer, self.metadata):
            if parameterizition is not None:
                self.parameters.extend(parameterizition.parameters)
        if self.alpha is not None:
            self.parameters.append(self.alpha)
        

    def get_alpha(self, pars):
        try:
            return pars['alpha']
        except (KeyError, TypeError):
            if self.alpha is None:
                return 1.0
            return self.alpha

    def compare(self, calc, data, selection = None):
        if selection==None:
            return (data - calc).ravel()
        else:
            return (selection*data - selection*calc).ravel()

    def cost_func(self, data): 
        if not isinstance(self.theory, scatterpy.theory.ScatteringTheory):
            theory = self.theory(data.optics, data.shape)
        else:
            theory = self.theory

        if self.selection is not None and self._selection is None:
            if not hasattr(self.selection, 'shape'):
                # if the user specified a float fraction for selection, we need
                # to instantiate a selection array
                self._selection = np.random.random(data.shape) > (1.0-self.selection)
            
        def cost(pars):
            calc = theory.calc_holo(self.scatterer.make_from(pars),
                                    self.get_alpha(pars),
                                    self._selection)
            return self.compare(calc, data, self._selection)
        return cost

    # TODO: make a user overridable cost function that gets physical
    # parameters so that the unscaling happens only in one place (and
    # as close to the minimizer as possible).

    # TODO: Allow a layer on top of theory to do things like moving sphere
        
    
class Minimizer(SerializeByConstructor):
    def __init__(self, algorithm='nmpfit'):
        raise NotImplementedError() # pragma: nocover
    def minimize(self, parameters, cost_func, selection=None):
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
        def resid_wrapper(p, fjac=None):
            status = 0                    
            return [status, cost_func(self.pars_from_minimizer(parameters, p))]
        nmp_pars = []

        # marshall the paramters into a dict of the form nmpfit wants
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
                raise ParameterSpecficationError("nmpfit requires an "
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
                    raise ParameterSpecficationError("Parameter " + par.name +
                                                      " contains kwargs that" +
                                                      " are not supported by" +
                                                      " nmpfit")
            nmp_pars.append(d)

        # now fit it
        fitresult = nmpfit.mpfit(resid_wrapper, parinfo=nmp_pars, ftol = self.ftol,
                                 xtol = self.xtol, gtol = self.gtol, damp = self.damp,
                                 maxiter = self.maxiter, quiet = self.quiet)
        if fitresult.status == 5:
            raise MinimizerConvergenceFailed(fitresult.params, fitresult)

        result_pars = self.pars_from_minimizer(parameters, fitresult.params)
        
        if debug == True:
            return result_pars, fitresult, nmp_pars
        else:
            return result_pars, fitresult


class Parameter(SerializeByConstructor):
    def __init__(self, guess = None, limit = None, name = None, **kwargs):
        self.name = name
        self.guess = guess
        self.limit = limit
        self.kwargs = kwargs
        
        if self.fixed:
            if guess is not None and guess != limit:
                raise GuessOutOfBoundsError(self)
            self.guess = limit
        else:
            if limit is not None:
                if guess > limit[1] or guess < limit[0]:
                    raise GuessOutOfBoundsError(self)
                                 
            if guess is not None:
                if abs(guess) > 1e-12:
                    self.scale_factor = abs(guess)
                else: # values near 0
                    if limit is not None:
                        self.scale_factor = (limit[1] - limit[0])/10.
                    else:
                        self.scale_factor = 1. # guess if necessary
            elif limit is not None:
                self.scale_factor = np.sqrt(limit[0]*limit[1])
            else:
                raise ParameterSpecficationError("In order to specify a parameter "
                                                    "you must provide at least an "
                                                    "initial guess or limit")
    @property
    def fixed(self):
        if self.limit is not None:
            try:
                self.limit[1]
            except TypeError:
                return True
        return False
        
    def scale(self, physical):
        """
        Scales parameters to approximately unity

        Parameters
        ----------
        physical: np.array(dtype=float)

        Returns
        -------
        scaled: np.array(dtype=float)
        """
        return physical / self.scale_factor
     
    def unscale(self, scaled):
        """
        Inverts scale's transformation

        Parameters
        ----------
        scaled: np.array(dtype=float)

        Returns
        -------
        physical: np.array(dtype=float)
        """
        return scaled * self.scale_factor
"""
    def __mul__(self, other):
        def mult(x):
            # attempt multiplication of each element, if we fail, assume it was
            # something like None or a string and we just want to return the
            # value
                
            try:
                if not np.isscalar:
                    return np.array(x) * other
                # try an addition first since strings have multiplication
                # defined but we don't want to multiply them
                x+2 
                return x*other
            except TypeError:
                return x

        kwargs = {}
        for key in self.kwargs:
            try:
                self.kwargs[key] + 2
                kwargs[key] = self.kwargs[key] * other
            except TypeError:
                kwargs[key] = self.kwargs[key]
            
        return Parameter(mult(self.guess), mult(self.limit), self.name,
                         **kwargs)
    
    def __rmul__(self, other):
        return self.__mul__(other)
"""

# ComplexParameters must be explicitly created. We will disallow sugar like
# par(1.59) + 1e-4j.
class ComplexParameter(Parameter):
    def __init__(self, real, imag, name = None):
        '''
        real and imag may be scalars or Parameters. If Parameters, they must be
        pure real.
        '''
        if not isinstance(real, Parameter):
            real = Parameter(real, real)
        self.real = real
        if not isinstance(imag, Parameter):
            imag = Parameter(imag, imag)
        self.imag = imag
        self.name = name

    @property
    def guess(self):
        try:
            return self.real.guess + self.imag.guess*1.0j
        except AttributeError: # in case self.imag is a scalar
            #TODO: won't work if self.real is scalar, self.imag is a Parameter
            return self.real.guess + self.imag * 1.j

def fit(model, data, minimizer=Nmpfit()):
    time_start = time.time()

    try:
        fitted_pars, minimizer_info = minimizer.minimize(model.parameters,
                                                         model.cost_func(data))
        converged = True
    except MinimizerConvergenceFailed as cf:
        warnings.warn("Minimizer Convergence Failed, your results may not be "
                      "correct")
        fitted_pars, minimizer_info  = cf.result, cf.details
        converged = False

    fitted_scatterer = model.scatterer.make_from(fitted_pars)
        
    theory = model.theory(data.optics, data.shape)
    fitted_holo = theory.calc_holo(fitted_scatterer, model.get_alpha(fitted_pars)) 
    
    chisq = float((((fitted_holo-data))**2).sum() / fitted_holo.size)
    rsq = float(1 - ((data - fitted_holo)**2).sum()/((data - data.mean())**2).sum())

    time_stop = time.time()

    return FitResult(fitted_pars, fitted_scatterer, chisq, rsq, converged,
                     time_stop - time_start, model, minimizer, minimizer_info)

    
# provide a shortcut name for Parameter since users will have to type it a lot
par = Parameter
