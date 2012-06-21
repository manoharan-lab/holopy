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



class FitResult(SerializeByConstructor):
    def __init__(self, scatterer, alpha, chisq, rsq, converged, time, model,
                 minimizer, minimization_details):
        self.scatterer = scatterer
        self.alpha = alpha
        self.chisq = chisq
        self.rsq = rsq
        self.converged = converged
        self.time = time
        self.model = model
        self.minimizer = minimizer
        self.minimization_details = minimization_details
        
    def __repr__(self):
        return ("{s.__class__.__name__}(scatterer={s.scatterer}, "
                "alpha={s.alpha}, chisq={s.chisq}, rsq={s.rsq}, "
                "converged={s.converged}, time={s.time}, model={s.model}, "
                "minimizer={s.minimizer}, "
                "minimization_details={s.minimization_details})".format(s=self))  #pragma: no cover

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
    selection : array of integers (optional)
        An array with 1's in the locations of pixels where you
        want to calculate the field, defaults to 1 at all pixels

    Notes
    -----
    Any arbitrary parameterization can be used by simply specifying a
    make_scatterer function which can turn the parameters into a scatterer
    
    """
    def __init__(self, parameters, theory, make_scatterer=None, selection=None):
        self._user_make_scatterer = make_scatterer
        self.make_scatterer = make_scatterer

        self.theory = theory
        self.selection = selection

        self.scatterer = None
        self.parameters = []

        def unpack_scatterer(scatterer):
            parameters = []
            for name, par in scatterer.parameters.iteritems():
                def add_par(p, name):
                    p.name = name
                    if p.fixed:
                        parameters.append((name, p.limit))
                    else:
                        parameters.append((name, p))
                if isinstance(par, ComplexParameter):
                    add_par(par.real, name+'.real')
                    add_par(par.imag, name+'.imag')
                elif isinstance(par, Parameter):
                    add_par(par, name)
                else:
                    # probably just a number, (ie fixed), so just return it
                    parameters.append((name, par))

            if self.scatterer is None:
                self.scatterer = scatterer.from_parameters(dict(parameters))
            else:
                raise ModelDefinitionError(
                   "A model cannot contain more than one scatterer.  If you want"
                   "to include multiple scatterers include them in a single"
                   "composite Scatterer")

            return [p[1] for p in parameters if isinstance(p[1], Parameter)]

        if isinstance(parameters, (list, tuple)):
            for item in parameters:
                if isinstance(item, scatterpy.scatterer.Scatterer):
                    self.parameters.extend(unpack_scatterer(item))
                elif isinstance(item, Parameter):
                    self.parameters.append(item)
                else:
                    raise ModelDefinitionError(
                        "{0} is not a valid parameter".format(item))
        elif isinstance(parameters, scatterpy.scatterer.Scatterer):
            self.parameters = unpack_scatterer(parameters)
        elif isinstance(parameters, Parameter):
            self.parameters = [parameters]
                
        if self.scatterer is not None and make_scatterer is None:
            def make_scatterer(pars):
                for_scatterer = self.scatterer.parameters
                par_dict = {}
                if isinstance(pars, dict):
                    par_dict = pars
                else:
                    for i, p in enumerate(self.parameters):
                        par_dict[p.name] = pars[i] 
                for par, val in par_dict.iteritems():
                    for_scatterer[par] = val
                try:
                    del for_scatterer['alpha']
                except KeyError:
                    pass
                return self.scatterer.from_parameters(for_scatterer)
            
            self.make_scatterer = make_scatterer
        elif make_scatterer is not None:
            self.make_scatterer = make_scatterer
        else:
            raise ModelDefinitionError(
                "You must either give a model a template scatterer in its "
                "parameters, or provide a custom make_scatterer function.")

        
    @property
    def guess_scatterer(self):
        pars = self.scatterer.parameters
        for key, val in pars.iteritems():
            if isinstance(val, Parameter):
                pars[key] = val.guess
        return self.scatterer.from_parameters(pars)

        
    @property
    def alpha_par(self):
        for i, par in enumerate(self.parameters):
            if par.name == 'alpha':
                return par
        return None
            
        
    @property
    def guess_alpha(self):
        for i, par in enumerate(self.parameters):
            if par.name == 'alpha':
                return par.guess
        return 1.0
            
    def make_scatterer_from_par_values(self, par_values):
        all_pars = {}
        for i, p in enumerate(self.parameters):
            all_pars[p.name] = p.unscale(par_values[i])
        if self._user_make_scatterer is not None:
            for_scatterer = {}
            for arg in inspect.getargspec(self.make_scatterer).args:
                for_scatterer[arg] = all_pars[arg] 
            # user make_scatterer functions will most likely take traditional
            # function arguments, so we need reorganize for that
            return self._user_make_scatterer(**for_scatterer)
        else:
            return self.make_scatterer(all_pars)
        
    # TODO: add a make_optics function so that you can have parameters
    # affect optics things (fit to beam divergence, lens abberations, ...)

    def compare(self, calc, data, selection = None):
        if selection==None:
            return (data - calc).ravel()
        else:
            return (selection*data - selection*calc).ravel()
    
    def alpha(self, par_values):
        for i, par in enumerate(self.parameters):
            if par.name == 'alpha':
                return par.unscale(par_values[i])
        # if the user does not provide alpha as a parameter, we just use 1.0,
        # the default alpha for theories (which is hopefully what they want)
        return 1.0
    
    def cost_func(self, data, selection = None): 
        if not isinstance(self.theory, scatterpy.theory.ScatteringTheory):
            theory = self.theory(data.optics, data.shape)
        else:
            theory = self.theory
            
        def cost(par_values, selection=None):
            calc = theory.calc_holo(
                self.make_scatterer_from_par_values(par_values),
                self.alpha(par_values), selection)
            return self.compare(calc, data, selection)
        return cost

    # TODO: make a user overridable cost function that gets physical
    # parameters so that the unscaling happens only in one place (and
    # as close to the minimizer as possible).

    # TODO: Allow a layer on top of theory to do things like moving sphere

class ModelDefinitionError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg
    
class InvalidParameterSpecification(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg

class GuessOutOfBounds(InvalidParameterSpecification):
    def __init__(self, parameter):
        self.par = parameter
    def __str__(self):
        if self.par.fixed:
            return "guess {s.guess} does not match fixed value {s.limit}".format(s=self.par)
        return "guess {s.guess} is not within bounds {s.limit}".format(s=self.par)
    
class MinimizerConvergenceFailed(Exception):
    def __init__(self, result, details):
        self.result = result
        self.details = details
        
    
class Minimizer(SerializeByConstructor):
    def __init__(self, algorithm='nmpfit', ):
        raise NotImplementedError()

    def minimize(self, parameters, cost_func, selection=None):
        raise NotImplementedError()
    
    def __repr__(self):
        return "Minimizer(algorithm='{0}')".format(self.algorithm)

class Nmpfit(Minimizer):
    def __init__(self, quiet = False, ftol = 1e-10, xtol = 1e-10, gtol = 1e-10, damp = 0,
                 maxiter = 100, err=None):
        # do the import on demand so the user doesn't need a minimizer present
        # unless they are using it
        from holopy.third_party import nmpfit
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.damp = 0
        self.maxiter = 100
        self.quiet = quiet
        self.err = None

    def minimize(self, parameters, cost_func, selection = None):
        from holopy.third_party import nmpfit
        def resid_wrapper(p, fjac=None):
            status = 0                    
            return [status, cost_func(p, selection)]
        nmp_pars = []

        # marshall the paramters into a dict of the form nmpfit wants
        for i, par in enumerate(parameters):
            d = {'parname': par.name}
            if par.limit is not None:
                d['limited'] = [par.scale(l) is not None for l in par.limit]
                d['limits'] = par.scale(np.array(par.limit))
            else:
                d['limited'] = [False, False]    
            if par.guess is not None:
                d['value'] = par.scale(par.guess)
            else:
                raise InvalidParameterSpecification("nmpfit requires an "
                                                    "initial guess for all "
                                                    "parameters")
            nmp_pars.append(d)

        # now fit it
        fitresult = nmpfit.mpfit(resid_wrapper, parinfo=nmp_pars, ftol = self.ftol,
                                 xtol = self.xtol, gtol = self.gtol, damp = self.damp,
                                 maxiter = self.maxiter, quiet = self.quiet)
        if fitresult.status > 3:
            raise MinimizerConvergenceFailed(fitresult.params, fitresult)

        return fitresult.params, fitresult
        


class Parameter(SerializeByConstructor):
    def __init__(self, guess = None, limit = None, name = None, misc = None):
        self.name = name
        self.guess = guess
        self.limit = limit
        self.misc = misc
        
        if self.fixed:
            if guess is not None and guess != limit:
                raise GuessOutOfBounds(self)
            self.guess = limit
        else:
            if limit is not None:
                try:
                    if guess > limit[1] or guess < limit[0]:
                        raise GuessOutOfBounds(self)
                except TypeError:
                    if (guess.real !=0 or limit[1].real != 0 or
                        limit[0].real != 0):
                        raise GuessOutOfBounds(self)
                    if guess.imag > limit[1].imag or guess.imag < limit[0].imag:
                        raise GuessOutOfBounds(self)
                    
            if guess is not None:
                self.scale_factor = guess
            elif limit is not None:
                self.scale_factor = np.sqrt(limit[0]*limit[1])
            else:
                raise InvalidParameterSpecification("In order to specify a parameter "
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

    def __add__(self, other):
        # Compose a Parameter and a complex number or complex Parameter into a
        # ComplexParameter
        if isinstance(other, Parameter):
            def imag_or_bust(val):
                if val is None:
                    return None
                elif not np.isscalar(val):
                    return [imag_or_bust(v) for v in val]
                elif np.iscomplex(val):
                    return val.imag
                else:
                    raise InvalidParameterSpecification(
                        "Addition of parameters is only defined for composing "
                        "complex parameters")

            return ComplexParameter(self, Parameter(imag_or_bust(other.guess),
                                                    imag_or_bust(other.limit),
                                                    other.name,
                                                    other.misc))
        else:
            return ComplexParameter(self, other.imag)

    def __radd__(self, other):
        return self.__add__(self, other)

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

        return Parameter(mult(self.guess), mult(self.limit), self.name,
                         self.misc)
    
    def __rmul__(self, other):
        return self.__mul__(other)
            
    def __repr__(self):
        args = []
        if self.guess is not None:
            args.append('guess={0}'.format(self.guess))
        if self.limit is not None:
            args.append('limit={0}'.format(self.limit))
        if self.misc is not None:
            args.append('misc={0}'.format(self.misc))
        return "Parameter(name={0}, {1})".format(repr(self.name), ', '.join(args))
            
# user in general will not be creating ComplexParameters, they are created when
# you do something like: par(1.59) + 1e-4j or par(1.59) + par(1e-4j)
class ComplexParameter(Parameter):
    def __init__(self, real, imag, name = None):
        self.real = real
        self.imag = imag                              
        self.name = name

    @property
    def guess(self):
        return self.real.guess + self.imag.guess*1.0j

    def __repr__(self):
        return "{0} + {1}".format(self.real, 1.0j*self.imag)
    

def fit(model, data, minimizer=Nmpfit()):
    time_start = time.time()

    try:
        fitted_pars, minimizer_info = minimizer.minimize(model.parameters,
                                                         model.cost_func(data),
                                                         model.selection)
        converged = True
    except MinimizerConvergenceFailed as cf:
        warnings.warn("Minimizer Convergence Failed, your results may not be "
                      "correct")
        fitted_pars, minimizer_info  = cf.result, cf.details
        converged = False
        
    
    fitted_scatterer = model.make_scatterer_from_par_values(fitted_pars)
    fitted_alpha = model.alpha(fitted_pars)
    theory = model.theory(data.optics, data.shape)
    fitted_holo = theory.calc_holo(fitted_scatterer, fitted_alpha)
    
    chisq = float((((fitted_holo-data))**2).sum() / fitted_holo.size)
    rsq = float(1 - ((data - fitted_holo)**2).sum()/((data - data.mean())**2).sum())

    time_stop = time.time()

    return FitResult(fitted_scatterer, fitted_alpha, chisq, rsq, converged,
                     time_stop - time_start, model, minimizer, minimizer_info)

    
# provide a shortcut name for Parameter since users will have to type it a lot
par = Parameter


# Archiving:
# Model (parameters, theory, cost function, 

################################################################
# Fitseries engine

# default
# Load, normalize, background
# fit
# archive

# provide customization hooks
# prefit - user supplied
# fit
# postfit - user supplied
# archive

# archive to a unique directory name (probably from time and hostname)
