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
Classes for defining models of scattering for fitting

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <jfung@physics.harvard.edu>
"""
from __future__ import division

import inspect
import numpy as np
from copy import copy

from ..core.holopy_object import HolopyObject
from ..scattering.theory.scatteringtheory import ScatteringTheory
from .parameter import Parameter, ComplexParameter

class Parametrization(HolopyObject):
    def __init__(self, make_scatterer, parameters):
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
        self.make_scatterer = make_scatterer

    def make_from(self, parameters):
        # parameters is an ordered dictionary
        for_target = {}
        for arg in inspect.getargspec(self.make_scatterer).args:
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
        return self.make_scatterer(**for_target)

    @property
    def guess(self):
        guess_pars = {}
        for par in self.parameters:
            guess_pars[par.name] = par.guess
        return self.make_from(guess_pars)


class ParameterizedTarget(Parametrization):
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



class Model(HolopyObject):
    """
    Representation of a model to fit to data

    Parameters
    ----------
    parameters: :class:`Paramatrization`
        The parameters which can be varied in this model.  
    theory: :function:`scattering.theory.ScatteringTheory.calc_*`
        The scattering calc function that should be used to compute results for
        comparison with the data
    metadata: :class:`core.data.DataTarget`
        A DataTarget object with overrides for and of the metadata of the data
        you fit to.  Do not bother to provide entries for shape, position, and
        things that are provided in the data, use this for replacing wavelen with a
        par, or adding a use_random_fraction entry.
    alpa: float or Parameter
        Extra scaling parameter, hopefully this will be removed by improvements
        in our theory soon.  
    """
    def __init__(self, scatterer, theory, metadata=None, alpha = None):
        if not isinstance(scatterer, Parametrization):
            scatterer = ParameterizedTarget(scatterer)
        self.scatterer = scatterer

        self.theory = theory

        if (metadata is not None) and (not isinstance(metadata, Parametrization)):
            metadata = ParameterizedTarget(metadata)
        self.metadata = metadata

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
        return (data - calc).ravel()

    def cost_func(self, data):
        target = copy(data)
        if self.metadata is not None:
            target._update_metadata(self.metadata._metadata)
        def cost(pars):
            calc = self.theory(self.scatterer.make_from(pars), target, scaling =
                          self.get_alpha(pars))
            return self.compare(calc, data)
        return cost

    # TODO: make a user overridable cost function that gets physical
    # parameters so that the unscaling happens only in one place (and
    # as close to the minimizer as possible).

    # TODO: Allow a layer on top of theory to do things like moving sphere
        

    
