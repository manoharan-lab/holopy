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
Classes for defining models of scattering for fitting

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <jfung@physics.harvard.edu>
"""
from __future__ import division

import inspect
import numpy as np
from os.path import commonprefix
from .errors import ModelDefinitionError
from ..core.holopy_object import HoloPyObject
from .parameter import Parameter, ComplexParameter
from holopy.core.helpers import ensure_listlike


class Parametrization(HoloPyObject):
    """
    Description of free parameters and how to make a scatterer from them

    Parameters
    ----------
    make_scatterer : function
        A function which should take the Parametrization parameters by name as
        keyword arguments and return a scatterer
    parameters : list
        The list of parameters for this Parametrization
    """
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
        for_schema = {}
        for arg in inspect.getargspec(self.make_scatterer).args:
            if (arg + '.real') in parameters and (arg + '.imag') in parameters:
                for_schema[arg] = (parameters[arg + '.real'] + 1.j *
                                   parameters[arg + '.imag'])
            elif (arg + '.real') in self._fixed_params and \
                    (arg + '.imag') in parameters:
                for_schema[arg] = (self._fixed_params[arg + '.real'] + 1.j *
                                   parameters[arg + '.imag'])
            elif (arg + '.real') in parameters and (arg + '.imag') in \
                    self._fixed_params:
                for_schema[arg] = (parameters[arg + '.real'] + 1.j *
                                   self._fixed_params[arg + '.imag'])
            else:
                for_schema[arg] = parameters[arg]
        return self.make_scatterer(**for_schema)

    @property
    def guess(self):
        guess_pars = {}
        for par in self.parameters:
            guess_pars[par.name] = par.guess
        return self.make_from(guess_pars)

def tied_name(name1, name2):
    common_suffix = commonprefix([name1[::-1], name2[::-1]])[::-1]
    return common_suffix.strip(':_')

class ParameterizedObject(Parametrization):
    """
    Specify parameters for a fit by including them in an object

    Parameters are named automatically from their position in the object

    Parameters
    ----------
    obj : :mod:`.scatterer`
        Object containing parameters specifying any values vary in the fit.  It
        can also include numbers for any fixed values
    """
    def __init__(self, obj):
        self.obj = obj

        # find all the Parameter's in the obj
        parameters = []
        ties = {}
        for name, par in obj.parameters.iteritems():
            def add_par(p, name):
                if p in parameters:
                    # if the parameter is already in the parameters list, it
                    # means the parameter is tied

                    # we will rename the parameter so that when it is printed it
                    # better reflects how it is used
                    new_name = tied_name(p.name, name)

                    if p.name in ties:
                        # if there is already an existing tie group we need to
                        # do a few things to get the name right
                        group = ties[p.name]
                        if p.name != new_name:
                            del ties[p.name]
                    else:
                        group = [p.name]

                    group.append(name)
                    ties[new_name] = group
                    p.name = new_name

                else:
                    p.name = name
                    if not p.fixed:
                        parameters.append(p)
            if isinstance(par, ComplexParameter):
                add_par(par.real, name+'.real')
                add_par(par.imag, name+'.imag')
            elif isinstance(par, Parameter):
                add_par(par, name)

        self.parameters = parameters
        self.ties = ties

    @property
    def guess(self):
        pars = self.obj.parameters
        for key in pars.keys():
            if hasattr(pars[key], 'guess'):
                if isinstance(pars[key], ComplexParameter):
                    pars[key+'.real'] = pars[key].real.guess
                    pars[key+'.imag'] = pars[key].imag.guess
                else:
                    pars[key] = pars[key].guess
        return self.make_from(pars)

    def make_from(self, parameters):
        obj_pars = {}

        for name, par in self.obj.parameters.iteritems():
            # if this par is in a tie group, we need to work with its tie group
            # name since that will be what is in parameters
            for groupname, group in self.ties.iteritems():
                if name in group:
                    name = groupname

            def get_val(par, name):
                if par.fixed:
                    return par.limit
                else:
                    return parameters[name]

            if isinstance(par, ComplexParameter):
                par_val = (get_val(par.real, name+'.real') +
                           1j * get_val(par.imag, name+'.imag'))
            elif isinstance(par, Parameter):
                par_val = get_val(par, name)
            else:
                par_val = par


            if name in self.ties:
                for tied_name in self.ties[name]:
                    obj_pars[tied_name] = par_val
            else:
                obj_pars[name] = par_val



        return self.obj.from_parameters(obj_pars)

def limit_overlaps(fraction=.1):
    """
    Generator for constraint prohibiting overlaps beyond a certain tolerance

    Parameters
    ----------
    fraction : float
        Fraction of the sphere diameter that the spheres should be allowed to
        overlap by


    Returns
    -------
    constraint : function (scatterer -> bool)
        A function which tests scatterers to see if the exceed the specified
        tolerance
    """
    def constraint(s):
        return s.largest_overlap() < ((np.min(s.r) * 2) * fraction)
    return constraint


class Model(HoloPyObject):
    """
    Representation of a model to fit to data

    Parameters
    ----------
    parameters  :class:`.Paramatrization`
        The parameters which can be varied in this model.
    theory : :func:`scattering.theory.ScatteringTheory.calc_*`
        The scattering calc function that should be used to compute results for
        comparison with the data
    alpha : float or Parameter
        Extra scaling parameter, hopefully this will be removed by improvements
        in our theory soon.
    constraints : function or list of functions
        One or a list of constraint functions. A constraint function should take
        a scaterer as an argument and return False if you wish to disallow that
        scatterer (usually because it is un-physical for some reason)
    """
    def __init__(self, scatterer, theory, alpha=None,
                 use_random_fraction=None, constraints=[]):
        if not isinstance(scatterer, Parametrization):
            scatterer = ParameterizedObject(scatterer)
        self.scatterer = scatterer
        self.theory = theory
        self.use_random_fraction = use_random_fraction

        if isinstance(alpha, Parameter) and alpha.name is None:
            alpha.name = 'alpha'
        self.alpha = alpha

        self.parameters = self.scatterer.parameters
        if isinstance(self.alpha, Parameter):
            self.parameters.append(self.alpha)

        if len(self.parameters) == 0:
            raise ModelDefinitionError("You must specify at least one parameter to vary in a fit")

        self.constraints = ensure_listlike(constraints)


    def get_alpha(self, pars=None):
        try:
            return pars['alpha']
        except (KeyError, TypeError):
            if self.alpha is None:
                return 1.0
            return self.alpha

    def guess_holo(self, schema):
        if isinstance(self.alpha, Parameter):
            alpha = self.alpha.guess
        else:
            alpha = self.alpha
        return self.theory(self.scatterer.guess, schema, alpha)

    # TODO: Allow a layer on top of theory to do things like moving sphere
