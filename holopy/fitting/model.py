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
Classes for defining models of scattering for fitting

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <jfung@physics.harvard.edu>
"""


from copy import copy, deepcopy
import numpy as np
import xarray as xr
import inspect
from os.path import commonprefix
from .errors import ParameterSpecificationError
from ..core.holopy_object import HoloPyObject
from .parameter import Parameter, ComplexParameter
from holopy.core.utils import ensure_listlike
from holopy.core.metadata import get_values

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
        for arg in inspect.signature(self.make_scatterer).parameters:
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
    obj : :mod:`~holopy.scattering.scatterer.scatterer`
        Object containing parameters specifying any values vary in the fit.  It
        can also include numbers for any fixed values
    """
    def __init__(self, obj):
        self.obj = obj
        parameters = []
        names = []
        ties = {}

        def add_par(p, name):
            if not isinstance(p, Parameter):
                p = Parameter(p,p)
            for par_check in parameters + [None]:
                if p is par_check:
                    break
            if par_check is not None:
                # if the above loop encountered a break, it
                # means the parameter is tied

                # we will rename the parameter so that when it is printed it
                # better reflects how it is used
                new_name = tied_name(names[parameters.index(p)], name)
                names[parameters.index(p)] = new_name

                if new_name in ties:
                    # if there is already an existing tie group we need to
                    # do a few things to get the name right
                    group = ties[new_name]

                else:
                    group = [name]

                group.append(name)
                ties[new_name] = group

            else:
                if not p.fixed:
                    # add to dictionary, using parameter as key and name as value
                    # this is because we need to look up name by parameter.
                    parameters.append(p)
                    names.append(name)

        # find all the Parameter's in the obj
        for name, par in sorted(iter(obj.parameters.items()), key=lambda x: x[0]):
            if isinstance(par, ComplexParameter):
                add_par(par.real, name+'.real')
                add_par(par.imag, name+'.imag')
            elif isinstance(par, dict):
                for key, val in par.items():
                    add_par(val, name + '_' + key)
            elif isinstance(par, xr.DataArray):
                if len(par.dims)==1:
                    dimname = par.dims[0]
                else:
                    raise ParameterSpecificationError('Multi-dimensional parameters are not supported')
                for key in par[dimname]:
                    add_par(np.asscalar(par.sel(**{dimname:key})),name+'_'+np.asscalar(key))
            elif isinstance(par, Parameter):
                add_par(par, name)

        parameters = deepcopy(parameters)
        for i, name in enumerate(names):
            parameters[i].name = name
        self.parameters = parameters
        self.ties = ties

    @property
    def guess(self):
        pars = self.obj.parameters
        for key in list(pars.keys()):
            if hasattr(pars[key], 'guess'):
                if isinstance(pars[key], ComplexParameter):
                    pars[key+'.real'] = pars[key].real.guess
                    pars[key+'.imag'] = pars[key].imag.guess
                else:
                    pars[key] = pars[key].guess
        return self.make_from(pars)

    def make_from(self, parameters):
        obj_pars = {}

        for name, par in self.obj.parameters.items():
            # if this par is in a tie group, we need to work with its tie group
            # name since that will be what is in parameters
            for groupname, group in self.ties.items():
                if name in group:
                    name = groupname

            def get_val(par, name):
                if not isinstance(par, Parameter):
                    return par
                elif par.fixed:
                    return par.limit
                else:
                    return parameters[name]

            if isinstance(par, ComplexParameter):
                par_val = (get_val(par.real, name+'.real') +
                           1j * get_val(par.imag, name+'.imag'))
            elif isinstance(par, dict):
                par_val = {key:get_val(val, name+'_'+key) for key, val in par.items()}
            elif isinstance(par, xr.DataArray):
                par_val = par.copy()
                dimname = par_val.dims[0]
                for key, val in zip(par[dimname], par):
                    par_val.loc[{dimname:key}] = get_val(np.asscalar(val), name+'_'+np.asscalar(key))
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

class BaseModel(HoloPyObject):
    def __init__(self, scatterer, medium_index=None, illum_wavelen=None, illum_polarization=None, theory='auto'):
        if not isinstance(scatterer, Parametrization):
            scatterer = ParameterizedObject(scatterer)
        self.scatterer = scatterer
        self._parameters = self.scatterer.parameters
        self._use_parameter(medium_index, 'medium_index')
        self._use_parameter(illum_wavelen, 'illum_wavelen')
        self._use_parameter(illum_polarization, 'illum_polarization')
        self._use_parameter(theory, 'theory')

    @property
    def parameters(self):
        return self._parameters

    def par(self, name, schema=None, default=None):
        if hasattr(self, name) and getattr(self, name) is not None:
            return getattr(self, name)
        if schema is not None and hasattr(schema, name):
            return getattr(schema, name)
        if default is not None:
            return default

        if schema is not None:
            schematxt = " or Schema"

        raise ValueError("Cannot find value for {} in Model{}".format(name, schema))

    def get_par(self, name, pars, schema=None, default=None):
        if name in pars.keys():
            return pars.pop(name)
        elif hasattr(self, name+'_names'):
            return {key:self.get_par(name+'_'+key, pars) for key in getattr(self, name+'_names')}
        else:
            return self.par(name, schema, default)

    def get_pars(self, names, pars, schema=None):
        r = {}
        for name in names:
            r[name] = self.get_par(name, pars, schema)
        return r

    def _use_parameter(self, par, name):
        if isinstance(par, dict):
            setattr(self, name+'_names', list(par.keys()))
            for key, val in par.items():
                self._use_parameter(val, name+'_'+key)
        elif isinstance(par, xr.DataArray):
            if len(par.dims)==1:
                dimname = par.dims[0]
            else:
                raise ParameterSpecificationError('Multi-dimensional parameters are not supported')
            setattr(self, name+'_names', list(par[dimname].values))
            for key in par[dimname]:
                self._use_parameter(par.sel(**{dimname:key}).item(),name+'_'+key.item())
        else:
            setattr(self, name, par)
            if isinstance(par, Parameter):
                if par.name is None:
                    par.name = name
                self._parameters.append(par)

    def _optics_scatterer(self, pars, schema):
        optics = self.get_pars(['medium_index', 'illum_wavelen', 'illum_polarization'], pars, schema)
        scatterer = self.scatterer.make_from(pars)
        return optics, scatterer



class Model(BaseModel):
    """
    Representation of a model to fit to data

    Parameters
    ----------
    alpha : float or Parameter
        Extra scaling parameter, hopefully this will be removed by improvements
        in our theory soon.
    constraints : function or list of functions
        One or a list of constraint functions. A constraint function should take
        a scaterer as an argument and return False if you wish to disallow that
        scatterer (usually because it is un-physical for some reason)
    """
    def __init__(self, scatterer, calc_func, medium_index=None, illum_wavelen=None, illum_polarization=None, theory='auto', alpha=None,
                 use_random_fraction=None, constraints=[]):
        super().__init__(scatterer, medium_index, illum_wavelen, illum_polarization, theory)
        self.calc_func = calc_func

        self.use_random_fraction = use_random_fraction

        self._use_parameter(alpha, 'alpha')

        if len(self.parameters) == 0:
            raise ParameterSpecificationError("You must specify at least one parameter to vary in a fit")

        self.constraints = ensure_listlike(constraints)

    @property
    def guess(self):
        return [p.guess for p in self.parameters]

    @property
    def guess_dict(self):
        return {p.name: p.guess for p in self.parameters}

    def get_alpha(self, pars=None):
        try:
            return pars['alpha']
        except (KeyError, TypeError):
            if self.alpha is None:
                return 1.0
            return self.alpha

    def _calc(self, pars, schema):
        pars = copy(pars)
        alpha = self.get_par(pars=pars, name='alpha', default=1.0)
        optics, scatterer = self._optics_scatterer(pars, schema)

        valid = True
        for constraint in self.constraints:
            valid = valid and constraint(scatterer)
        if not valid:
            return np.ones_like(schema) * np.inf

        try:
            return self.calc_func(schema=schema, scatterer=scatterer, scaling=alpha, theory=self.theory, **optics)
        except:
            return np.ones_like(schema) * np.inf

    def residual(self, pars, data):
        return get_values(self._calc(pars, data)) - get_values(data)

    # TODO: Allow a layer on top of theory to do things like moving sphere
