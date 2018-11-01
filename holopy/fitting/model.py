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
from ..scattering.errors import MissingParameter
from ..core.holopy_object import HoloPyObject
from holopy.core.utils import ensure_listlike
from holopy.core.metadata import get_values
class limit_overlaps(HoloPyObject):
    """
    Constraint prohibiting overlaps beyond a certain tolerance.
    fraction is the largest overlap allowed, in terms of sphere diameter.

    """
    def __init__(self, fraction=.1):
        self.fraction = fraction

    def check(self, s):
        return s.largest_overlap() <= ((np.min(s.r) * 2) * self.fraction)


class BaseModel(HoloPyObject):
    def __init__(self, scatterer, medium_index=None, illum_wavelen=None,
                 illum_polarization=None, theory='auto', constraints=None):
        if not isinstance(scatterer, ParameterizedObject):
            scatterer = ParameterizedObject(scatterer)
        self.scatterer = scatterer
        self.constraints = ensure_listlike(constraints)
        self._parameters = self.scatterer.parameters
        self._use_parameter(medium_index, 'medium_index')
        self._use_parameter(illum_wavelen, 'illum_wavelen')
        self._use_parameter(illum_polarization, 'illum_polarization')
        self._use_parameter(theory, 'theory')

    @property
    def parameters(self):
        return self._parameters

    def get_parameter(self, name, pars, schema=None):
        if name in pars.keys():
            return pars.pop(name)
        elif hasattr(self, name) and getattr(self, name) is not None:
            return getattr(self, name)
        elif hasattr(self, name+'_names'):
            return {key: self.get_parameter(name + '_' + key, pars)
                    for key in getattr(self, name + '_names')}
        elif schema is not None and hasattr(schema, name):
            return getattr(schema, name)
        else:
            raise MissingParameter(name)

    def _use_parameter(self, par, name):
        from holopy.inference.prior import Prior, Fixed, ComplexPrior #TODO deleteme
        if isinstance(par, dict):
            setattr(self, name+'_names', list(par.keys()))
            for key, val in par.items():
                self._use_parameter(val, name+'_'+key)
        elif isinstance(par, xr.DataArray):
            if len(par.dims)==1:
                dimname = par.dims[0]
            else:
                msg = 'Multi-dimensional parameters are not supported'
                raise ParameterSpecificationError(msg)
            setattr(self, name + '_names', list(par[dimname].values))
            for key in par[dimname]:
                self._use_parameter(
                    par.sel(**{dimname: key}).item(), name + '_' + key.item())
        else:
            setattr(self, name, par)
            if isinstance(par, Prior):
                if par.name is None:
                    par.name = name
                self._parameters.append(par)

    def _optics_scatterer(self, pars, schema):
        optics_keys = ['medium_index', 'illum_wavelen', 'illum_polarization']
        optics = {key:self.get_parameter(key, pars, schema)
                            for key in optics_keys}
        scatterer = self.scatterer.make_from(pars)
        return optics, scatterer


class Model(BaseModel):
    """
    Representation of a model to fit to data

    Parameters
    ----------
    scatterer :
    calc_func :
    medium_index : float
    illum_wavelen : float
    illum_polarization :
    theory : {'auto', `holpy.scattering.theory.ScatteringTheory`}
        The theory used to compute the scattering from the particle.
        Default is ``'auto'``, which eventually calls
        ``holopy.scattering.determine_theory`` when the model is
        evaluated.
    alpha : float or Parameter
        Extra scaling parameter, hopefully this will be removed by
        improvements in our theory soon.
    constraints : function or list of functions
        One or a list of constraint objects. A constraint object
        should have a method 'check' that takes a scatterer as an
        argument and returns False if the scatterer is disallowed
        (usually because it is unphysical for some reason).
    """
    def __init__(self, scatterer, calc_func, medium_index=None,
                 illum_wavelen=None, illum_polarization=None, theory='auto',
                 alpha=None, constraints=[]):
        super().__init__(scatterer, medium_index, illum_wavelen,
                         illum_polarization, theory, constraints)
        self.calc_func = calc_func
        self._use_parameter(alpha, 'alpha')

        if len(self.parameters) == 0:
            msg = "You must specify at least one parameter to vary in a fit"
            raise ParameterSpecificationError(msg)

    @property
    def guess(self):
        return [p.guess for p in self.parameters]

    @property
    def guess_dict(self):
        return {p.name: p.guess for p in self.parameters}

    def _calc(self, pars, schema):
        pars = copy(pars)
        try:
            alpha = self.get_parameter(pars=pars, name='alpha')
        except MissingParameter:
            alpha=1
        optics, scatterer = self._optics_scatterer(pars, schema)

        valid = True
        for constraint in self.constraints:
            valid = valid and constraint.check(scatterer)
        if not valid:
            return np.ones_like(schema) * np.inf

        try:
            return self.calc_func(schema=schema, scatterer=scatterer, scaling=alpha, theory=self.theory, **optics)
        except:
            return np.ones_like(schema) * np.inf

    def residual(self, pars, data):
        return get_values(self._calc(pars, data) - data).flatten()

    # TODO: Allow a layer on top of theory to do things like moving sphere
