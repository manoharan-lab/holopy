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

'''
The abstract base class for all scattering objects

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''
from __future__ import division
from collections import defaultdict
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
from itertools import chain
from copy import copy

import numpy as np
import yaml

from scatterpy.io import SerializeByConstructor
from scatterpy.errors import ScattererDefinitionError


class Scatterer(SerializeByConstructor):
    """
    Base class for scatterers

    """
    def __init__(self):
        raise NotImplementedError()

    def translate(self, x, y, z):
        """
        Make a copy of this scatterer translated to a new location

        Parameters
        ----------
        x, y, z : float
            Value of the translation along each axis

        Returns
        -------
        translated : Scatterer
            A copy of this scatterer translated to a new location
        """
        raise NotImplementedError() #pragma: no cover
    

    # eliminate parameters and from_parameters?  This is kind of fitting
    # specific information.  Or should it be in serializable?  In many ways this
    # is just a slight variation on what we do to put something in yaml format.
    # It is probably possible to have to_dict, to_string, to_yaml all with
    # mostly common code
    @property
    def parameters(self):
        """
        Get a dictionary of this scatterer's parameters

        Parameters
        ----------
        None

        Returns
        -------
        parameters: dict
            A dictionary of this scatterer's parameters.  This dict can be
            passed to Scatterer.from_parameters to make a copy of this scatterer
        """

        # classes that have anything complicated happening with their variables
        # should override this, but for simple classes the variable dict is the
        # correct answer

        # we return an OrderedDict to make it easer to keep parameters in the
        # same order in cases where a list of parameters is needed and will be
        # later passed to Scatterer.from_parameters

        def expand(key, par):
            if isinstance(par, (list, tuple, np.ndarray)):
                subs = (expand('{0}[{1}]'.format(key, p[0]), p[1]) for p in enumerate(par)) 
                return chain(*subs)
            # could be eliminated because of ComplexParameter
            if hasattr(par, 'imag') and par.imag != 0:
                return [('{0}.real'.format(key), par.real),
                        ('{0}.imag'.format(key), par.imag)]
            else:
                return [(key, par)]

        return OrderedDict(sorted(chain(*[expand(*p) for p in
                                          self.__dict__.iteritems()]))) 
    

    @classmethod
    def from_parameters(cls, parameters):
        """
        Create a Scatterer from a dictionary of parameters

        Parameters
        ----------
        parameters: dict or list
            Parameters for a scatterer.  This should be of the form returned by
            Scatterer.parameters.

        Returns
        -------
        scatterer: Scatterer class
            A scatterer with the given parameter values
        """
        # This will need to be overriden for subclasses that do anything
        # complicated with parameters

        collected = defaultdict(dict)

        for key, val in parameters.iteritems():
            tok = key.split('.', 1)
            if len(tok) > 1:
                collected[tok[0]][tok[1]] = val
            else:
                collected[key] = val

        collected_arrays = defaultdict(dict)
        for key, val in collected.iteritems():
            tok = key.split('[', 1)
            if len(key.split('[', 1)) > 1:
                sub_key, n = key.split('[', 1)
                n = int(n[:-1])
                collected_arrays[sub_key][n] = val
            else:
                collected_arrays[key] = val

        built = {}

        def build(par):
            if isinstance(par, dict):
                # could be eliminated because of ComplexParameter
                if sorted(par.keys()) == ['imag', 'real']:
                    return par['real'] + 1.0j * par['imag']
                elif reduce(lambda x, i: isinstance(i, int) and x,
                            par.keys(), True):

                    d = [p[1] for p in sorted(par.iteritems(), key =
                                                 lambda x: x[0])]
                    return [build(p) for p in d]
            return par
            
        for key, val in collected_arrays.iteritems():
            built[key] = build(val)

        return cls(**built)

    def like_me(self, **overrides):
        pars = dict(self.__dict__)
        pars.update(overrides)

        return self.__class__(**pars)

class SingleScatterer(Scatterer):
    def __init__(self, center = None):
        if np.isscalar(center) or len(center) != 3:
            raise ScattererDefinitionError("center specified as {0}, center "
                "should be specified as (x, y, z)".format(center), self)
        self.center = center

    def translate(self, x, y, z):
        """
        Make a copy of this scatterer translated to a new location

        Parameters
        ----------
        x, y, z : float
            Value of the translation along each axis

        Returns
        -------
        translated : Scatterer
            A copy of this scatterer translated to a new location
        """
        new = copy(self)
        new.center = self.center + np.array([x, y, z])
        return new
    @property
    def x(self):
        return self.center[0]
    @property
    def y(self):
        return self.center[1]
    @property
    def z(self):
        return self.center[2]        

class SphericallySymmetricScatterer(SingleScatterer):
    def rotate(self, alpha, beta, gamma):
        return copy(self)
    

    # Legacy, Deprecated.  Kept around in case anyone has files saved with
    # xyzTriples, this should read thim into the new format, though I haven't
    # tested it -tgd 2012-04-10
def xyzTriple_yaml_constructor(loader, node):
    value = loader.construct_scalar(node)
    return [float(v) for v in value[1:-1].split(',')]
yaml.add_constructor(u'!xyzTriple', xyzTriple_yaml_constructor)
