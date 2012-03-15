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
from collections import OrderedDict


import numpy as np
import yaml

from holopy.io.yaml_io import Serializable
from scatterpy.errors import ScattererDefinitionError

class Scatterer(Serializable):
    """
    abstract base class for scatterers

    """
    yaml_tag = u'!Scatterer'
    
    def __init__(self):
        raise NotImplementedError

    def validate(self):
        """
        Check that a scatterer is physically realistic.  Theories should call
        this function before attempting computation with a scatterer.  

        Parameters
        ----------
        None

        Returns
        -------
        True is scatterer is valid

        Raises
        ------
        InvalidScatterer or subclass if the scatterer is unphysical for some reason
        """
        # We default to just returning True subclasses that implement overlap
        # checking or other validity constraints override this
        return True

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
    
        return OrderedDict(sorted(self.__dict__.items(), key = lambda t: t[0]))


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
        return cls.__init__(**parameters)
    
    @property
    def parameter_list(self):
        """
        Return's the scatterer's parameters as an 1d array in a defined order.
        This form is suitable for passing to a minimizer

        Note: if the scatter has complex values (like index of refraction) they
        need to be split into two seperate variables
        """
        raise NotImplementedError

    @classmethod
    def make_from_parameter_list(cls, params):
        """
        Make a new scatterer from a parameter list of the form reterned by
        parameter_list().
        """
        raise NotImplementedError


class xyzTriple(np.ndarray):
    """
    
    """
    def __new__(cls, x=None, y=None, z=None, xyz=None):
        if xyz is not None:
            if np.isscalar(xyz) or len(xyz) != 3:
                raise InvalidxyzTriple(repr(xyz))
        elif (x is not None) and (y is not None) and (z is not None):
            xyz =  np.array([x, y, z])
        else:
            raise InvalidxyzTriple('x={0}, y={1}, z={2}'.format(x, y, z))

        return np.asarray(xyz).view(cls)

    def __array_finalize__(self, obj):
        pass

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    @property
    def parameters(self):
        return {'x': self[0], 'y': self[1], 'z': self[2]}

    def parameters_prefix(self, prefix):
        p = {}
        for i, c in enumerate(('x', 'y', 'z')):
            p['{0}_{1}'.format(prefix, c)] = self[i]
        return p
        

class InvalidxyzTriple(Exception):
    def __init__(self, xyz):
       self.xyz = xyz

       
    def __str__(self):
       return ("Specification of xyzTriple({0}) is not valid, should be "
                "specified as (x, y, z)".format(self.xyz))

def xyzTriple_yaml_representer(dumper, data):
    return dumper.represent_scalar(u'!xyzTriple', str(list(data)))
yaml.add_representer(xyzTriple, xyzTriple_yaml_representer)

def xyzTriple_yaml_constructor(loader, node):
    value = loader.construct_scalar(node)
    return xyzTriple(xyz = [float(v) for v in value[1:-1].split(',')])
yaml.add_constructor(u'!xyzTriple', xyzTriple_yaml_constructor)
