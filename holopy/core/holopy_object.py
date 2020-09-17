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
Root class for all of holopy.  This class provides serialization to and from
yaml text file for all holopy objects.

yaml files are structured text files designed to be easy for humans to
read and write but also easy for computers to read.  HoloPy uses them
to store information about experimental conditions and to describe
analysis procedures.

.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.harvard.edu>
"""
import numpy as np
import yaml

try:
    FullLoader = yaml.FullLoader
except AttributeError:
    # Using pyyaml version < 5, technically unsafe
    FullLoader = yaml.Loader
YAMLLOADERS = (FullLoader, yaml.SafeLoader)

# Metaclass black magic to eliminate need for adding yaml_tag lines to classes
class SerializableMetaclass(yaml.YAMLObjectMetaclass):
    def __init__(cls, name, bases, kwds):
        super().__init__(name, bases, kwds)
        # Replace the normal yaml constructor with one that uses the class name
        # as the yaml tag.
        for loader in YAMLLOADERS:
            tag = '!{0}'.format(cls.__name__)
            yaml.add_constructor(tag, cls.from_yaml, Loader=loader)
        cls.yaml_dumper.add_representer(cls, cls.to_yaml)


class Serializable(yaml.YAMLObject, metaclass=SerializableMetaclass):
    """
    Base class for any object that wants a nice clean yaml output
    """

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_yaml_object('!{0}'.format(data.__class__.__name__), data, cls,
                                            flow_style=cls.yaml_flow_style)

class HoloPyObject(Serializable):
    """Ancestor class for all HoloPy classes.

    HoloPy object's purpose is to provide the machinery for saving to
    and loading from HoloPy yaml files

    """

    @property
    def _dict(self):
        return dict(self._iteritems())

    def _iteritems(self):
        for var in self.__init__.__code__.co_varnames[1:]:
            if getattr(self, var, None) is not None:
                item = getattr(self, var)
                if isinstance(item, np.ndarray) and item.ndim == 1:
                    item = list(item)
                yield var, item

    @classmethod
    def to_yaml(cls, dumper, data):
        tag = '!{0}'.format(data.__class__.__name__)
        value = []
        # heavily inspired by the source of PyYAML's represent_mapping
        node = yaml.nodes.MappingNode(tag, value)
        for key, item in data._iteritems():
            node_key = dumper.represent_data(key)
            node_value = dumper.represent_data(item)
            value.append((node_key, node_value))
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        fields = loader.construct_mapping(node, deep=True)
        return cls(**fields)

    def __repr__(self):
        keywpairs = ["{0}={1}".format(k[0], repr(k[1])) for k in self._iteritems()]
        return "{0}({1})".format(self.__class__.__name__, ", ".join(keywpairs))

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if hasattr(other, '_dict'):
            # Note, this is possibly a slightly weak form of equality, but well behaved holopy objects are equal if their _dict's are equal
            return self._dict == other._dict
        else:
            return False
