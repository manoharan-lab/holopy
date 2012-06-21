# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca
# W. Perry, Jerome Fung, and Ryan McGorty
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
Metaclass to provide default saving to yaml files for scatterpy and holopy objects

yaml files are structured text files designed to be easy for humans to
read and write but also easy for computers to read.  Holopy uses them
to store information about experimental conditions and to describe
analysis procedures.

.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.harvard.edu>
"""
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
import inspect
import yaml

# ordered_dump code is heavily inspired by the source of PyYAML's represent_mapping
def ordered_dump(dumper, tag, data):
    value = []
    node = yaml.nodes.MappingNode(tag, value)
    for key, item in data.iteritems():
        node_key = dumper.represent_data(key)
        node_value = dumper.represent_data(item)
        value.append((node_key, node_value))

    return node

# Metaclass black magic to eliminate need for adding yaml_tag lines to classes
class SerializableMetaclass(yaml.YAMLObjectMetaclass):
    def __init__(cls, name, bases, kwds):
        super(SerializableMetaclass, cls).__init__(name, bases, kwds)
        cls.yaml_loader.add_constructor('!{0}'.format(cls.__name__), cls.from_yaml)
        cls.yaml_dumper.add_representer(cls, cls.to_yaml)

class Serializable(yaml.YAMLObject):
    """
    Base class for any object that wants a nice clean yaml output
    """
    __metaclass__ = SerializableMetaclass
    
    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_yaml_object('!{0}'.format(data.__class__.__name__), data, cls,
                                            flow_style=cls.yaml_flow_style)
    
class SerializeByConstructor(Serializable):
    @classmethod
    def to_yaml(cls, dumper, data):
        dump_dict = OrderedDict()

        # just grabbing all of the constructor arguments that have a
        # corresponding attribute is a correct serialization for many objects.
        # Ones for which it is not will need to override to_yaml
        for var in inspect.getargspec(cls.__init__).args[1:]:
            if hasattr(data, var) and getattr(data, var) is not None:
                dump_dict[var] = getattr(data, var)

        return ordered_dump(dumper, '!{0}'.format(data.__class__.__name__), dump_dict)
        return dumper.represent_mapping('!{0}'.format(data.__class__.__name__), dump_dict)

    @classmethod
    def from_yaml(cls, loader, node):
        fields = loader.construct_mapping(node, deep=True)
        return cls(**fields)
        return loader.construct_yaml_object(node, cls)
