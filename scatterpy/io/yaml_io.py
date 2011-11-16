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

'''
IO of scatterpy objects to and from yaml files

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''



import yaml
import numpy as np

def save(outf, obj):
    yaml.dump(obj, file(outf, 'w'))

def load(inf):
    yaml.load(file(inf))


# Represent 1d ndarrays as lists in yaml files because it makes them much
# prettier
def ndarray_representer(dumper, data):
    if data.ndim == 1:
        return dumper.represent_list([float(a) for a in data])
    else:
        raise NotImplementedError

yaml.add_representer(np.ndarray, ndarray_representer)



# Metaclass black magic to eliminate need for adding yaml_tag lines to classes
class SerializableMetaclass(yaml.YAMLObjectMetaclass):
    def __init__(cls, name, bases, kwds):
        super(SerializableMetaclass, cls).__init__(name, bases, kwds)
        cls.yaml_loader.add_constructor(cls.yaml_tag, cls.from_yaml)
        cls.yaml_dumper.add_representer(cls, cls.to_yaml)


class Serializable(yaml.YAMLObject):
    """
    Base class for any object that wants a nice clean yaml output
    """
    __metaclass__ = SerializableMetaclass
    def to_yaml(cls, dumper, data):

        return dumper.represent_yaml_object('!{0}'.format(data.__class__.__name__), data, cls,
                                            flow_style=cls.yaml_flow_style)
    to_yaml = classmethod(to_yaml)
