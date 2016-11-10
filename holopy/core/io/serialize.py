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
Reading and writing of yaml files.

yaml files are structured text files designed to be easy for humans to
read and write but also easy for computers to read.  HoloPy uses them
to store information about experimental conditions and to describe
analysis procedures.

.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.harvard.edu>
"""

import numpy as np
import yaml
from yaml.reader import ReaderError
import inspect
import types

from ..utils import is_none
from ..holopy_object import SerializableMetaclass

def save(outf, obj):
    if isinstance(outf, str):
        outf = open(outf, 'wb')

    outf.write(yaml.dump(obj).encode())

def load(inf):
    if isinstance(inf, str):
        with open(inf, mode='rb') as inf:
            return yaml.load(inf)
    else:
        return yaml.load(inf)

def _pickle_method(method):
    func_name = method.__func__.__name__
    obj = method.__self__
    return _unpickle_method, (func_name, obj)

def _unpickle_method(func_name, obj):
    return getattr(obj, func_name)



import copyreg
import types
copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)

###################################################################
# Custom Yaml Representers
###################################################################

def ignore_aliases(data):
    try:
        # numpy arrays no longer want to be compared to None, so instead check for a none by looking for if it is an instance of NoneType
        if is_none(data) or data is ():
            return True
        if isinstance(data, (str, bool, int, float)):
            return True
    except TypeError as e:
        pass
yaml.representer.SafeRepresenter.ignore_aliases = \
    staticmethod(ignore_aliases)

# Represent 1d ndarrays as lists in yaml files because it makes them much
# prettier
def ndarray_representer(dumper, data):
    return dumper.represent_list(data.tolist())
yaml.add_representer(np.ndarray, ndarray_representer)

# represent tuples as lists because yaml doesn't have tuples
def tuple_representer(dumper, data):
    return dumper.represent_list(list(data))
yaml.add_representer(tuple, tuple_representer)

# represent numpy types as things that will print more cleanly
def complex_representer(dumper, data):
    return dumper.represent_scalar('!complex', repr(data.tolist()))
yaml.add_representer(np.complex128, complex_representer)
def complex_constructor(loader, node):
    return complex(node.value)
yaml.add_constructor('!complex', complex_constructor)

def numpy_float_representer(dumper, data):
    return dumper.represent_float(float(data))
yaml.add_representer(np.float64, numpy_float_representer)

def numpy_int_representer(dumper, data):
    return dumper.represent_int(int(data))
yaml.add_representer(np.int64, numpy_int_representer)
yaml.add_representer(np.int32, numpy_int_representer)

def numpy_dtype_representer(dumper, data):
    return dumper.represent_scalar('!dtype', data.name)
yaml.add_representer(np.dtype, numpy_dtype_representer)

def numpy_dtype_loader(loader, node):
    name = loader.construct_scalar(node)
    return np.dtype(name)
yaml.add_constructor('!dtype', numpy_dtype_loader)

def class_representer(dumper, data):
    return dumper.represent_scalar('!class', "{0}.{1}".format(data.__module__,
                                                              data.__name__))
yaml.add_representer(SerializableMetaclass, class_representer)

def class_loader(loader, node):
    name = loader.construct_scalar(node)
    tok = name.split('.')
    mod = __import__(tok[0])
    for t in tok[1:]:
        mod = mod.__getattribute__(t)
    return mod
yaml.add_constructor('!class', class_loader)

def instancemethod_representer(dumper, data):
    func = data.__func__.__name__
    obj = data.__self__
    if isinstance(obj, SerializableMetaclass):
        obj = obj()
    rep = yaml.dump(obj)
    # if the obj has arguments, we need to switch it to flow style so that it is
    # emitted properly
    tok = rep.split('\n')
    # TODO: this is a hack to get this working, I think it may cause bugs in
    # some corner cases
    if len(tok) > 2:
        rep = '{0} {{{1}}}'.format(tok[0], ', '.join(tok[1:]))
    return dumper.represent_scalar('!method', "{0} of {1}".format(func, rep))
yaml.add_representer(types.MethodType, instancemethod_representer)

def instancemethod_constructor(loader, node):
    name = loader.construct_scalar(node)
    tok = name.split('of')
    method = tok[0].strip()
    obj = 'dummy: '+ tok[1]
    obj = yaml.load(obj)['dummy']
    return getattr(obj, method)
yaml.add_constructor('!method', instancemethod_constructor)
