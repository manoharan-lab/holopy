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
Reading and writing of yaml files.

yaml files are structured text files designed to be easy for humans to
read and write but also easy for computers to read.  HoloPy uses them
to store information about experimental conditions and to describe
analysis procedures.

.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.harvard.edu>
"""
from __future__ import division
import numpy as np
import yaml
from yaml.reader import ReaderError
import re
import inspect
import types

from holopy.core.helpers import OrderedDict
from holopy.core.holopy_object import SerializableMetaclass
from holopy.core.marray import Marray
from holopy.core import marray

def save(outf, obj):
    if isinstance(outf, basestring):
        outf = file(outf, 'wb')

    yaml.dump(obj, outf)
    if isinstance(obj, Marray):
        # yaml saves of large arrays are very slow, so we have numpy save the array
        # parts of Marray objects.  This will mean the file isn't stricktly
        # a valid yaml (or even a valid text file really), but we can still read
        # it, and with the right programs (like linux more) you can still see
        # the text yaml information, and it keeps everything in one file
        outf.write('array: !NpyBinary\n')
        np.save(outf, obj)


def load(inf):
    if isinstance(inf, basestring):
        inf = file(inf, mode = 'rU')

    line = inf.readline()
    cls = line.strip('{} !\n')
    lines = []
    if hasattr(marray, cls) and issubclass(getattr(marray, cls), Marray):
        while not re.search('!NpyBinary', line):
            lines.append(line)
            line = inf.readline()
        arr = np.load(inf)
        head = ''.join(lines[1:])
        kwargs = yaml.load(head)
        if kwargs is None:
            kwargs = {} #pragma: nocover
        return getattr(marray, cls)(arr, **kwargs)


    else:
        inf.seek(0)
        obj = yaml.load(inf)
        if isinstance(obj, dict):
            # sometimes yaml doesn't convert strings to floats properly, so we
            # have to check for that.
            for key in obj:
                if isinstance(obj[key], basestring):
                    try:
                        obj[key] = float(obj[key])
                    except ValueError: #pragma: nocover
                        pass #pragma: nocover

        return obj


###################################################################
# Custom Yaml Representers
###################################################################

def ignore_aliases(data):
    try:
        if data in [None, ()]:
            return True
        if isinstance(data, (str, unicode, bool, int, float)):
            return True
    except TypeError, e:
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
yaml.add_constructor(u'!class', class_loader)

def OrderedDict_representer(dumper, data):
    return dumper.represent_dict(data)
yaml.add_representer(OrderedDict, OrderedDict_representer)

def instancemethod_representer(dumper, data):
    func = data.im_func.func_name
    obj = data.im_self
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

def function_representer(dumper, data):
    code = inspect.getsource(data)
    code = code.split('\n',)
    # first line will be function name, we don't want that
    code = code[1].strip()
    return dumper.represent_scalar('!function', code)
# here I refer to function_representer.__class__ because I am not sure how else
# to access the type of a fuction (function does not work)
yaml.add_representer(function_representer.__class__, function_representer)

# for now punt if we attempt to read in functions.
# make_scatterer in model is allowed to be any function, so we may encounter
# them.  This constructor allows the read to succeed, but the function will be
# absent.
# It is possible to read in functions from the file, but it is more than a
# little subtle and kind of dangrous, so I want to think more about it before
# doing it - tgd 2012-06-4
def function_constructor(loader, node):
    return None
yaml.add_constructor('!function', function_constructor)
