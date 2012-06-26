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
Reading and writing of yaml files.

yaml files are structured text files designed to be easy for humans to
read and write but also easy for computers to read.  Holopy uses them
to store information about experimental conditions and to describe
analysis procedures.

.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.harvard.edu>
"""
from __future__ import division
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
import numpy as np
import holopy as hp
import yaml
import os
import re
import os.path
import inspect
import scatterpy.io.serializable
from scatterpy.io.serializable import ordered_dump
from holopy.analyze.fit import FitResult, Model, Parameter, Nmpfit

class LoadError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return msg

def save(outf, obj):
    if isinstance(outf, basestring):
        outf = file(outf, 'w')
    yaml.dump(obj, outf)

def load(inf):
    if isinstance(inf, basestring):
        inf = file(inf)
    return yaml.load(inf)

def model_representer(dumper, data):
    dump_dict = OrderedDict()
    if data._user_make_scatterer is None:
        if data.alpha_par:
            dump_dict['parameters'] = [data.scatterer, data.alpha_par]
        else:
            dump_dict['parameters'] = data.scatterer

        dump_dict['theory'] = data.theory
    else:
        dump_dict['parameters'] = data.parameters
        dump_dict['theory'] = data.theory
        dump_dict['make_scatterer'] = data._user_make_scatterer
        
    dump_dict['selection'] = data.selection
    return ordered_dump(dumper, '!Model', dump_dict)
yaml.add_representer(Model, model_representer)

# legacy loader, this is only here because for a while we saved things as
# !Minimizer {algorithm = nmpfit} and we still want to be able to read those yamls
def minimizer_constructor(loader, node):
    data = loader.construct_mapping(node, deep=True)
    if data['algorithm'] == 'nmpfit':
        return Nmpfit()
    else:
        raise LoadError('Could not load Minimizer with: {0}'.format(data))
yaml.add_constructor("!Minimizer", minimizer_constructor)

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


