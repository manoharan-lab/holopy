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

import numpy as np
import holopy
import yaml
import os

def save(outf, obj):
    if isinstance(outf, basestring):
        outf = file(outf, 'w')
    yaml.dump(obj, outf)

def load(inf):
    if isinstance(inf, basestring):
        inf = file(inf)
    return yaml.load(inf)


# Represent 1d ndarrays as lists in yaml files because it makes them much
# prettier
def ndarray_representer(dumper, data):
    return dumper.represent_list(data.tolist())
#    if data.ndim == 1:
#        return dumper.represent_list([float(a) for a in data])
#    if data.ndim == 2:
        
#    else:
#        raise NotImplementedError

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
    return eval(node.value)
yaml.add_constructor('!complex', complex_constructor)
def numpy_float_representer(dumper, data):
    return dumper.represent_float(float(data))
yaml.add_representer(np.float64, numpy_float_representer)

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
    
    def to_yaml(cls, dumper, data):

        return dumper.represent_yaml_object('!{0}'.format(data.__class__.__name__), data, cls,
                                            flow_style=cls.yaml_flow_style)
    to_yaml = classmethod(to_yaml)




def load_yaml(filename):
    """
    Load a yaml config file

    Does a fair amount of preprocessing to ensure numbers are read
    correctly, arrays are read as numpy arrays and compute simple
    arithmetic expressions 

    Parameters
    ----------
    filename : string
        File to load

    Returns
    -------
    yaml : dict
       Dictionary with all key value pairs from the yaml file and with
       arithmetic expressions evaluated 

    Raises
    ------
    LoadError if it can't find the file
    """

    yf = yaml.load(open(filename))
       
    # yaml doesn't always return things as numbers, so make sure
    # everything that is a number in the file is treated as such
    def preprocess(yf):
        def token_process(tok):
            if isinstance(tok, basestring):
                if tok[0:1] == '~/':
                    return os.path.expanduser(tok)
                if tok[0] == '/':
                    return tok
                if tok.lower() == 'none':
                    return None
                elif tok.lower() == 'pi':
                    return np.pi
                try:
                    # Attempt some very simple calculator operations
                    if '/' in tok:
                        t1, t2 = tok.split('/')
                        return token_process(t1) / token_process(t2)
                    elif '*' in tok:
                        t1, t2 = tok.split('*')
                        return token_process(t1) * token_process(t2)
                    else:
                        return _numberize(tok)
                except:
                    # if arithmetic fails, just return our normal attempt at
                    # interpreting the string
                    return _numberize(tok)
            else:
                return tok
                    
        for k in yf:
            if isinstance(yf[k], dict):
                preprocess(yf[k])
            elif isinstance(yf[k], list) or isinstance(yf[k], tuple):
                for i in range(len(yf[k])):
                    yf[k][i] = token_process(yf[k][i])
                yf[k] = np.array(yf[k])
            else:
                yf[k] = token_process(yf[k])

        return yf

    return preprocess(yf)

def _numberize(string):
    '''Turns a string into a number

    if the string is an integer return that integer
    if the string is a float return that float
    else return the string
    '''
    try:
        return int(string)
    except ValueError:
        try:
            return float(string)
        except ValueError:
            return string

def _clean_for_yaml(data_dict):
    def process(d):
        for k in d:
            if isinstance(d[k], dict):
                process(d[k])
            else:
                if isinstance(d[k], np.ndarray):
                    d[k] = d[k].tolist()
        return d

    return process(data_dict)
