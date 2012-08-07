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
Misc utility functions to make coding more convenient

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""
from __future__ import division

import os
import shutil
import errno
from functools import wraps
import numpy as np
import inspect
import types
from decorator import decorator

def _ensure_array(x):
    if np.isscalar(x):
        return np.array([x])
    else:
        return np.array(x)

def _ensure_pair(x):
    if x is None:
        return None
    try:
        x[1]
        return np.array(x)
    except (IndexError, TypeError):
        return np.array([x, x])


def _preserve_holo_type(func):
    """
    Wraps a function that takes an array as its first argument, this
    will make sure if it gets a hologram, it returns a hologram
    """
    @wraps(func)
    def wrapper(*args, **kw):
        ret = func(*args, **kw)
        if (isinstance(args[0], holopy.hologram.Hologram) and 
            not isinstance(ret, holopy.hologram.Hologram)):
            return holopy.hologram.Hologram(ret, args[0].optics, 
                                            name=args[0].name)
        else:
            return ret
    return wrapper

    
def _mkdir_p(path):
    '''
    Equivalent to mkdir -p at the shell, this function makes a
    directory and its parents as needed, silently doing nothing if it
    exists.

    Mainly a convenience function for including in scripts that crunch
    through a lot of files.
    '''
    try:
        os.makedirs(path)
    except OSError as exc: 
        if exc.errno == errno.EEXIST:
            pass
        else: raise

def _copy_file(source, dest):
    '''
    Copies a file from source to dest.  Does nothing if source == dest

    Mainly a convenience function for including in scripts that crunch
    through a lot of files.
    '''
    try:
        return shutil.copy2(source, dest)
    except shutil.Error:
        pass

###############################################################################
# The code below allows a new pattern of classmethods which bind to created
# objects of the class or to a default instance.  
###############################################################################
# So if you have a class like this
#class Foo(object):
#    def __init__(name = 'joe'):
#        name = name
#        finish_binding(self)
#    @classmethod
#    @binding
#    def bar(self):
#        return self.name
#
# it will behave as
# >>> Foo.bar()
# joe
# Foo('bob').bar()
# bob
# The bar method will be given an default instance of the class if called as a
# classmethod, or will behave as a normal instance method if called on an
# instance
###############################################################################
    
def finish_binding(obj):
    """
    binds classmethods decorated with the binding decorator to obj

    This is intended for to be called within __init__ and will replace any
    @classmethod methods that are also decorated with @binding with instance methods
    """
    # use inspect to find all of the methods of the supplied class that have a
    # _bindme attribute (which was put there by the @binding decorator)
    fs = inspect.getmembers(obj, lambda x: inspect.ismethod(x) and
                            hasattr(x,'_bindme'))

    # overwrite the classmethod with an instance method referencing obj for each
    # method that wants bound
    for name, f in fs:
        setattr(obj, name, types.MethodType(f.undecorated, obj))

def binding(f, *args, **kw):
    r = decorator(_binding, f)
    r._bindme = True
    return r

        
def _binding(f, *args, **kw):
    if isinstance(args[0], type):
        args = (args[0](),)+args[1:]
    return f(*args, **kw)   

