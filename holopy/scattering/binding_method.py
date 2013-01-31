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
Allow mixed class/instance methods that operate either on a default object of a
class when called as a classmethod or on a supplied instance when called as an
instance method.  

So if you have a class like this
---
class Foo(object):
   def __init__(name = 'joe'):
       name = name
       finish_binding(self)
   @classmethod
   @binding
   def bar(self):
       return self.name
---
it will behave as

>>> Foo.bar()
joe
Foo('bob').bar()
bob

The bar method will be given a default instance of the class if called as a
classmethod or will behave as a normal instance method if called on an
instance

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""
from __future__ import division

import inspect
import types
try:
    from decorator import decorator
except ImportError:
    from .third_party.decorator import decorator # pragma: no cover
    

def finish_binding(obj):
    """
    binds classmethods decorated with the binding decorator to obj

    This is intended to be called within __init__ and will replace any
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
