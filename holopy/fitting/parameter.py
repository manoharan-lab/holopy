# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
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
Classes for describing free parameters in fitting models

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <jfung@physics.harvard.edu>
"""



def Parameter(**kwargs):
    from holopy.core.utils import ensure_array
    from holopy.inference.prior import Uniform, Fixed
    """
    Deprecated. Use inference.prior objects instead
    """
    args = {'limit':None, 'guess':None, 'name':None}
    args.update(kwargs)
    
    limit = ensure_array(args['limit'])
    if len(limit) == 2:
        return Uniform(limit[0], limit[1], args['guess'], args['name'])
    else:
        return Fixed(args['guess'], args['name'])

def ComplexParameter(**kwargs):
    from holopy.inference.prior import ComplexPrior
    return ComplexPrior(kwargs)

