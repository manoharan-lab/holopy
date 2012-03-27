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
Test fortran-based multilayered Mie calculations and python interface.  

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''

import sys
import os
hp_dir = (os.path.split(sys.path[0])[0]).rsplit(os.sep, 1)[0]
sys.path.append(hp_dir)
from nose.tools import with_setup
import holopy as hp
import scatterpy

from nose.plugins.attrib import attr

import common


from scatterpy.scatterer.coatedsphere import CoatedSphere
from scatterpy.theory import Mie


@attr('medium')
def test_Shell():
    s = CoatedSphere(center=[7.141442573813124, 7.160766866147957, 11.095409800342143],
              n=[(1.27121212428+0j), (1.49+0j)], r=[0.960957713253-0.0055,
                                                    0.960957713253]) 

    optics = hp.Optics(wavelen=0.658, index=1.36, polarization=[1.0, 0.0],
              pixel_scale=[0.071332999999999994, 0.071332999999999994])
    
    th = Mie(optics, 200)

    h = th.calc_holo(s, 0.4826042444701572)

    common.verify(h, 'shell')
