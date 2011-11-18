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
Test file IO of scatterpy objects

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''

import scatterpy
import holopy as hp
import tempfile
from numpy.testing import assert_, assert_allclose, assert_equal

def assert_obj_equal(o1, o2):
    d1, d2 = o1.__dict__, o2.__dict__
    assert_equal(sorted(d1.keys()), sorted(d2.keys()))
    for key, val in d1.iteritems():
        if isinstance(val, hp.Optics):
            assert_obj_equal(val, d2[key])
        else:
            try:
                assert_equal(val, d2[key])
            except ValueError:
                assert_allclose(val, d2[key])
            
def test_theory_io():
    t = scatterpy.theory.Multisphere(hp.Optics(wavelen=.66, index=1.33,
                                               pixel_scale=.1))
    tempf = tempfile.TemporaryFile()
    scatterpy.io.save(tempf, t)
    tempf.flush()
    tempf.seek(0)
    loaded = scatterpy.io.load(tempf)
    assert_obj_equal(t, loaded)

def test_scatterer_io():
    s = scatterpy.Sphere()
    tempf = tempfile.TemporaryFile()
    scatterpy.io.save(tempf, s)
    tempf.flush()
    tempf.seek(0)
    loaded = scatterpy.io.load(tempf)
    assert_obj_equal(s, loaded)
