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
from nose.plugins.attrib import attr

from holopy.core.tests.common import assert_pickle_roundtrip
from holopy.scattering.theory import Mie

def assert_method_roundtrip(o):
    #assert_method_equal(o, pickle.loads(pickle.dumps(o)), 'pickled method')
    assert_method_equal(o, cPickle.loads(cPickle.dumps(o)), 'pickled method')

@attr("fast")
def test_pickle_mie_object():
    m = Mie()
    assert_pickle_roundtrip(m)
