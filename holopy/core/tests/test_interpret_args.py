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
import numpy as np
from numpy.testing import assert_allclose
from .common import get_example_data, assert_equal
from ..metadata import Optics, interpret_args
from ..marray import Schema

holo=get_example_data('image0001.yaml')
positions = holo.positions
optics = holo.optics
wavelen = optics.wavelen
index = optics.index
polarization = optics.polarization

wrong_optics=Optics(wavelen=1e-6,index=1.5,polarization=[0,1])
schema=Schema(positions=positions,optics=optics)
empty_schema=Schema(positions=positions)
wrong_schema=Schema(positions=positions,optics=wrong_optics)

def test_pass_schema():
    test_schema = interpret_args(schema=schema)
    assert_equal(schema, test_schema)

def test_pass_optics():
    test_schema = interpret_args(schema=empty_schema, optics=optics)
    assert_equal(schema, test_schema)

def test_ind_vars():
    test_schema = interpret_args(schema=empty_schema,wavelen=wavelen, index=index, polarization=polarization)
    assert_equal(schema, test_schema)

def test_optics_vs_schema():
    test_schema = interpret_args(schema=wrong_schema, optics=optics)
    assert_equal(schema, test_schema)

def test_ind_vars_vs_schema():
    test_schema = interpret_args(schema=wrong_schema, wavelen=wavelen, index=index, polarization=polarization)
    assert_equal(schema, test_schema)

def test_ind_vars_vs_optics():
    test_schema = interpret_args(schema=empty_schema, wavelen=wavelen, index=index, polarization=polarization, optics=wrong_optics)
    assert_equal(schema, test_schema)

def test_pass_image():
    test_holo = interpret_args(schema=holo)
    assert_equal(holo,test_holo)

def test_pass_positions():
    test_schema = interpret_args(schema=positions, wavelen=wavelen, index=index, polarization = polarization)
    assert_equal(schema, test_schema)


