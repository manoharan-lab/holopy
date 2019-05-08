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
"""
Base class for scattering theories.  Implements python-based
calc_intensity and calc_holo, based on subclass's calc_field

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""
import unittest

from nose.plugins.attrib import attr

from holopy.scattering import (
    Sphere, Spheres, Mie, Multisphere, Spheroid, Cylinder, Tmatrix)
from holopy.core import detector_grid
from holopy.core.tests.common import assert_obj_close
from holopy.scattering.calculations import *

scatterer = Sphere(n=1.6, r=.5, center=(5, 5, 5))
medium_index = 1.33
locations = detector_grid(shape=(20, 20), spacing=.1)
wavelen = 0.66
polarization=(0, 1)

@attr("fast")
def test_calc_holo():
    holo = calc_holo(locations, scatterer, medium_index, wavelen, polarization)


def test_calc_field():
    field = calc_field(locations, scatterer, medium_index, wavelen, polarization)


@attr("fast")
def test_calc_cross_section():
    cross = calc_cross_sections(scatterer, medium_index, wavelen, polarization)


def test_calc_intensity():
    intensity = calc_intensity(locations, scatterer, medium_index, wavelen, polarization)


@attr("fast")
def test_calc_scat_matrix():
    matr = calc_scat_matrix(locations, scatterer, medium_index, wavelen)


class TestDetermineTheory(unittest.TestCase):
    @attr("fast")
    def test_determine_theory_on_sphere(self):
        default_theory = determine_theory(Sphere())
        correct_theory = Mie()
        self.assertTrue(default_theory == correct_theory)

    @attr('fast')
    def test_determine_theory_on_spheres(self):
        default_theory = determine_theory(Spheres([Sphere(), Sphere()]))
        correct_theory = Multisphere()
        self.assertTrue(default_theory == correct_theory)

    @attr('fast')
    def test_determine_theory_on_spheroid(self):
        scatterer = Spheroid(n=1.33, r=(1.0, 2.0))
        default_theory = determine_theory(scatterer)
        correct_theory = Tmatrix()
        self.assertTrue(default_theory == correct_theory)

    @attr('fast')
    def test_determine_theory_on_cylinder(self):
        scatterer = Cylinder(n=1.33, h=2, d=1)
        default_theory = determine_theory(scatterer)
        correct_theory = Tmatrix()
        self.assertTrue(default_theory == correct_theory)


if __name__ == '__main__':
    unittest.main()

