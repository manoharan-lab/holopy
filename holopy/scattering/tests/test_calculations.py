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

import xarray as xr

SCATTERER = Sphere(n=1.6, r=.5, center=(5, 5, 5))
MED_INDEX = 1.33
LOCATIONS = detector_grid(shape=(20, 20), spacing=.1)
WAVELEN = 0.66
POL=(0, 1)


class TestCalculations(unittest.TestCase):
    @attr("fast")
    def test_calc_holo(self):
        holo = calc_holo(LOCATIONS, SCATTERER, MED_INDEX, WAVELEN, POL)
        self.assertTrue(True)

    def test_calc_field(self):
        field = calc_field(LOCATIONS, SCATTERER, MED_INDEX, WAVELEN, POL)
        self.assertTrue(True)

    @attr("fast")
    def test_calc_cross_sections(self):
        cross = calc_cross_sections(SCATTERER, MED_INDEX, WAVELEN, POL)
        self.assertTrue(True)

    def test_calc_intensity(self):
        intensity = calc_intensity(LOCATIONS, SCATTERER, MED_INDEX, WAVELEN, POL)
        self.assertTrue(True)

    @attr("fast")
    def test_calc_scat_matrix(self):
        matr = calc_scat_matrix(LOCATIONS, SCATTERER, MED_INDEX, WAVELEN)
        self.assertTrue(True)

    def test_finalize(self):
        detector = finalize(LOCATIONS.values, LOCATIONS)
        self.assertTrue(True)

    def test_scattered_field_to_hologram(self):
        size = 3
        coords = np.linspace(0, 1, size)
        scat = xr.DataArray(np.array([0.5, 0, 0]), coords=[('vector', coords)])
        ref = xr.DataArray(np.array([0.5, 0, 0]), coords=[('vector', coords)])
        normals = np.array((0, 0, 1))
        holo = scattered_field_to_hologram(scat, ref, normals)
        self.assertEquals(holo.values.mean(), 1.)

class TestDetermineDefaultTheoryFor(unittest.TestCase):
    @attr("fast")
    def test_determine_default_theory_for_sphere(self):
        default_theory = determine_default_theory_for(Sphere())
        correct_theory = Mie()
        self.assertTrue(default_theory == correct_theory)

    @attr('fast')
    def test_determine_default_theory_for_spheres(self):
        default_theory = determine_default_theory_for(
            Spheres([Sphere(), Sphere()]))
        correct_theory = Multisphere()
        self.assertTrue(default_theory == correct_theory)

    @attr('fast')
    def test_determine_default_theory_for_spheroid(self):
        scatterer = Spheroid(n=1.33, r=(1.0, 2.0))
        default_theory = determine_default_theory_for(scatterer)
        correct_theory = Tmatrix()
        self.assertTrue(default_theory == correct_theory)

    @attr('fast')
    def test_determine_default_theory_for_cylinder(self):
        scatterer = Cylinder(n=1.33, h=2, d=1)
        default_theory = determine_default_theory_for(scatterer)
        correct_theory = Tmatrix()
        self.assertTrue(default_theory == correct_theory)


class TestPrepSchema(unittest.TestCase):
    pass


class TestInterpretTheory(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
