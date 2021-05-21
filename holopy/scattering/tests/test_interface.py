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
.. moduleauthor:: R. Alexander <ralexander@g.harvard.edu>
"""
import unittest
import warnings

from nose.plugins.attrib import attr

from holopy.scattering import (Sphere, Spheres, Mie, Multisphere,
                               Spheroid, Cylinder, Tmatrix)
from holopy.core import detector_grid
from holopy.core.tests.common import assert_obj_close
from holopy.scattering.interface import *
from holopy.scattering.errors import MissingParameter, InvalidScatterer
from holopy.inference import prior

import xarray as xr

SCATTERER = Sphere(n=1.6, r=.5, center=(5, 5, 5))
MED_INDEX = 1.33
LOCATIONS = detector_grid(shape=(2, 2), spacing=1)
WAVELEN = 0.66
POL = (0, 1)


class TestInterface(unittest.TestCase):
    @attr('fast')
    def test_calc_holo(self):
        # FIXME: Test results change when 'auto' theory for SCATTERER changes
        result = calc_holo(LOCATIONS, SCATTERER, MED_INDEX, WAVELEN, POL)
        expected = np.array([[1.03670094, 1.05260144], [1.04521558, 1.01477807]])
        self.assertTrue(np.allclose(result.values.squeeze(), expected))

    @attr('medium')
    def test_calc_field(self):
        # FIXME: Test results change when 'auto' theory for SCATTERER changes
        result = calc_field(LOCATIONS, SCATTERER, MED_INDEX, WAVELEN, POL)
        expected = np.array([[[-.00673848-.00213145j, -.0080032 +.00183085j],
                              [-.0080032 +.00183085j, -.00310756+.00865182j]],
                             [[ .0181347 +.00726132j,  .02591295-.00605441j],
                              [ .02231151-.00523052j,  .00689569-.02923405j]],
                             [[ .01184803+.00651537j,  .01382257-.00236663j],
                              [ .01727821-.00295829j,  .00384075-.01752032j]]])
        self.assertTrue(np.allclose(result.values.squeeze(), expected))

    @attr('fast')
    def test_calc_cross_sections(self):
        # FIXME: Test results change when 'auto' theory for SCATTERER changes
        result = calc_cross_sections(SCATTERER, MED_INDEX, WAVELEN, POL)
        expected = np.array([2.04017098e+00, -4.44089210e-16,
                             2.04017098e+00, 9.13750771e-01])
        self.assertTrue(np.allclose(result.values.squeeze(), expected))

    @attr('medium')
    def test_calc_intensity(self):
        # FIXME: Test results change when 'auto' theory for SCATTERER changes
        result = calc_intensity(LOCATIONS, SCATTERER, MED_INDEX, WAVELEN, POL)
        expected = np.array([[0.00043154, 0.00077554],
                             [0.00059256, 0.00098669]])
        self.assertTrue(np.allclose(result.values.squeeze(), expected))

    @attr('fast')
    def test_calc_scat_matrix(self):
        # FIXME: Test results change when 'auto' theory for SCATTERER changes 
        result = calc_scat_matrix(LOCATIONS, SCATTERER, MED_INDEX, WAVELEN)
        expected = np.array([[[[-2.3818862 +1.10607989j, -2.41362056+1.74943249j],
                               [-2.41362056+1.74943249j, -2.15238106+2.50562808j]],
                              [[ 0.        +0.j        ,  0.        +0.j        ],
                               [ 0.        +0.j        ,  0.        +0.j        ]]],
                             [[[ 0.        +0.j        ,  0.        +0.j        ],
                               [ 0.        +0.j        ,  0.        +0.j        ]],
                              [[-2.55065057+1.60766597j, -2.74726197+2.19673594j],
                               [-2.74726197+2.19673594j, -2.66462981+2.81492861j]]]])
        self.assertTrue(np.allclose(result.values.squeeze(), expected))

    @attr('fast')
    def test_finalize(self):
        result = finalize(LOCATIONS.values, LOCATIONS)
        expected = copy_metadata(LOCATIONS.values, LOCATIONS)
        self.assertTrue(result.equals(expected))

    @attr('medium')
    def test_scattered_field_to_hologram(self):
        coords = ['x', 'y', 'z']
        scat = xr.DataArray(np.array([1, 0, 0]), coords=[('vector', coords)])
        ref = xr.DataArray(np.array([1, 0, 0]), coords=[('vector', coords)])
        correct_holo = (np.abs(scat + ref)**2).sum(dim='vector')
        holo = scattered_field_to_hologram(scat, ref)
        self.assertEqual(holo.values.mean(), correct_holo.values.mean())


class TestDetermineDefaultTheoryFor(unittest.TestCase):
    @attr('fast')
    def test_determine_default_theory_for_sphere(self):
        default_theory = determine_default_theory_for(Sphere())
        correct_theory = Mie()
        self.assertEqual(default_theory, correct_theory)

    @attr('fast')
    def test_determine_default_theory_for_spheres(self):
        default_theory = determine_default_theory_for(
            Spheres([Sphere(center=(1, 1, 1)), Sphere(center=(1, 1, 2))]))
        correct_theory = Multisphere()
        self.assertEqual(default_theory, correct_theory)

    @attr('fast')
    def test_determine_default_theory_for_spheroid(self):
        scatterer = Spheroid(n=1.33, r=(1.0, 2.0))
        default_theory = determine_default_theory_for(scatterer)
        correct_theory = Tmatrix()
        self.assertEqual(default_theory, correct_theory)

    @attr('fast')
    def test_determine_default_theory_for_cylinder(self):
        scatterer = Cylinder(n=1.33, h=2, d=1)
        default_theory = determine_default_theory_for(scatterer)
        correct_theory = Tmatrix()
        self.assertEqual(default_theory, correct_theory)

    @attr('fast')
    def test_determine_default_theory_for_layered_spheres(self):
        layered_spheres = Spheres([
            Sphere(center=(1, 1, 1), r=[0.5, 1], n=[1, 1.5]),
            Sphere(center=(3, 2, 2), r=[0.5, 1], n=[1, 1.5])])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            default_theory = determine_default_theory_for(layered_spheres)
        correct_theory = Mie()
        self.assertEqual(default_theory, correct_theory)

    @attr('fast')
    def test_gives_mie_when_spheres_very_far_apart(self):
        sphere1 = Sphere(r=0.5, n=1.59, center=(100, 0, 10))
        sphere2 = Sphere(r=0.5, n=1.59, center=(-100, 0, 10))
        spheres = Spheres([sphere1, sphere2])
        default_theory = determine_default_theory_for(spheres)
        correct_theory = Mie()
        self.assertEqual(default_theory, correct_theory)

    @attr('fast')
    def test_raises_invalid_scatterer_when_center_not_set_for_spheres(self):
        spheres = Spheres([Sphere(), Sphere()])
        self.assertRaises(
            InvalidScatterer,
            determine_default_theory_for,
            spheres)


class TestPrepSchema(unittest.TestCase):
    @attr('fast')
    def test_wavelength_missing(self):
        args = (LOCATIONS, MED_INDEX, None, POL)
        self.assertRaises(MissingParameter, prep_schema, *args)

    @attr('fast')
    def test_medium_index_missing(self):
        args = (LOCATIONS, None, WAVELEN, POL)
        self.assertRaises(MissingParameter, prep_schema, *args)

    @attr('fast')
    def test_polarization_missing(self):
        args = (LOCATIONS, MED_INDEX, WAVELEN, None)
        self.assertRaises(MissingParameter, prep_schema, *args)

    @attr('fast')
    def test_multiple_illumination_via_polarization_shape(self):
        coords = ['red', 'green']
        polarization = xr.DataArray(np.array([[1, 0], [0, 1]]),
                                    coords=[('illumination', coords),
                                            ('vector', ['x', 'y'])])
        detector = prep_schema(LOCATIONS, MED_INDEX, WAVELEN, polarization)
        self.assertTrue(len(detector.illum_wavelen == 2))

    @attr('fast')
    def test_multiple_illumination_via_detector_wavelength_shape(self):
        coords = ['red', 'green']
        wavelength = xr.DataArray(np.array([0.66, 0.532]),
                                  coords=[('illumination', coords)])
        detector = prep_schema(LOCATIONS, MED_INDEX, wavelength, POL)
        self.assertTrue(len(detector.illum_polarization) == 2)


class TestInterpretTheory(unittest.TestCase):
    @attr('fast')
    def test_interpret_auto_theory(self):
        theory = interpret_theory(SCATTERER, theory='auto')
        theory_ok = type(theory) == Mie
        self.assertTrue(theory_ok)

    @attr('fast')
    def test_interpret_specified_theory(self):
        theory = interpret_theory(SCATTERER, theory=Mie)
        theory_ok = type(theory) == Mie
        self.assertTrue(theory_ok)


class TestValidateScatterer(unittest.TestCase):
    @attr('fast')
    def test_initial_guess_if_prior_in_scatterer(self):
        r = prior.Uniform(0.5, 0.6, 0.59)
        n = prior.Gaussian(1.5, 0.2)
        scatterer = Sphere(r, n, center=[5, 5, 5])
        best_guess = Sphere(r.guess, n.guess, scatterer.center)
        self.assertEqual(validate_scatterer(scatterer), best_guess)


if __name__ == '__main__':
    unittest.main()
