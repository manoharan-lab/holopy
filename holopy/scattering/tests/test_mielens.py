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
Test the ability of the mielens calculations to interface with holopy.

.. moduleauthor:: Brian D. Leahy <bleahy@seas.harvard.edu>
"""

import os
import unittest

import numpy as np
from nose.plugins.attrib import attr

from holopy.core.metadata import detector_grid
from holopy.scattering.theory import Mie, MieLens, mielensfunctions
from holopy.scattering.scatterer import Sphere, Spheres
from holopy.scattering.interface import calc_holo

from holopy.scattering.tests.common import (
    sphere, xschema, scaling_alpha, yschema, xpolarization, ypolarization,
    x, y, z, n, radius, wavelen, index)


TOLS = {'atol': 1e-13, 'rtol': 1e-13}
MEDTOLS = {"atol": 1e-6, "rtol": 1e-6}
SOFTTOLS = {"atol": 1e-3, "rtol": 1e-3}


class TestMieLens(unittest.TestCase):
    @attr("medium")
    def test_does_not_crash(self):
        theory = MieLens()
        holo = calc_holo(xschema, sphere, index, wavelen, xpolarization,
                         theory=theory)
        self.assertTrue(holo is not None)

    @attr("fast")
    def test_raises_error_if_multiple_z_values(self):
        theory = MieLens()
        np.random.seed(10)
        positions = np.random.randn(3, 10)  # the zs will differ by chance
        self.assertRaises(
            ValueError, theory._raw_fields, positions, sphere, 1.0, 1.33,
            xschema.illum_polarization)

    @attr("fast")
    def test_desired_coordinate_system_is_cylindrical(self):
        self.assertTrue(MieLens.desired_coordinate_system == 'cylindrical')

    @attr("medium")
    def test_can_calculate_for_positive_and_negative_z(self):
        theory = MieLens()
        ka = 5.0
        radius = ka * wavelen * 0.5 / np.pi
        sphere_index = 1.59
        is_ok = []
        for kz in [-50., 0., 50.]:
            center = (0, 0, kz * wavelen * 0.5 / np.pi)
            this_sphere = Sphere(n=sphere_index, r=radius, center=center)
            holo = calc_holo(xschema, this_sphere, index, wavelen,
                             xpolarization, theory=theory)
            is_ok.append(holo.data.ptp() > 0)
        self.assertTrue(all(is_ok))

    @attr("fast")
    def test_holopy_hologram_equal_to_exact_calculation(self):
        # Checks that phase shifts and wrappers for mielens are correct
        theory_mielens = MieLens()
        illum_wavelength = 0.66  # 660 nm red light
        k = 2 * np.pi / illum_wavelength
        center = (10, 10, 5.)

        kwargs = {'particle_kz': center[2] * k,
                  'index_ratio': 1.2,
                  'size_parameter': 0.5 * k,
                  'lens_angle': theory_mielens.lens_angle}
        detector = detector_grid(10, 2.0)
        x = detector.x.values.reshape(-1, 1) - center[0]
        y = detector.y.values.reshape(1, -1) - center[1]

        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        calculator = mielensfunctions.MieLensCalculator(**kwargs)
        scatterer = Sphere(n=kwargs['index_ratio'],
                           r=kwargs['size_parameter'] / k,
                           center=center)

        holo_calculate = calculator.calculate_total_intensity(k * rho, phi)
        holo_holopy = calc_holo(
            detector, scatterer, illum_wavelen=illum_wavelength,
            medium_index=1., illum_polarization=(1, 0), theory=theory_mielens)

        is_ok = np.allclose(holo_calculate, holo_holopy.values.squeeze(),
                            **TOLS)
        self.assertTrue(is_ok)

    @attr("medium")
    def test_central_lobe_is_bright_when_particle_is_above_focus(self):
        # This test only works at low index contrast, when the scattered
        # beam is everywhere weaker than the unscattered:
        zs = np.linspace(2, 10, 11)
        central_lobes = calculate_central_lobe_at(zs)
        self.assertTrue(np.all(central_lobes > 1))

    @attr("medium")
    def test_central_lobe_is_dark_when_particle_is_below_focus(self):
        # This test only works at low index contrast, when the scattered
        # beam is everywhere weaker than the unscattered:
        zs = np.linspace(-2, -10, 11)
        central_lobes = calculate_central_lobe_at(zs)
        self.assertTrue(np.all(central_lobes < 1))

    @attr('medium')
    def test_mielens_is_close_to_mieonly(self):
        """Tests that a mielens hologram is similar to a mie-only hologram."""
        theory_mielens = MieLens()
        theory_mieonly = Mie()

        holo_mielens = calc_holo(
            xschema, sphere, index, wavelen, xpolarization,
            theory=theory_mielens)
        holo_mieonly = calc_holo(
            xschema, sphere, index, wavelen, xpolarization,
            scaling=1.0, theory=theory_mieonly)

        # the two arrays should not be equal
        self.assertFalse(np.allclose(holo_mielens, holo_mieonly, **TOLS))

        # but their max and min values should be close:
        ptp_close_ish = np.isclose(
            holo_mielens.values.ptp(), holo_mieonly.values.ptp(), atol=0.1)
        # and their median should be close:
        median_close_ish = np.isclose(
            np.median(holo_mielens), np.median(holo_mieonly), atol=0.1)

        self.assertTrue(ptp_close_ish)
        self.assertTrue(median_close_ish)

    @attr("medium")
    def test_mielens_x_polarization_differs_from_y(self):
        # test holograms for orthogonal polarizations; make sure they're
        # not the same, nor too different from one another.
        theory = MieLens()
        holo_x = calc_holo(xschema, sphere, index, wavelen,
                           illum_polarization=xpolarization, theory=theory)
        holo_y = calc_holo(yschema, sphere, index, wavelen,
                           illum_polarization=ypolarization, theory=theory)

        # the two arrays should not be equal
        self.assertFalse(np.allclose(holo_x, holo_y, **SOFTTOLS))

        # but their max and min values should be very close
        # (really exact because we lose no symmetry from the grid)
        self.assertTrue(np.isclose(holo_x.max(), holo_y.max(), **MEDTOLS))
        self.assertTrue(np.isclose(holo_x.min(), holo_y.min(), **MEDTOLS))

    @attr('medium')
    def test_mielens_multiple_returns_nonzero(self):
        scatterers = [
            Sphere(n=1.59, r=5e-7, center=(1e-6, -1e-6, 10e-6)),
            Sphere(n=1.59, r=1e-6, center=[8e-6, 5e-6, 5e-6]),
            Sphere(n=1.59 + 0.0001j, r=5e-7, center=[5e-6, 10e-6, 3e-6]),
            ]
        sphere_collection = Spheres(scatterers=scatterers)
        theory = MieLens()

        schema = yschema
        holo = calc_holo(schema, sphere_collection, index, wavelen,
                         theory=theory)
        self.assertTrue(holo is not None)
        self.assertTrue(holo.values.std() > 0)


def calculate_central_lobe_at(zs):
    illum_wavelength = 0.66  # 660 nm red light
    k = 2 * np.pi / illum_wavelength
    detector = detector_grid(4, 2.0)

    central_lobes = []
    for z in zs:
        center = (0, 0, z)
        scatterer = Sphere(n=1.59, r=0.5, center=center)
        holo = calc_holo(
            detector, scatterer, illum_wavelen=illum_wavelength,
            medium_index=1.33, illum_polarization=(1, 0), theory=MieLens())
        central_lobe = holo.values.squeeze()[0, 0]
        central_lobes.append(central_lobe)
    return np.array(central_lobes)


if __name__ == '__main__':
    unittest.main()
