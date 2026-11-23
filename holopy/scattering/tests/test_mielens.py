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

import numpy as np
from numpy.testing import assert_allclose, assert_array_less, assert_equal
import xarray as xr

import pytest

from holopy.core.metadata import detector_grid
from holopy.scattering.theory import (
    Mie, MieLens, AberratedMieLens, mielensfunctions
)
from holopy.scattering.scatterer import Sphere, Spheres
from holopy.scattering.interface import calc_holo

from holopy.scattering.tests.common import (
    sphere, xschema, yschema, xpolarization, ypolarization,
    wavelen, index
)


TOLS = {'atol': 1e-13, 'rtol': 1e-13}
MEDTOLS = {"atol": 1e-6, "rtol": 1e-6}


class TestMieLens:
    @pytest.mark.fast
    def test_raises_error_if_multiple_z_values(self):
        theory = MieLens()
        np.random.seed(10)
        positions = np.random.randn(3, 10)  # the zs will differ by chance
        with pytest.raises(ValueError):
            theory.raw_fields(positions, sphere, 1.0, 1.33,
                              xschema.illum_polarization)

    @pytest.mark.fast
    def test_desired_coordinate_system_is_cylindrical(self):
        assert MieLens.desired_coordinate_system == 'cylindrical'

    @pytest.mark.fast
    def test_create_calculator(self):
        lens_angle = 0.948
        kwargs = {
            "particle_kz": 10.0,
            "index_ratio": 1.4,
            "size_parameter": 5.0,
            }

        theory = MieLens(lens_angle=lens_angle)
        calculator = theory._create_calculator(**kwargs)
        kwargs.update({"lens_angle": lens_angle})

        assert isinstance(calculator, mielensfunctions.MieLensCalculator)
        for key, value in kwargs.items():
            assert getattr(calculator, key) == value

    @pytest.mark.medium
    @pytest.mark.parametrize("kz", [-50., 0., 50.])
    def test_can_calculate_for_positive_and_negative_z(self, kz):
        theory = MieLens()
        ka = 5.0
        radius = ka * wavelen * 0.5 / np.pi
        center = (0, 0, kz * wavelen * 0.5 / np.pi)
        test_sphere = Sphere(n=1.59, r=radius, center=center)
        holo = calc_holo(xschema, test_sphere, index, wavelen, xpolarization,
                         theory=theory)
        assert np.ptp(holo.data) > 0

    @pytest.mark.fast
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

        assert_allclose(holo_calculate, holo_holopy.values.squeeze(), **TOLS)

    @pytest.mark.medium
    def test_central_lobe_is_bright_when_particle_is_above_focus(self):
        # This test only works at low index contrast, when the scattered
        # beam is everywhere weaker than the unscattered:
        zs = np.linspace(2, 10, 11)
        central_lobes = calculate_central_lobe_at(zs)
        assert_array_less(1.0, central_lobes)

    @pytest.mark.medium
    def test_central_lobe_is_dark_when_particle_is_below_focus(self):
        # This test only works at low index contrast, when the scattered
        # beam is everywhere weaker than the unscattered:
        zs = np.linspace(-2, -10, 11)
        central_lobes = calculate_central_lobe_at(zs)
        assert_array_less(central_lobes, 1.0)

    @pytest.mark.medium
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
        assert_array_less(1e-3, np.abs(holo_mielens - holo_mieonly).max())

        # but their max and min values should be close:
        assert_allclose(np.ptp(holo_mielens.values),
                        np.ptp(holo_mieonly.values), atol=0.1)
        # and their median should be close:
        assert_allclose(np.median(holo_mielens),
                        np.median(holo_mieonly), atol=0.1)

    @pytest.mark.medium
    def test_mielens_matches_mieonly_far_from_focus(self):
        """Tests that MieLens and Mie produce nearly the same hologram when the
        particle is far from the focal plane.

        """
        # use detector with fine spacing but large area to subtend an angle
        # smaller than the lens angle but still large enough to test for errors
        # in the MieLens calculations
        pixels = 128
        spacing = 0.3
        detector = detector_grid(pixels, spacing)
        # 1 micrometer PS sphere 40 micrometers above the detector center
        sphere_center = (pixels*spacing/2, pixels*spacing/2, 40)
        test_sphere = Sphere(n=1.59, r=0.5, center=sphere_center)
        holo_mielens = calc_holo(detector, test_sphere, 1.33, 0.66, (1, 0),
                                 theory=MieLens(lens_angle=1.0))
        holo_mieonly = calc_holo(detector, test_sphere, 1.33, 0.66, (1, 0),
                                 theory=Mie(), scaling=1.0)
        assert_allclose(holo_mielens, holo_mieonly, atol=1e-2)

    @pytest.mark.medium
    def test_mielens_x_polarization_differs_from_y(self):
        # test holograms for orthogonal polarizations; make sure they're
        # not the same, nor too different from one another.
        theory = MieLens()
        holo_x = calc_holo(xschema, sphere, index, wavelen,
                           illum_polarization=xpolarization, theory=theory)
        holo_y = calc_holo(yschema, sphere, index, wavelen,
                           illum_polarization=ypolarization, theory=theory)

        # the two arrays should not be equal
        assert_array_less(1e-3, np.abs(holo_x - holo_y).max())

        # but their max and min values should be very close
        # (really exact because we lose no symmetry from the grid)
        assert_allclose(holo_x.max(), holo_y.max(), **MEDTOLS)
        assert_allclose(holo_x.min(), holo_y.min(), **MEDTOLS)

    @pytest.mark.medium
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
        assert holo.values.std() > 0

    @pytest.mark.fast
    def test_transforms_correctly_with_polarization_rotation(self):
        # We test that rotating the lab frame correctly rotates
        # the polarization.
        # If we rotate (x0, y0) -> (y1, -x1), then the polarization
        # in the new coordinates should be
        # E1x = E0y, E1y = -E1x
        scatterer = sphere
        medium_wavevec = 2 * np.pi / wavelen
        medium_index = index
        theory = MieLens()

        krho = np.linspace(0, 100, 11)
        phi_0 = 0 * krho + np.pi / 180.0  # a small component along y
        phi_1 = phi_0 - np.pi / 2
        kz = np.full_like(krho, 20.0)

        pol_0 = xr.DataArray([1.0, 0, 0])
        pos_0 = np.array([krho, phi_0, kz])

        pol_1 = xr.DataArray([0, -1.0, 0])
        pos_1 = np.array([krho, phi_1, kz])

        args = (scatterer, medium_wavevec, medium_index)

        fields_0 = theory.raw_fields(pos_0, *args, pol_0)
        fields_1 = theory.raw_fields(pos_1, *args, pol_1)

        assert_allclose(fields_1[0],  fields_0[1], **TOLS)
        assert_allclose(fields_1[1], -fields_0[0], **TOLS)

    @pytest.mark.fast
    def test_parameters_returns_correct_keys_and_values(self):
        np.random.seed(1707)
        lens_angle = np.random.rand()
        theory = MieLens(lens_angle=lens_angle)

        correct = {'lens_angle': lens_angle}
        assert theory.parameters == correct

    @pytest.mark.fast
    def test_from_parameters_correctly_sets_parameters(self):
        np.random.seed(1709)
        lens_angle = np.random.rand()
        parameters = {'lens_angle': lens_angle}

        theory = MieLens().from_parameters(parameters)
        assert isinstance(theory, MieLens)
        assert theory.lens_angle == lens_angle

    @pytest.mark.fast
    def test_from_parameters_leaves_original_theory_alone(self):
        np.random.seed(1709)
        lens_angle_original = np.random.rand()
        theory = MieLens(lens_angle=lens_angle_original)

        lens_angle_new = np.random.rand()
        parameters = {'lens_angle': lens_angle_new}
        _ = theory.from_parameters(parameters)

        assert theory.lens_angle == lens_angle_original

    @pytest.mark.fast
    def test_theory_from_parameters_respects_nonfittable_options(self):
        pars = {'lens_angle': 0.6}
        # Since the theory doesn't actually construct a calculator until
        # the hologram is generated, we can pass in nonsense calculator
        # accuracy kwargs
        correct = {
            'some': 123,
            'additional': True,
            'kwargs': 42,
            'structure': 'check',
            }
        theory_in = MieLens(
            lens_angle=1.0,
            calculator_accuracy_kwargs=correct)
        theory_out = theory_in.from_parameters(pars)
        assert theory_out.calculator_accuracy_kwargs == correct


class TestAberratedMieLens:
    @pytest.mark.fast
    def test_init_stores_params(self):
        np.random.seed(1007)
        kwargs = {
            'lens_angle': np.random.rand(),
            'spherical_aberration': np.random.randn(),
            }
        theory = AberratedMieLens(**kwargs)
        for key, value in kwargs.items():
            assert hasattr(theory, key)
            assert getattr(theory, key) == value

    @pytest.mark.fast
    def test_create_calculator(self):
        np.random.seed(1011)
        init_kwargs = {
            'lens_angle': np.random.rand(),
            'spherical_aberration': np.random.randn(),
            }
        calc_kwargs = {
            "particle_kz": 10.0,
            "index_ratio": 1.4,
            "size_parameter": 5.0,
            }
        theory = AberratedMieLens(**init_kwargs)
        calculator = theory._create_calculator(**calc_kwargs)

        assert isinstance(calculator, mielensfunctions.MieLensCalculator)
        for kwargs in [init_kwargs, calc_kwargs]:
            for key, value in kwargs.items():
                assert getattr(calculator, key) == value

    @pytest.mark.fast
    def test_holopy_hologram_equal_to_exact_calculation(self):
        np.random.seed(1023)
        theory_kwargs = {
            'lens_angle': np.random.rand(),
            'spherical_aberration': 5 * np.random.randn()}
        theory_aberrated = AberratedMieLens(**theory_kwargs)

        illum_wavelength = 0.66  # 660 nm red light
        k = 2 * np.pi / illum_wavelength
        center = (10, 10, 5.)
        calculator_kwargs = theory_kwargs.copy()
        calculator_kwargs.update({
            'particle_kz': center[2] * k,
            'index_ratio': 1.2,
            'size_parameter': 0.5 * k,
            })
        detector = detector_grid(10, 2.0)
        x = detector.x.values.reshape(-1, 1) - center[0]
        y = detector.y.values.reshape(1, -1) - center[1]

        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        calculator = mielensfunctions.AberratedMieLensCalculator(
            **calculator_kwargs)
        scatterer = Sphere(n=calculator_kwargs['index_ratio'],
                           r=calculator_kwargs['size_parameter'] / k,
                           center=center)

        holo_calculate = calculator.calculate_total_intensity(k * rho, phi)
        holo_holopy = calc_holo(
            detector, scatterer, illum_wavelen=illum_wavelength,
            medium_index=1., illum_polarization=(1, 0),
            theory=theory_aberrated)

        assert_allclose(holo_calculate, holo_holopy.values.squeeze(), **TOLS)

    @pytest.mark.medium
    def test_aberratedmielens_is_same_as_mielens_when_unaberrated(self):
        lens_angle = 0.8
        theory_unaberrated = MieLens(lens_angle=lens_angle)
        theory_aberrated = AberratedMieLens(
            lens_angle=lens_angle, spherical_aberration=0)

        holo_unaberrated = calc_holo(
            xschema, sphere, index, wavelen, xpolarization,
            theory=theory_unaberrated)
        holo_aberrated = calc_holo(
            xschema, sphere, index, wavelen, xpolarization,
            theory=theory_aberrated)

        assert_allclose(holo_unaberrated, holo_aberrated, **TOLS)

    @pytest.mark.medium
    def test_aberratedmielens_differs_from_mielens_when_aberrated(self):
        # use a large lens angle to change hologram significantly
        lens_angle = 0.8
        theory_unaberrated = MieLens(lens_angle=lens_angle)
        theory_aberrated = AberratedMieLens(
            lens_angle=lens_angle, spherical_aberration=1.0)

        holo_unaberrated = calc_holo(
            xschema, sphere, index, wavelen, xpolarization,
            theory=theory_unaberrated)
        holo_aberrated = calc_holo(
            xschema, sphere, index, wavelen, xpolarization,
            theory=theory_aberrated)

        assert_array_less(1e-3,
                          np.abs(holo_unaberrated - holo_aberrated).max())

    @pytest.mark.medium
    def test_aberratedmielens_accepts_arbitrary_order_aberration(self):
        np.random.seed(1017)
        # use a lens angle large enough to test higher-order aberrations
        lens_angle = 0.8
        aberrations = 10 * np.random.randn(10)
        theory_low = AberratedMieLens(
            lens_angle=lens_angle, spherical_aberration=aberrations[0])
        theory_hi = AberratedMieLens(
            lens_angle=lens_angle, spherical_aberration=aberrations)

        holo_low = calc_holo(
            xschema, sphere, index, wavelen, xpolarization, theory=theory_low)
        holo_hi = calc_holo(
            xschema, sphere, index, wavelen, xpolarization, theory=theory_hi)

        # there should be a significant difference between the model with
        # higher-order aberrations and the one with only first-order
        assert_array_less(1e-3, np.abs(holo_hi - holo_low).max())

    @pytest.mark.fast
    def test_parameters_returns_correct_keys_and_values(self):
        np.random.seed(1707)
        lens_angle = np.random.rand()
        sph_ab = np.random.randn(5)
        theory = AberratedMieLens(
            lens_angle=lens_angle,
            spherical_aberration=sph_ab)

        parameters = theory.parameters
        correct_keys = {'lens_angle', 'spherical_aberration'}
        assert set(parameters.keys()) == correct_keys
        assert parameters['lens_angle'] == lens_angle
        assert sph_ab.tolist() == parameters['spherical_aberration'].tolist()

    @pytest.mark.fast
    def test_from_parameters_correctly_sets_parameters(self):
        np.random.seed(1709)
        lens_angle = np.random.rand()
        parameters = {'lens_angle': lens_angle,
                      'spherical_aberration': np.random.randn(3)}

        theory = AberratedMieLens().from_parameters(parameters)
        assert isinstance(theory, AberratedMieLens)
        assert theory.lens_angle == lens_angle
        assert_equal(theory.spherical_aberration,
                     parameters['spherical_aberration'])


def calculate_central_lobe_at(zs):
    illum_wavelength = 0.66  # 660 nm red light
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
