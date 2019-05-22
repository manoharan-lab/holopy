import unittest

import numpy as np
import xarray as xr
from numpy.testing import assert_allclose, assert_equal
from nose.plugins.attrib import attr

from holopy.core import detector_grid, detector_points
from holopy.core.metadata import update_metadata
from holopy.scattering.theory.scatteringtheory import (
    ScatteringTheory, stack_spherical)
from holopy.scattering.scatterer import Sphere, Spheres, Ellipsoid
from holopy.scattering.errors import TheoryNotCompatibleError
from holopy.scattering.tests.common import xschema as XSCHEMA


SPHERE = Sphere(n=1.5, r=1.0, center=(0, 0, 2))
SPHERES = Spheres([
    Sphere(n=1.5, r=1.0, center=(-1, -1, 2)),
    Sphere(n=1.5, r=1.0, center=(+1, +1, 2)),
    ])
ELLIPSOID = Ellipsoid(n=1.5, r=(0.5, 0.5, 1.2), center=(0, 0, 2))
TOLS = {'atol': 1e-14, 'rtol': 1e-14}


class TestSphereCoords(unittest.TestCase):
    @attr("fast")
    def test_sphere_coords(self):
        detector = detector_grid(shape=(2, 2), spacing=0.1)
        spherical = ScatteringTheory.sphere_coords(
            detector, wavevec=2*np.pi*1.33/.66, origin=(0, 0, 1))
        pos = stack_spherical(spherical).T
        true_pos = np.array([
            [ 12.66157039,   0.        ,   0.        ],
            [ 12.72472076,   0.09966865,   1.57079633],
            [ 12.72472076,   0.09966865,   0.        ],
            [ 12.78755927,   0.1404897 ,   0.78539816]])
        self.assertTrue(np.allclose(pos, true_pos))

    @attr("fast")
    def test_transform_to_desired_coordinates(self):
        detector = detector_grid(shape=(2, 2), spacing=0.1)
        spherical = ScatteringTheory._transform_to_desired_coordinates(
            detector, origin=(0, 0, 1), wavevec=2*np.pi*1.33/.66)

        true_pos = np.transpose([
            [ 12.66157039,   0.        ,   0.        ],
            [ 12.72472076,   0.09966865,   1.57079633],
            [ 12.72472076,   0.09966865,   0.        ],
            [ 12.78755927,   0.1404897 ,   0.78539816]])
        self.assertTrue(np.allclose(spherical, true_pos))


class TestScatteringTheory(unittest.TestCase):
    @attr("fast")
    def test_calc_field_equals_calc_singlecolor_for_single_color(self):
        theory = MockTheory()
        from_calc_scat = theory.calculate_scattered_field(SPHERE, XSCHEMA)
        from_calc_single = theory._calculate_single_color_scattered_field(
            SPHERE, XSCHEMA)
        is_ok = np.allclose(
            from_calc_scat.values, from_calc_single.values, **TOLS)
        self.assertTrue(is_ok)

    @attr("fast")
    def test_calc_singlecolor_equals_get_field_from_for_sphere(self):
        theory = MockTheory()
        from_calc_single = theory._calculate_single_color_scattered_field(
            SPHERE, XSCHEMA)
        from_get_field = theory._get_field_from(SPHERE, XSCHEMA)
        is_ok = np.allclose(
            from_get_field.values, from_calc_single.values, **TOLS)
        self.assertTrue(is_ok)

    @attr("fast")
    def test_calc_singlecolor_raises_error_for_cant_handle(self):
        theory = MockTheory()
        assert not theory._can_handle(ELLIPSOID)
        self.assertRaises(
            TheoryNotCompatibleError,
            theory._calculate_single_color_scattered_field,
            ELLIPSOID, XSCHEMA)

    @attr("fast")
    def test_calc_singlecolor_adds_get_field_from_for_spheres(self):
        theory = MockTheory()
        from_calc_single = theory._calculate_single_color_scattered_field(
            SPHERES, XSCHEMA)
        components = SPHERES.get_component_list()
        from_get_field = sum([
            theory._get_field_from(c, XSCHEMA).values for c in components])

        is_ok = np.allclose(
            from_get_field, from_calc_single.values, **TOLS)
        self.assertTrue(is_ok)


class TestMockTheory(unittest.TestCase):
    @attr("fast")
    def test_creation(self):
        theory = MockTheory()
        self.assertTrue(theory is not None)

    @attr("fast")
    def test_can_handle_sphere(self):
        theory = MockTheory()
        self.assertTrue(theory._can_handle(SPHERE))

    @attr("fast")
    def test_cannot_handle_spheres(self):
        theory = MockTheory()
        self.assertFalse(theory._can_handle(SPHERES))

    @attr("fast")
    def test_raw_fields_returns_correct_shape(self):
        theory = MockTheory()
        positions = np.random.randn(65, 3)

        medium_wavevec = 1  # doesn't matter
        medium_index = 1.33  # doesn't matter
        illum_polarization = (1, 0)  # doesn't matter
        fields = theory._raw_fields(
            positions, SPHERE, medium_wavevec, medium_index,
            illum_polarization)

        self.assertTrue(fields.shape == positions.shape)

    @attr("fast")
    def test_raw_fields_returns_ones(self):
        theory = MockTheory()
        positions = np.random.randn(65, 3)

        medium_wavevec = 1  # doesn't matter
        medium_index = 1.33  # doesn't matter
        illum_polarization = (1, 0)  # doesn't matter
        fields = theory._raw_fields(
            positions, SPHERE, medium_wavevec, medium_index,
            illum_polarization)

        self.assertTrue(np.allclose(fields, 1.0, **TOLS))

    @attr("fast")
    def test_raw_fields_returns_dtype_complex(self):
        theory = MockTheory()
        positions = np.random.randn(65, 3)

        medium_wavevec = 1  # doesn't matter
        medium_index = 1.33  # doesn't matter
        illum_polarization = (1, 0)  # doesn't matter
        fields = theory._raw_fields(
            positions, SPHERE, medium_wavevec, medium_index,
            illum_polarization)

        self.assertTrue(fields.dtype.name == 'complex128')


class MockTheory(ScatteringTheory):
    def _can_handle(self, scatterer):
        return isinstance(scatterer, Sphere)

    def _raw_fields(self, positions, *args, **kwargs):
        return np.ones(positions.shape, dtype='complex128')


if __name__ == '__main__':
    unittest.main()

