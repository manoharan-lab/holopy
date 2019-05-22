import unittest

import numpy as np
import xarray as xr
from numpy.testing import assert_allclose, assert_equal
from nose.plugins.attrib import attr

from holopy.core import detector_grid, detector_points
from holopy.core.metadata import update_metadata
from holopy.scattering.theory.scatteringtheory import (
    ScatteringTheory, stack_spherical)
from holopy.scattering.scatterer import Sphere, Spheres


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


class TestMockTheory(unittest.TestCase):
    @attr("fast")
    def test_creation(self):
        theory = MockTheory()
        self.assertTrue(theory is not None)

    @attr("fast")
    def test_can_handle_sphere(self):
        theory = MockTheory()
        sphere = Sphere(n=1.5, r=1.0, center=(0, 0, 0))
        self.assertTrue(theory._can_handle(sphere))

    @attr("fast")
    def test_cannot_handle_spheres(self):
        theory = MockTheory()
        spheres = Spheres([
            Sphere(n=1.5, r=1.0, center=(0, 0, 0)),
            Sphere(n=1.5, r=1.0, center=(2, 2, 2)),
            ])
        self.assertFalse(theory._can_handle(spheres))

    @attr("fast")
    def test_raw_fields_returns_correct_shape(self):
        theory = MockTheory()
        sphere = Sphere(n=1.5, r=1.0, center=(0, 0, 0))
        positions = np.random.randn(65, 3)

        medium_wavevec = 1  # doesn't matter
        medium_index = 1.33  # doesn't matter
        illum_polarization = (1, 0)  # doesn't matter
        fields = theory._raw_fields(
            positions, sphere, medium_wavevec, medium_index,
            illum_polarization)

        self.assertTrue(fields.shape == positions.shape)

    @attr("fast")
    def test_raw_fields_returns_ones(self):
        theory = MockTheory()
        sphere = Sphere(n=1.5, r=1.0, center=(0, 0, 0))
        positions = np.random.randn(65, 3)

        medium_wavevec = 1  # doesn't matter
        medium_index = 1.33  # doesn't matter
        illum_polarization = (1, 0)  # doesn't matter
        fields = theory._raw_fields(
            positions, sphere, medium_wavevec, medium_index,
            illum_polarization)

        self.assertTrue(np.allclose(fields, 1.0, atol=1e-14, rtol=1e-14))


class MockTheory(ScatteringTheory):
    def _can_handle(self, scatterer):
        return isinstance(scatterer, Sphere)

    def _raw_fields(self, positions, *args, **kwargs):
        return np.ones_like(positions)


if __name__ == '__main__':
    unittest.main()

