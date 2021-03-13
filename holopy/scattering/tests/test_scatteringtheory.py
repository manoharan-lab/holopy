import unittest

import numpy as np
import xarray as xr
from numpy.testing import assert_allclose, assert_equal
from nose.plugins.attrib import attr

from holopy.core import detector_grid, detector_points
from holopy.core.metadata import update_metadata, flat
from holopy.scattering.theory.scatteringtheory import ScatteringTheory
from holopy.scattering.theory import Mie
from holopy.scattering.scatterer import Sphere, Spheres, Ellipsoid
from holopy.scattering.errors import TheoryNotCompatibleError
from holopy.scattering.interface import prep_schema
from holopy.scattering.tests.common import xschema as XSCHEMA


SPHERE = Sphere(n=1.5, r=1.0, center=(0, 0, 2))
SPHERES = Spheres([
    Sphere(n=1.5, r=1.0, center=(-1, -1, 2)),
    Sphere(n=1.5, r=1.0, center=(+1, +1, 2)),
    ])
ELLIPSOID = Ellipsoid(n=1.5, r=(0.5, 0.5, 1.2), center=(0, 0, 2))
TOLS = {'atol': 1e-14, 'rtol': 1e-14}
MEDTOLS = {'atol': 1e-7, 'rtol': 1e-7}


SCAT_SCHEMA = prep_schema(
    detector_grid(shape=(5, 5), spacing=.1),
    medium_index=1.33, illum_wavelen=0.66, illum_polarization=False)


class MockTheory(ScatteringTheory):
    """Minimally-functional daughter of ScatteringTheory for fast tests."""
    def __init__(*args, **kwargs):
        pass  # an init is necessary for the repr

    def can_handle(self, scatterer):
        return isinstance(scatterer, Sphere)

    def raw_fields(self, positions, *args, **kwargs):
        return np.ones(positions.shape, dtype='complex128')


class MockScatteringMatrixBasedTheory(ScatteringTheory):
    """Minimally-functional daughter of ScatteringTheory which
    uses the scattering matrix pathway, for fast tests.
    Smells like a Rayleigh scatterer, just for fun. But it's not"""
    def __init__(*args, **kwargs):
        pass  # an init is necessary for the repr

    def can_handle(self, scatterer):
        return isinstance(scatterer, Sphere)

    def raw_scat_matrs(self, scatterer, positions, *args, **kwargs):
        strength = scatterer.n * scatterer.r
        scattering_matrix = np.array(
            [np.eye(2) for _ in range(positions.shape[1])])
        return strength * scattering_matrix.astype('complex128')


class TestScatteringTheory(unittest.TestCase):
    @attr('fast')
    def test_default_desired_coordinate_system_is_spherical(self):
        for cls in [ScatteringTheory, Mie, MockTheory]:
            self.assertTrue(cls.desired_coordinate_system == 'spherical')

    @attr('fast')
    def test_can_handle_not_implemented(self):
        theory = ScatteringTheory()
        self.assertRaises(NotImplementedError, theory.can_handle, SPHERE)

    @attr('fast')
    def test_raw_scat_matrs_not_implemented(self):
        theory = ScatteringTheory()
        args = (None,) * 4  # 4 positional arguments....
        self.assertRaises(NotImplementedError, theory.raw_scat_matrs, *args)

    @attr('fast')
    def test_raw_cross_sections_not_implemented(self):
        theory = ScatteringTheory()
        args = (None,) * 4  # 4 positional arguments....
        self.assertRaises(
            NotImplementedError, theory.raw_cross_sections, *args)

    @attr('fast')
    def test_default_parameters_is_empty_dict(self):
        theory = ScatteringTheory()
        self.assertEqual(theory.parameters, dict())

    @attr('fast')
    def test_from_parameters_callable_by_default(self):
        tmp = ScatteringTheory()
        theory = tmp.from_parameters(tmp.parameters)
        self.assertIsInstance(theory, ScatteringTheory)



class TestMockTheory(unittest.TestCase):
    @attr("fast")
    def test_creation(self):
        theory = MockTheory()
        self.assertTrue(theory is not None)

    @attr("fast")
    def test_can_handle_sphere(self):
        theory = MockTheory()
        self.assertTrue(theory.can_handle(SPHERE))

    @attr("fast")
    def test_cannot_handle_spheres(self):
        theory = MockTheory()
        self.assertFalse(theory.can_handle(SPHERES))

    @attr("fast")
    def test_raw_fields_returns_correct_shape(self):
        theory = MockTheory()
        positions = np.random.randn(65, 3)

        medium_wavevec = 1  # doesn't matter
        medium_index = 1.33  # doesn't matter
        illum_polarization = (1, 0)  # doesn't matter
        fields = theory.raw_fields(
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
        fields = theory.raw_fields(
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
        fields = theory.raw_fields(
            positions, SPHERE, medium_wavevec, medium_index,
            illum_polarization)

        self.assertTrue(fields.dtype.name == 'complex128')


class TestMockScatteringMatrixBasedTheory(unittest.TestCase):
    @attr("fast")
    def test_creation(self):
        theory = MockScatteringMatrixBasedTheory()
        self.assertTrue(theory is not None)

    @attr("fast")
    def test_can_handle_sphere(self):
        theory = MockScatteringMatrixBasedTheory()
        self.assertTrue(theory.can_handle(SPHERE))

    @attr("fast")
    def test_cannot_handle_spheres(self):
        theory = MockScatteringMatrixBasedTheory()
        self.assertFalse(theory.can_handle(SPHERES))

    @attr("fast")
    def test_raw_scat_matrs_returns_correct_shape(self):
        theory = MockScatteringMatrixBasedTheory()
        positions = np.random.randn(3, 65)
        scattering_matrices = theory.raw_scat_matrs(SPHERE, positions)
        self.assertTrue(scattering_matrices.shape == (positions.shape[1], 2, 2))

    @attr("fast")
    def test_raw_scat_matrs_returns_eyes(self):
        theory = MockScatteringMatrixBasedTheory()
        positions = np.random.randn(3, 65)
        scattering_matrices = theory.raw_scat_matrs(SPHERE, positions)
        eye = np.eye(2)
        each_is_eye = [
            np.isclose(np.diag(m).std(), 0, **TOLS)
            for m in scattering_matrices]
        self.assertTrue(all(each_is_eye))

    @attr("fast")
    def test_raw_fields_returns_dtype_complex(self):
        theory = MockScatteringMatrixBasedTheory()
        positions = np.random.randn(3, 65)
        scattering_matrices = theory.raw_scat_matrs(SPHERE, positions)
        self.assertTrue(scattering_matrices.dtype.name == 'complex128')


if __name__ == '__main__':
    unittest.main()

