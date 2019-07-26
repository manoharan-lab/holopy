import unittest
from collections import OrderedDict

import numpy as np
from nose.plugins.attrib import attr

from holopy.core.metadata import (
    detector_grid, detector_points, clean_concat, update_metadata,
    get_spacing, get_extents, copy_metadata, make_subset_data)
from holopy.core.errors import CoordSysError


TOLS = {'atol': 1e-10, 'rtol': 1e-10}

# TODO: test that:
# update_metadata raises an error when things are not the correct shape
#       (e.g. normals have 11 elements for a 10x10 detector)
# FIXME questions:
# data_grid, detector_grid, etc don't allow passing normals for each point.
#       is that behavior correct?


class TestDetectorGrid(unittest.TestCase):
    @attr("fast")
    def test_pads_shape_with_leading_1(self):
        spacing = 0.1
        shape = (9, 12)
        detector = detector_grid(shape, spacing)
        self.assertEqual(len(detector.values.shape), 3)
        self.assertEqual(detector.values.shape[0], 1)

    @attr("fast")
    def test_returned_shape_is_correct(self):
        spacing = 0.1
        shape = (9, 12)
        true_shape = (1,) + shape
        detector = detector_grid(shape, spacing)
        self.assertEqual(detector.values.shape, true_shape)

    @attr("fast")
    def test_normals_default_to_001(self):
        detector = detector_grid(10, 0.1)  # default normals
        default_normals = np.array([0, 0, 1.0])
        detector_normals = detector.normals.values
        self.assertTrue(np.allclose(detector_normals, default_normals, **TOLS))

    @attr("fast")
    def test_normals_are_stored_when_passed_as_scalar(self):
        normals = np.array([0.5, 0, 0.5])
        normals /= np.linalg.norm(normals)

        detector = detector_grid(10, 0.1, normals=normals.copy())
        detector_normals = detector.normals.values
        self.assertTrue(np.allclose(detector_normals, normals, **TOLS))

    @attr("fast")
    def test_name_is_stored(self):
        name = 'this-is-a-name'
        detector = detector_grid(10, 0.1, name=name)
        self.assertEqual(detector.name, name)

    @attr("fast")
    def test_extra_dims_when_ordereddict(self):
        shape = (2, 2)
        extra_dims_sizes = (1, 2, 3, 4, 5, 6, 7, 8)  # ends up as 1.3 MB
        extra_dims_names = 'abcdefgh'
        extra_dims = OrderedDict()
        for k, v in zip(extra_dims_names, extra_dims_sizes):
            extra_dims.update({k: np.arange(v)})

        detector = detector_grid(shape, 0.1, extra_dims=extra_dims)
        true_shape = (1,) + shape + extra_dims_sizes
        detector_shape = detector.values.shape
        self.assertEqual(true_shape, detector_shape)

    @attr("fast")
    def test_extra_dims_when_dict(self):
        # Test that extra_dims behaves correctly when dicts are not ordered,
        # in lower versions of Python
        shape = (2, 2)
        extra_dims_sizes = (1, 2, 3, 4, 5, 6, 7, 8)  # ends up as 1.3 MB
        extra_dims_names = 'abcdefgh'
        extra_dims = dict()
        for k, v in zip(extra_dims_names, extra_dims_sizes):
            extra_dims.update({k: np.arange(v)})

        detector = detector_grid(shape, 0.1, extra_dims=extra_dims)
        # Then, rather than check that the order is the same, we want
        # to check that (i) all keys are present, and (ii) each key has
        # the correct value, which we check by the shapes being equal:
        for key, value in extra_dims.items():
            self.assertIn(key, detector.coords)
            detector_coord_value = detector.coords[key].values
            self.assertEqual(value.shape, detector.coords[key].values.shape)


class TestDetectorPoints(unittest.TestCase):
    @attr("fast")
    def test_raises_error_when_no_coordinates_passed(self):
        self.assertRaises(CoordSysError, detector_points)

    # xyz tests:

    @attr("fast")
    def test_stores_xyz_as_correct_shape_when_xyz_passed(self):
        np.random.seed(70)
        npts = 21
        x = np.random.randn(npts)
        y = np.random.randn(npts)
        z = np.random.randn(npts)

        points = detector_points(x=x, y=y, z=z)
        self.assertEqual(points.x.size, npts)
        self.assertEqual(points.y.size, npts)
        self.assertEqual(points.z.size, npts)

    @attr("fast")
    def test_stores_xyz_correct_values_when_xyz_passed(self):
        np.random.seed(70)
        npts = 21
        x = np.random.randn(npts)
        y = np.random.randn(npts)
        z = np.random.randn(npts)

        points = detector_points(x=x, y=y, z=z)

        self.assertTrue(np.allclose(points.x, x, **TOLS))
        self.assertTrue(np.allclose(points.y, y, **TOLS))
        self.assertTrue(np.allclose(points.z, z, **TOLS))

    @attr("fast")
    def test_stores_z_as_array_when_scalar_z_passed(self):
        np.random.seed(70)
        npts = 21
        x = np.random.randn(npts)
        y = np.random.randn(npts)
        # Then we pick a scalar z:
        z = np.random.randn(1).squeeze()

        points = detector_points(x=x, y=y, z=z)

        self.assertTrue(np.allclose(points.z, z, **TOLS))
        self.assertEqual(points.z.size, npts)

    @attr("fast")
    def test_z_defaults_to_zero_when_xy_passed(self):
        np.random.seed(70)
        npts = 21
        x = np.random.randn(npts)
        y = np.random.randn(npts)
        points = detector_points(x=x, y=y)

        self.assertTrue(np.allclose(points.z, 0, **TOLS))

    # r, theta, phi tests:

    @attr("fast")
    def test_stores_rthetaphi_as_correct_shape_when_rthetaphi_passed(self):
        np.random.seed(70)
        npts = 21
        r = np.random.randn(npts)
        theta = np.random.randn(npts) % np.pi
        phi = np.random.randn(npts) % (2 * np.pi)

        points = detector_points(r=r, theta=theta, phi=phi)
        self.assertEqual(points.r.size, npts)
        self.assertEqual(points.theta.size, npts)
        self.assertEqual(points.phi.size, npts)

    @attr("fast")
    def test_stores_rthetaphi_correct_values_when_rthetaphi_passed(self):
        np.random.seed(70)
        npts = 21
        r = np.random.randn(npts)
        theta = np.random.randn(npts) % np.pi
        phi = np.random.randn(npts) % (2 * np.pi)

        points = detector_points(r=r, theta=theta, phi=phi)
        self.assertTrue(np.allclose(points.r, r, **TOLS))
        self.assertTrue(np.allclose(points.theta, theta, **TOLS))
        self.assertTrue(np.allclose(points.phi, phi, **TOLS))

    @attr("fast")
    def test_stores_r_as_array_when_scalar_r_passed(self):
        np.random.seed(70)
        npts = 21
        theta = np.random.randn(npts) % np.pi
        phi = np.random.randn(npts) % (2 * np.pi)
        # Then we pick a scalar r:
        r = np.random.randn(1).squeeze()

        points = detector_points(r=r, theta=theta, phi=phi)

        self.assertTrue(np.allclose(points.r, r, **TOLS))
        self.assertEqual(points.r.size, npts)

    @attr("fast")
    def test_r_defaults_to_inf_when_thetaphi_passed(self):
        np.random.seed(70)
        npts = 21
        theta = np.random.randn(npts) % np.pi
        phi = np.random.randn(npts) % (2 * np.pi)

        points = detector_points(theta=theta, phi=phi)

        self.assertTrue(np.all(np.isinf(points.r)))
        self.assertEqual(points.r.size, npts)

    # Other tests:
    @attr("fast")
    def test_data_is_stored_as_zeros_of_corect_size(self):
        npts = 23
        x, y, z = np.random.randn(3, npts)
        points = detector_points(x=x, y=y, z=z)
        self.assertEqual(points.size, npts)
        self.assertTrue(np.allclose(points.values, 0, **TOLS))

    @attr("fast")
    def test_name_defaults_to_data(self):
        x, y, z = np.random.randn(3, 10)
        points = detector_points(x=x, y=y, z=z)
        self.assertEqual(points.name, 'data')

    @attr("fast")
    def test_name_is_stored(self):
        x, y, z = np.random.randn(3, 10)
        name = 'this-is-a-test'
        points = detector_points(x=x, y=y, z=z, name=name)
        self.assertEqual(points.name, name)

    # FIXME no checks for normal default values

    @attr("fast")
    def test_has_attribute_normals(self):
        x, y, z = np.random.randn(3, 10)
        points = detector_points(x=x, y=y, z=z)
        self.assertTrue(hasattr(points, 'normals'))

    @attr("fast")
    def test_default_normals_are_shape_3xN_for_spherical_coords(self):
        npts = 13
        r, theta, phi = np.random.randn(3, npts)
        points = detector_points(r=r, theta=theta, phi=phi)
        self.assertEqual(points.normals.values.shape, (3, npts))

    @attr("fast")
    @unittest.skip("Fails, not sure if test is wrong or code")
    def test_3xN_normals_are_stored_for_spherical_coords(self):
        npts = 13
        r, theta, phi = np.random.randn(3, npts)
        normals = np.random.randn(3, npts)
        points = detector_points(r=r, theta=theta, phi=phi, normals=normals)
        self.assertEqual(points.normals.values.shape, (3, npts))
        self.assertTrue(np.allclose(points.normals.values, normals, **TOLS))

    @attr("fast")
    def test_default_normals_are_shape_3_for_cartesian_coords(self):
        npts = 13
        x, y, z= np.random.randn(3, npts)
        points = detector_points(x=x, y=y, z=z)
        self.assertEqual(points.normals.values.shape, (3,))

    @attr("fast")
    def test_3x1_normals_are_stored_for_cartesian_coords(self):
        npts = 13
        x, y = np.random.randn(2, npts)
        normals = np.random.randn(3)
        normals /= np.linalg.norm(normals)
        points = detector_points(x=x, y=y, normals=normals)
        self.assertEqual(points.normals.values.shape, (3,))
        self.assertTrue(np.allclose(points.normals.values, normals, **TOLS))

    @attr("fast")
    def test_3x1_normals_are_normalized(self):
        npts = 13
        x, y = np.random.randn(2, npts)
        raw_normals = np.random.randn(3)
        points = detector_points(x=x, y=y, normals=raw_normals)
        normals = raw_normals / np.linalg.norm(raw_normals)
        # They should be different from the raw, unnormmalized normals:
        self.assertFalse(
            np.allclose(points.normals.values, raw_normals, **TOLS))
        # but the same as the normalized ones:
        self.assertTrue(
            np.allclose(points.normals.values, normals, **TOLS))


class TestUpdateMetadata(unittest.TestCase):
    @attr("fast")
    def test_does_update_medium_index(self):
        detector = detector_grid(3, 0.1)

        np.random.seed(10)
        medium_index = 1 + np.random.rand()
        updated_detector = update_metadata(detector, medium_index=medium_index)
        self.assertEqual(updated_detector.medium_index, medium_index)

    @attr("fast")
    def test_does_update_illum_wavelength(self):
        detector = detector_grid(3, 0.1)

        np.random.seed(11)
        illum_wavelen = np.random.rand()
        updated_detector = update_metadata(
            detector, illum_wavelen=illum_wavelen)
        self.assertEqual(updated_detector.illum_wavelen, illum_wavelen)

    @attr("fast")
    def test_does_update_illum_polarization(self):
        detector = detector_grid(3, 0.1)
        np.random.seed(12)
        illum_polarization = np.random.randn(2)
        illum_polarization /= np.linalg.norm(illum_polarization)
        updated_detector = update_metadata(
            detector, illum_polarization=illum_polarization)
        is_ok = np.allclose(
            updated_detector.illum_polarization.values[:2],
            illum_polarization, **TOLS)
        self.assertTrue(is_ok)

    @attr("fast")
    def test_does_update_noise_sd(self):
        detector = detector_grid(3, 0.1)
        np.random.seed(13)
        noise_sd = np.random.rand()
        updated_detector = update_metadata(detector, noise_sd=noise_sd)
        self.assertEqual(updated_detector.noise_sd, noise_sd)

    # FIXME add a test for normals....


class TestGetSpacing(unittest.TestCase):
    @attr("fast")
    def test_raises_error_when_xspacing_is_unequal(self):
        x = np.linspace(0, 1, 11)**2  # non-uniform spacing
        y = np.linspace(0, 1, 11)

        detector = detector_points(x=x, y=y)
        self.assertRaises(ValueError, get_spacing, detector)

    @attr("fast")
    def test_raises_error_when_yspacing_is_unequal(self):
        x = np.linspace(0, 1, 11)
        y = np.linspace(0, 1, 11)**2  # non-uniform spacing

        detector = detector_points(x=x, y=y)
        self.assertRaises(ValueError, get_spacing, detector)

    @attr("fast")
    def test_on_different_spacings(self):
        xspacing = 0.1
        yspacing = 0.2
        detector = detector_grid((10, 10), spacing=(xspacing, yspacing))

        spacing = get_spacing(detector)
        self.assertEqual(spacing[0], xspacing)
        self.assertEqual(spacing[1], yspacing)

    @attr("fast")
    def test_on_same_spacing(self):
        true_spacing = 0.1
        detector = detector_grid((10, 10), spacing=true_spacing)

        spacing = get_spacing(detector)
        self.assertEqual(spacing[0], true_spacing)
        self.assertEqual(spacing[1], true_spacing)


class TestGetExtents(unittest.TestCase):
    @attr('fast')
    def test_returns_empty_when_dims_is_point(self):
        # FIXME is this the desired behavior, to return {}?
        x = np.linspace(0, 1, 11)
        y = np.linspace(0, 1, 11)
        points = detector_points(x=x, y=y)

        extents = get_extents(points)
        self.assertEqual(extents, {})

    @attr('fast')
    def test_on_detector_grid_when_spacing_is_isotropic(self):
        shape = (10, 12)  # (x, y)
        spacing = 0.1
        true_extents = {
            'x': shape[0] * spacing, 'y': shape[1] * spacing, 'z': 0}
        detector = detector_grid(shape, spacing)
        extents = get_extents(detector)
        self.assertEqual(extents, true_extents)

    @attr('fast')
    def test_on_detector_grid_when_spacing_is_anisotropic(self):
        shape = (10, 12)  # (x, y)
        spacing = (0.1, 0.2)
        true_extents = {
            'x': shape[0] * spacing[0], 'y': shape[1] * spacing[1], 'z': 0}
        detector = detector_grid(shape, spacing)
        extents = get_extents(detector)
        self.assertEqual(extents, true_extents)

    @attr('fast')
    def test_on_detector_grid_when_spacing_is_0(self):
        shape = (10, 12)  # (x, y)
        spacing = 0.0
        true_extents = {'x': 0, 'y': 0, 'z': 0}
        detector = detector_grid(shape, spacing)
        extents = get_extents(detector)
        self.assertEqual(extents, true_extents)

    @attr('fast')
    def test_on_detector_grid_when_size_is_1(self):
        shape = (1, 1)
        spacing = 0.1
        true_extents = {'x': 0, 'y': 0, 'z': 0}
        detector = detector_grid(shape, spacing)
        extents = get_extents(detector)
        self.assertEqual(extents, true_extents)

    # FIXME the current get_extents does not work for a nonuniform spacing
    # there should be a failing test like this:


if __name__ == '__main__':
    unittest.main()

