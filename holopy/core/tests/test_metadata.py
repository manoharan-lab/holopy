import unittest

import numpy as np
from nose.plugins.attrib import attr

from holopy.core.metadata import (
    detector_grid, detector_points, clean_concat, update_metadata,
    get_spacing, get_extents, copy_metadata, make_subset_data, data_grid)
from holopy.core.errors import CoordSysError


TOLS = {'atol': 1e-15, 'rtol': 1e-15}
METADATA_VALUES = {
    'medium_index': 1.234,
    'illum_wavelen': 0.567,
    'illum_polarization': (1, 0),
    'noise_sd': 0.89,
    }


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
    def test_name_is_stored(self):
        name = 'this-is-a-name'
        detector = detector_grid(10, 0.1, name=name)
        self.assertEqual(detector.name, name)

    @attr("fast")
    def test_extra_dims_when_ordered_dict(self):
        shape = (2, 2)
        extra_dims_sizes = (1, 2, 3, 4, 5, 6, 7, 8)  # ends up as 1.3 MB
        extra_dims_names = 'abcdefgh'
        extra_dims = {}
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

        self.assertTrue(np.all(points.x == x))
        self.assertTrue(np.all(points.y == y))
        self.assertTrue(np.all(points.z == z))

    @attr("fast")
    def test_stores_z_as_array_when_scalar_z_passed(self):
        np.random.seed(70)
        npts = 21
        x = np.random.randn(npts)
        y = np.random.randn(npts)
        # Then we pick a scalar z:
        z = np.random.randn(1).squeeze()

        points = detector_points(x=x, y=y, z=z)

        self.assertTrue(np.all(points.z == z))
        self.assertEqual(points.z.size, npts)

    @attr("fast")
    def test_z_defaults_to_zero_when_xy_passed(self):
        np.random.seed(70)
        npts = 21
        x = np.random.randn(npts)
        y = np.random.randn(npts)
        points = detector_points(x=x, y=y)

        self.assertTrue(np.all(points.z == 0))

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
        self.assertTrue(np.all(points.r == r))
        self.assertTrue(np.all(points.theta == theta))
        self.assertTrue(np.all(points.phi == phi))

    @attr("fast")
    def test_stores_r_as_array_when_scalar_r_passed(self):
        np.random.seed(70)
        npts = 21
        theta = np.random.randn(npts) % np.pi
        phi = np.random.randn(npts) % (2 * np.pi)
        # Then we pick a scalar r:
        r = np.random.randn(1).squeeze()

        points = detector_points(r=r, theta=theta, phi=phi)

        self.assertTrue(np.all(points.r == r))
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
        self.assertTrue(np.all(points.values == 0))

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


class TestCleanConcat(unittest.TestCase):
    @attr("fast")
    def test_concatenates_data(self):
        data1 = make_data(seed=1)
        data2 = make_data(seed=2)

        concatenated = clean_concat([data1, data2], 'point')
        self.assertEqual(concatenated.shape, data1.shape + (2,))

    @attr("fast")
    def test_preserves_data_order(self):
        data1 = make_data(seed=1)
        data2 = make_data(seed=2)
        data = [data1, data2]

        concatenated = clean_concat(data, 'point')
        self.assertTrue(np.all(concatenated.values[..., 0] == data[0].values))
        self.assertTrue(np.all(concatenated.values[..., 1] == data[1].values))

    @attr("fast")
    def test_preserves_metadata_keys(self):
        data1 = update_metadata(make_data(seed=1), **METADATA_VALUES)
        data2 = update_metadata(make_data(seed=2), **METADATA_VALUES)
        data = [data1, data2]

        concatenated = clean_concat(data, 'point')
        for key in METADATA_VALUES.keys():
            self.assertIn(key, concatenated.attrs)
            self.assertTrue(hasattr(concatenated, key))

    @attr("fast")
    def test_preserves_metadata_values(self):
        data1 = update_metadata(make_data(seed=1), **METADATA_VALUES)
        data2 = update_metadata(make_data(seed=2), **METADATA_VALUES)
        data = [data1, data2]

        concatenated = clean_concat(data, 'point')
        for key, value in METADATA_VALUES.items():
            if key != 'illum_polarization':
                self.assertEqual(getattr(concatenated, key), value)
        polarization_ok = np.all(
            concatenated.illum_polarization[:2] ==
            METADATA_VALUES['illum_polarization'])
        self.assertTrue(polarization_ok)


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
        is_ok = np.all(
            updated_detector.illum_polarization.values[:2] ==
            illum_polarization)
        self.assertTrue(is_ok)

    @attr("fast")
    def test_does_update_noise_sd(self):
        detector = detector_grid(3, 0.1)
        np.random.seed(13)
        noise_sd = np.random.rand()
        updated_detector = update_metadata(detector, noise_sd=noise_sd)
        self.assertEqual(updated_detector.noise_sd, noise_sd)


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
        x = np.linspace(0, 1, 11)
        y = np.linspace(0, 1, 11)
        points = detector_points(x=x, y=y)

        msg = "Cannot get extent for detector_points"
        self.assertRaisesRegex(
            ValueError, msg, get_extents, points)

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


class TestCopyMetadata(unittest.TestCase):
    @attr('fast')
    def test_copies_metadata_keys(self):
        metadata = make_metadata()
        data = make_data()
        copied = copy_metadata(metadata, data)
        for key in METADATA_VALUES.keys():
            self.assertIn(key, copied.attrs)
            self.assertTrue(hasattr(copied, key))

    @attr('fast')
    def test_copies_metadata_values(self):
        metadata = make_metadata()
        data = make_data()
        copied = copy_metadata(metadata, data)
        # we check illum_polarization separately:
        illum_polarization = METADATA_VALUES['illum_polarization']
        self.assertTrue(
            np.all(illum_polarization == copied.illum_polarization.values[:2]))
        # Then we check the rest:
        for key, value in METADATA_VALUES.items():
            if key != 'illum_polarization':
                self.assertEqual(value, getattr(copied, key))

    @attr('fast')
    def test_copies_coords(self):
        metadata = make_metadata()
        data = make_data()
        copied = copy_metadata(metadata, data)
        for coordinate in data.coords.keys():
            old_coords = data.coords[coordinate].values
            copied_coords = copied.coords[coordinate].values
            self.assertTrue(np.all(old_coords == copied_coords))

    @attr('fast')
    def test_copies_name(self):
        metadata = make_metadata()
        data = make_data()
        copied = copy_metadata(metadata, data)
        self.assertEqual(metadata.name, copied.name)

    @attr('fast')
    def test_does_not_change_data(self):
        metadata = make_metadata()
        data = make_data()
        copied = copy_metadata(metadata, data)
        self.assertTrue(np.all(data.values == copied.values))


class TestMakeSubsetData(unittest.TestCase):
    # to test:
    # pixels, seed
    # Not used:
    #   random_subset
    #   return_selection
    @attr("fast")
    def test_returns_data_when_nothing_passed(self):
        data = make_data()
        subset = make_subset_data(data)
        self.assertTrue(data is subset)

    @attr("fast")
    def test_returns_correct_number_of_pixels(self):
        data = make_data()
        number_of_pixels_to_select = 3
        subset = make_subset_data(data, pixels=number_of_pixels_to_select)
        self.assertEqual(subset.size, number_of_pixels_to_select)

    @attr("fast")
    @unittest.skip("subset_data fails for detector_points")  # FIXME
    def test_returns_correct_number_of_pixels_on_detector_points(self):
        points = make_points()
        number_of_pixels_to_select = 3
        subset = make_subset_data(points, pixels=number_of_pixels_to_select)
        self.assertEqual(subset.size, number_of_pixels_to_select)

    @attr("fast")
    def test_returns_elements_of_data(self):
        data = make_data()
        number_of_pixels_to_select = 3
        subset = make_subset_data(data, pixels=number_of_pixels_to_select)
        for datum in subset.values:
            self.assertIn(datum, data.values.ravel())

    @attr('fast')
    def test_seed_returns_same_data(self):
        data = make_data()
        number_of_pixels_to_select = data.size // 2
        seed = 243
        subset1 = make_subset_data(
            data, pixels=number_of_pixels_to_select, seed=seed)
        subset2 = make_subset_data(
            data, pixels=number_of_pixels_to_select, seed=seed)
        self.assertTrue(np.all(subset1.values == subset2.values))

    @attr('fast')
    def test_returns_correct_xy_coords(self):
        data = make_data()
        number_of_pixels_to_select = 3
        subset = make_subset_data(data, pixels=number_of_pixels_to_select)

        data_xy_indices = [
            np.nonzero(data.values == from_subset)[1:]  # index[0] is z
            for from_subset in subset.values]
        for subset_index, data_xy_index in enumerate(data_xy_indices):
            coords_from_subset = [
                subset.coords[k].values[subset_index]
                for k in 'xy']
            coords_from_data = [
                data.coords[k].values[data_xy_index[which].squeeze()]
                for which, k in enumerate('xy')]
            self.assertEqual(coords_from_subset, coords_from_data)

    @attr('fast')
    def test_returns_correct_z_coords(self):
        data = make_data()
        number_of_pixels_to_select = 3
        subset = make_subset_data(data, pixels=number_of_pixels_to_select)

        subset_z_coords = subset.coords['z'].values
        data_z_coords = data.coords['z'].values
        self.assertTrue(np.all(subset_z_coords == data_z_coords))


def make_data(seed=1):
    np.random.seed(seed)
    shape = (5, 5)
    data_values = np.random.randn(*shape)
    data = detector_grid(shape, 0.1)
    data.values[:] = data_values
    return data


def make_points(seed=1):
    np.random.seed(seed)
    npts = 21
    x = np.random.randn(npts)
    y = np.random.randn(npts)
    z = np.random.randn(npts)

    points = detector_points(x=x, y=y, z=z)
    data_values = np.random.randn(npts)
    points.values[:] = data_values
    return points


def make_metadata():
    detector = detector_grid(7, 0.1, name='metadata')
    return update_metadata(detector, **METADATA_VALUES)


if __name__ == '__main__':
    unittest.main()

