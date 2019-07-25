import unittest
from collections import OrderedDict

import numpy as np
from nose.plugins.attrib import attr

from holopy.core.metadata import (
    detector_grid, detector_points, clean_concat, update_metadata,
    get_spacing, get_extents, copy_metadata, make_subset_data)


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
        shape = (2, 2)
        extra_dims_sizes = (1, 2, 3, 4, 5, 6, 7, 8)  # ends up as 1.3 MB
        extra_dims_names = 'abcdefgh'
        extra_dims = dict()
        for k, v in zip(extra_dims_names, extra_dims_sizes):
            extra_dims.update({k: np.arange(v)})

        detector = detector_grid(shape, 0.1, extra_dims=extra_dims)
        true_shape = (1,) + shape + extra_dims_sizes
        detector_shape = detector.values.shape
        self.assertEqual(true_shape, detector_shape)


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

