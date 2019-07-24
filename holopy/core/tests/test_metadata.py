import unittest

import numpy as np
from nose.plugins.attrib import attr

from holopy.core.metadata import (
    detector_grid, detector_points, clean_concat, update_metadata,
    get_spacing, get_extents, copy_metadata, make_subset_data)


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

