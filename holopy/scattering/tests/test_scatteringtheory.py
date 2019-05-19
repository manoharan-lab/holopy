import unittest

import numpy as np
import xarray as xr
from numpy.testing import assert_allclose, assert_equal
from nose.plugins.attrib import attr

from holopy.core import detector_grid, detector_points
from holopy.core.metadata import sphere_coords, update_metadata
from holopy.scattering.theory.scatteringtheory import stack_spherical



class TestSphereCoords(unittest.TestCase):
    @attr("fast")
    def test_sphere_coords(self):
        detector = detector_grid(shape=(2, 2), spacing=0.1)
        spherical = sphere_coords(
            detector, wavevec=2*np.pi*1.33/.66, origin=(0, 0, 1))
        pos = stack_spherical(spherical).T
        true_pos = np.array([
            [ 12.66157039,   0.        ,   0.        ],
            [ 12.72472076,   0.09966865,   1.57079633],
            [ 12.72472076,   0.09966865,   0.        ],
            [ 12.78755927,   0.1404897 ,   0.78539816]])
        self.assertTrue(np.allclose(pos, true_pos))


if __name__ == '__main__':
    unittest.main()

