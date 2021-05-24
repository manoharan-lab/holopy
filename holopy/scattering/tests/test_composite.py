import unittest
import tempfile

import numpy as np
from nose.plugins.attrib import attr

import holopy as hp
from holopy.scattering import Sphere, Spheres

class TestBasicMethods(unittest.TestCase):
    @attr("fast")
    def test_getattr(self):
        spheres = [Sphere(n=np.random.rand(), r=np.random.rand(),
                          center=[i, i, i]) for i in range(3)]
        spheres = Spheres(spheres)
        self.assertEqual(spheres[1], spheres.scatterers[1])

    @attr("fast")
    def test_from_parameters_keeps_attributes(self):
        spheres = [Sphere(n=np.random.rand(), r=0.1,
                          center=[i, i, i]) for i in range(3)]
        spheres = Spheres(spheres, warn="TEST")
        spheres = spheres.from_parameters(spheres.parameters)
        self.assertEqual(spheres.warn, "TEST")


if __name__ == '__main__':
    unittest.main()

