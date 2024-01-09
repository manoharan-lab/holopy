import unittest
import numpy as np

import pytest

from holopy.scattering import Sphere, Spheres

class TestBasicMethods(unittest.TestCase):
    @pytest.mark.fast
    def test_getattr(self):
        spheres = [Sphere(n=np.random.rand(), r=np.random.rand()/2,
                          center=[i, i, i]) for i in range(3)]
        spheres = Spheres(spheres)
        self.assertEqual(spheres[1], spheres.scatterers[1])

    @pytest.mark.fast
    def test_from_parameters_keeps_attributes(self):
        spheres = [Sphere(n=np.random.rand(), r=0.1,
                          center=[i, i, i]) for i in range(3)]
        spheres = Spheres(spheres, warn="TEST")
        spheres = spheres.from_parameters(spheres.parameters)
        self.assertEqual(spheres.warn, "TEST")


if __name__ == '__main__':
    unittest.main()

