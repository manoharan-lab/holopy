import unittest
import tempfile

import numpy as np
from nose.plugins.attrib import attr

import holopy as hp
from holopy.scattering import Sphere, Spheres
from holopy.inference.prior import Uniform

class TestBasicMethods(unittest.TestCase):
    @attr("fast")
    def test_ties(self):
        n = Uniform(0, 1)
        r = Uniform(0, 1)
        spheres = [Sphere(n=n, r=r, center=[i, i, i]) for i in range(3)]
        spheres = Spheres(spheres)
        expected_ties = {'n': ['0:n', '1:n', '2:n'],
                         'r': ['0:r', '1:r', '2:r']}
        self.assertEqual(spheres.ties, expected_ties)

    @attr("fast")
    def test_reversed_ties(self):
        n = Uniform(0, 1)
        r = Uniform(0, 1)
        spheres = [Sphere(n=n, r=r, center=[i, i, i]) for i in range(3)]
        spheres = Spheres(spheres)
        expected_ties = {'0:n': 'n', '1:n': 'n', '2:n': 'n',
                         '0:r': 'r', '1:r': 'r', '2:r': 'r'}
        self.assertEqual(spheres._reversed_ties, expected_ties)

    @attr("fast")
    def test_all_ties(self):
        n = Uniform(0, 1)
        r = Uniform(0, 1)
        spheres = [Sphere(n=n, r=r, center=[i, i, i]) for i in range(3)]
        spheres = Spheres(spheres)
        expected_ties = ['0:n', '1:n', '2:n', '0:r', '1:r', '2:r']
        self.assertEqual(spheres._all_ties, expected_ties)

    @attr("fast")
    def test_raw_parameters(self):
        max_radius = np.sqrt(3) / 2.0
        spheres = [
            Sphere(n=np.random.rand(),
            r=np.random.rand() * max_radius,
            center=[i, i, i])
            for i in range(3)]
        spheres = Spheres(spheres)
        expected_keys = {'0:n', '1:n', '2:n', '0:r', '1:r', '2:r',
                         '0:center.0', '0:center.1', '0:center.2',
                         '1:center.0', '1:center.1', '1:center.2',
                         '2:center.0', '2:center.1', '2:center.2'}
        self.assertEqual(set(spheres.raw_parameters.keys()), expected_keys)

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

class TestBrokenTies(unittest.TestCase):
    @attr("fast")
    def test_unequal_ties(self):
        spheres = [Sphere(n=Uniform(0, i+1), r=Uniform(0, 1)) for i in range(3)]
        ties = {'r': ['0:r', '1:r', '2:r'], 'n': ['0:n', '1:n', '2:n']}
        msg = 'Tied parameters 1:n and 0:n are not equal but have values'
        self.assertRaisesRegex(ValueError, msg, Spheres, spheres, ties=ties)

    @attr("fast")
    def test_missing_tie(self):
        spheres = [Sphere(n=Uniform(0, 1), r=Uniform(0, 1)) for i in range(3)]
        ties = {'r': ['0:r', '1:r', '2:r'], 'n': ['0:n', '1:n', 'dummy_name']}
        msg = 'Tied parameter dummy_name not present in raw parameters'
        self.assertRaisesRegex(ValueError, msg, Spheres, spheres, ties=ties)

    @attr("fast")
    def test_parameters_checks_ties(self):
        spheres = [Sphere(n=Uniform(0, 1), r=Uniform(0, 1)) for i in range(3)]
        ties = {'r': ['0:r', '1:r', '2:r'], 'n': ['0:n', '1:n', '2:n']}
        spheres = Spheres(spheres, ties=ties)
        spheres.ties['n'].append('dummy_name')
        msg = 'Tied parameter dummy_name not present in raw parameters'
        with self.assertRaisesRegex(ValueError, msg):
            spheres.parameters


class TestTiedParameters(unittest.TestCase):
    @attr("fast")
    def test_tied_if_same_object(self):
        n = Uniform(0, 1)
        r = Uniform(0, 1)
        spheres = [Sphere(n=n, r=r, center=[i, i, i]) for i in range(3)]
        spheres = Spheres(spheres)
        expected_keys = {'n', 'r', '0:center.0', '0:center.1', '0:center.2',
                                   '1:center.0', '1:center.1', '1:center.2',
                                   '2:center.0', '2:center.1', '2:center.2'}
        self.assertEqual(set(spheres.parameters.keys()), expected_keys)

    @attr("fast")
    def test_not_tied_if_equal(self):
        spheres = [Sphere(n=Uniform(0, 1), r=Uniform(0, 1)) for i in range(3)]
        spheres = Spheres(spheres)
        expected_keys = {'0:n', '0:r', '1:n', '1:r', '2:n', '2:r'}
        self.assertEqual(set(spheres.parameters.keys()), expected_keys)

    @attr("fast")
    def test_fixed_values_not_tied(self):
        n = 1.5
        r = 0.5
        spheres = [Sphere(n=n, r=r) for i in range(3)]
        spheres = Spheres(spheres)
        expected_keys = {'0:n', '0:r', '1:n', '1:r', '2:n', '2:r'}
        self.assertEqual(set(spheres.parameters.keys()), expected_keys)

    @attr("fast")
    def test_manual_tie(self):
        spheres = [Sphere(n=Uniform(0, 1), r=Uniform(0, 1)) for i in range(3)]
        ties = {'r': ['0:r', '1:r', '2:r'], 'n': ['0:n', '1:n', '2:n']}
        spheres = Spheres(spheres, ties=ties)
        expected_keys = {'n', 'r'}
        self.assertEqual(set(spheres.parameters.keys()), expected_keys)

    @attr("fast")
    def test_add_tie(self):
        spheres = [Sphere(n=Uniform(0, 1), r=Uniform(0, 1)) for i in range(3)]
        spheres = Spheres(spheres)
        spheres.add_tie('0:r', '1:r')
        spheres.add_tie('r', '2:r')
        spheres.add_tie('0:n', '1:n')
        spheres.add_tie('0:n', '2:n')
        expected_keys = {'n', 'r'}
        self.assertEqual(set(spheres.parameters.keys()), expected_keys)

    @attr("fast")
    def test_adding_tied_scatterer(self):
        n = Uniform(0, 1)
        r = Uniform(0, 1)
        spheres = [Sphere(n=n, r=r, center=[i, i, i]) for i in range(2)]
        spheres = Spheres(spheres)
        spheres.add(Sphere(n=n, r=r, center=[2, 2, 2]))
        expected_keys = {'n', 'r', '0:center.0', '0:center.1', '0:center.2',
                                   '1:center.0', '1:center.1', '1:center.2',
                                   '2:center.0', '2:center.1', '2:center.2'}
        self.assertEqual(set(spheres.parameters.keys()), expected_keys)

    @attr("fast")
    def test_adding_newly_tied_scatterer(self):
        n = Uniform(0, 1)
        r = Uniform(0, 1)
        spheres = [Sphere(n=n, r=r, center=[1, 1, 1])]
        spheres = Spheres(spheres)
        spheres.add(Sphere(n=n, r=r, center=[2, 2, 2]))
        expected_keys = {'n', 'r', '0:center.0', '0:center.1', '0:center.2',
                                   '1:center.0', '1:center.1', '1:center.2'}
        self.assertEqual(set(spheres.parameters.keys()), expected_keys)

    @attr("fast")
    def test_adding_untied_scatterer(self):
        n = Uniform(0, 1)
        r = Uniform(0, 1)
        spheres = [Sphere(n=n, r=r, center=[i, i, i]) for i in range(2)]
        spheres = Spheres(spheres)
        spheres.add(Sphere(n=1, r=1, center=[2, 2, 2]))
        expected_keys = {'n', 'r', '0:center.0', '0:center.1', '0:center.2',
                         '1:center.0', '1:center.1', '1:center.2', '2:n',
                         '2:r', '2:center.0', '2:center.1', '2:center.2'}
        self.assertEqual(set(spheres.parameters.keys()), expected_keys)

    @attr("fast")
    def test_tied_parameter_naming(self):
        n = Uniform(0, 1)
        r = [Uniform(0, 1), Uniform(0,1)]
        spheres = [Sphere(n=n, r=r[i%2]) for i in range(4)]
        spheres = Spheres(spheres)
        expected_ties = {'n': ['0:n', '1:n', '2:n', '3:n'],
                         'r': ['0:r', '2:r'],'1:r': ['1:r', '3:r']}
        self.assertEqual(spheres.ties, expected_ties)

    @attr("fast")
    def test_from_parameters(self):
        n = Uniform(0, 1)
        r = Uniform(0, np.pi/2)
        spheres = [Sphere(n=n, r=r, center=[i, i, i]) for i in range(3)]
        spheres = Spheres(spheres)
        parameters = spheres.parameters
        self.assertEqual(spheres.guess, spheres.from_parameters(parameters))

    @attr("medium")
    def test_save_and_open(self):
        n = Uniform(0, 1)
        r = Uniform(0, np.pi/2)
        spheres = [Sphere(n=n, r=r, center=[i, i, i]) for i in range(3)]
        spheres = Spheres(spheres)
        expected_keys = spheres.parameters.keys()
        with tempfile.TemporaryDirectory() as tempdir:
            hp.save(tempdir + '/test.out', spheres)
            spheres = hp.load(tempdir + '/test.out')
        self.assertEqual(spheres.parameters.keys(), expected_keys)


if __name__ == '__main__':
    unittest.main()

