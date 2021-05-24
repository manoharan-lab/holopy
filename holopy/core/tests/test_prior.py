# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, Anna Wang, Solomon Barkley
#
# This file is part of HoloPy.
#
# HoloPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HoloPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HoloPy.  If not, see <http://www.gnu.org/licenses/>.


import unittest

from scipy.stats import kstest
from numpy.testing import assert_equal, assert_allclose
import numpy as np
from nose.plugins.attrib import attr

from holopy.core.prior import (
    Prior, Gaussian, Uniform, BoundedGaussian, ComplexPrior, TransformedPrior,
    make_center_priors, updated, generate_guess)
from holopy.inference.result import UncertainValue
from holopy.core.metadata import data_grid
from holopy.scattering.errors import ParameterSpecificationError
from holopy.core.mapping import Mapper

GOLD_SIGMA = -1.4189385332  # log(sqrt(0.5/pi))-1/2


class TestBasics(unittest.TestCase):
    @attr("fast")
    def test_cannot_instantiate_baseclass(self):
        self.assertRaises(NotImplementedError, Prior)

    @attr("fast")
    def test_renamed_changes_name(self):
        original = Uniform(0, 1, guess=0.7, name='old')
        renamed = original.renamed('new')
        self.assertEqual(renamed.name, 'new')

    @attr("fast")
    def test_renamed_changes_nothing_else(self):
        original = Uniform(0, 1, guess=0.7, name='old')
        renamed = original.renamed('new')
        original.name = None
        renamed.name = None
        self.assertEqual(original, renamed)

    @attr("fast")
    def test_renamed_creates_new_object(self):
        original = Uniform(0, 1, guess=0.7, name='old')
        renamed = original.renamed('new')
        self.assertEqual(original.name, 'old')
        self.assertFalse(renamed == original)


class TestUniform(unittest.TestCase):
    @attr("fast")
    def test_construction_when_relying_argument_order(self):
        parameters = dict([
            ('lower_bound', 1),
            ('upper_bound', 3),
            ('guess', 2),
            ('name', 'a')])
        u = Uniform(*parameters.values())
        self.assertTrue(isinstance(u, Prior))
        for key, val in parameters.items():
            self.assertEqual(getattr(u, key), val)

    @attr("fast")
    def test_upper_bound_larger_than_lower_bound(self):
        self.assertRaises(ParameterSpecificationError, Uniform, 1, 0)
        self.assertRaises(ParameterSpecificationError, Uniform, 1, 1)

    @attr("fast")
    def test_guess_must_be_in_interval(self):
        self.assertRaises(ParameterSpecificationError, Uniform, 0, 1, 2)
        self.assertRaises(ParameterSpecificationError, Uniform, 0, 1, -1)

    @attr("fast")
    def test_interval_calculation(self):
        bounds = np.random.rand(2) + np.array([0, 1])
        u = Uniform(*bounds)
        self.assertEqual(u.interval, np.diff(bounds))

    @attr("fast")
    def test_interval_is_property(self):
        bounds = np.random.rand(2) + np.array([0, 1])
        u = Uniform(*bounds)
        self.assertRaises(AttributeError, setattr, u, 'interval', 2)

    @attr("fast")
    def test_prob(self):
        bounds = np.random.rand(2) + np.array([0, 1])
        u = Uniform(*bounds)
        self.assertEqual(u.prob(0), 0)
        self.assertEqual(u.prob(2), 0)
        self.assertAlmostEqual(u.prob(1), 1/np.diff(bounds))

    @attr("fast")
    def test_lnprob(self):
        bounds = np.random.rand(2) + np.array([0, 1])
        u = Uniform(*bounds)
        self.assertEqual(u.lnprob(0), -np.inf)
        self.assertEqual(u.lnprob(2), -np.inf)
        self.assertTrue(np.allclose(u.lnprob(1), -np.log(np.diff(bounds))))

    @attr("fast")
    def test_sample_shape(self):
        n_samples = 7
        bounds = np.random.rand(2) + np.array([0, 1])
        u = Uniform(*bounds)
        samples = u.sample(n_samples)
        self.assertEqual(samples.shape, np.array(n_samples))
        self.assertTrue(np.all(samples > bounds[0]))
        self.assertTrue(np.all(samples < bounds[1]))

    @attr("medium")
    def test_sample_distribution(self):
        n_samples = 1000
        np.random.seed(805)
        bounds = np.random.rand(2) + np.array([0, 1])
        u = Uniform(*bounds)
        samples = u.sample(n_samples)
        p_val = kstest(samples, 'uniform', (bounds[0], np.diff(bounds)))[1]
        self.assertTrue(p_val > 0.05)

    @attr("fast")
    def test_auto_guess(self):
        bounds = np.random.rand(2) + np.array([0, 1])
        u = Uniform(*bounds)
        self.assertEqual(u.guess, np.mean(bounds))

    @attr("fast")
    def test_auto_guess_improper(self):
        bound = np.random.rand()
        u = Uniform(bound, np.inf)
        self.assertEqual(u.guess, bound)
        u = Uniform(-np.inf, bound)
        self.assertEqual(u.guess, bound)
        u = Uniform(-np.inf, np.inf)
        self.assertEqual(u.guess, 0)
        u = Uniform(-np.inf, np.inf, bound)
        self.assertEqual(u.guess, bound)

    @attr("fast")
    def test_improper_prob(self):
        bound = np.random.rand()
        u = Uniform(-np.inf, bound)
        self.assertEqual(u.interval, np.inf)
        self.assertEqual(u.prob(bound-1), 0)
        self.assertEqual(u.lnprob(bound-1), -1e6)


class TestGaussian(unittest.TestCase):
    @attr("fast")
    def test_construction(self):
        parameters = {'mu': 1, 'sd': 3, 'name': 'a'}
        g = Gaussian(**parameters)
        self.assertTrue(isinstance(g, Prior))
        for key, val in parameters.items():
            self.assertEqual(getattr(g, key), val)

    @attr("fast")
    def test_sd_is_positive(self):
        self.assertRaises(ParameterSpecificationError, Gaussian, 1, 0)
        self.assertRaises(ParameterSpecificationError, Gaussian, 1, -1)

    @attr("fast")
    def test_variance(self):
        sd = np.random.rand()
        g = Gaussian(0, sd)
        self.assertEqual(g.variance, sd**2)
        self.assertRaises(AttributeError, setattr, g, 'variance', 2) # property

    @attr("fast")
    def test_guess(self):
        mean = np.random.rand()
        g = Gaussian(mean, 1)
        self.assertEqual(g.guess, mean)
        self.assertRaises(AttributeError, setattr, g, 'guess', 2)  # property

    @attr("fast")
    def test_prob(self):
        np.random.seed(992)
        mean, sd = np.random.rand(2)
        g = Gaussian(mean, sd)
        norm = 1/np.sqrt(2 * np.pi * sd**2)
        self.assertTrue(np.allclose(g.prob(mean), norm))
        self.assertTrue(np.allclose(g.prob(mean+sd), np.exp(-1/2) * norm))

    @attr("fast")
    def test_lnprob(self):
        g = Gaussian(0, 1)
        self.assertTrue(np.allclose(g.lnprob(1), GOLD_SIGMA))
        g = Gaussian(0, 2)
        self.assertTrue(np.allclose(g.lnprob(2), GOLD_SIGMA - np.log(2)))

    @attr("medium")
    def test_sample(self):
        n_samples = 10000
        np.random.seed(37)
        mean, sd = np.random.rand(2)
        g = Gaussian(mean, sd)
        samples = g.sample(n_samples)
        mean_sd = sd/np.sqrt(n_samples)  # by central limit theorem
        self.assertTrue(np.allclose(mean, samples.mean(), atol=3*mean_sd))
        self.assertTrue(np.allclose(sd, samples.std(ddof=1), atol=0.01))


class TestBoundedGaussian(unittest.TestCase):
    @attr("fast")
    def test_construction(self):
        parameters = {'mu': 1, 'sd': 3,
                      'lower_bound': 0, 'upper_bound': 2, 'name': 'a'}
        bg = BoundedGaussian(**parameters)
        self.assertTrue(isinstance(bg, Gaussian))
        for key, val in parameters.items():
            self.assertEqual(getattr(bg, key), val)

    @attr("fast")
    def test_bound_constraints(self):
        for bounds in [(-1, 0), (2, 3), (1, 1)]:
            self.assertRaises(
                ParameterSpecificationError, BoundedGaussian, 1, 1, *bounds)

    @attr("fast")
    def test_lnprob(self):
        mean, sd = np.random.rand(2)
        g = Gaussian(mean, sd)
        bg = BoundedGaussian(mean, sd, -1, 2)
        self.assertEqual(bg.lnprob(1), g.lnprob(1))
        self.assertEqual(bg.lnprob(-1), g.lnprob(-1))
        self.assertEqual(bg.lnprob(2), g.lnprob(2))
        self.assertEqual(bg.lnprob(-2), -np.inf)
        self.assertEqual(bg.lnprob(3), -np.inf)

    @attr("fast")
    def test_prob(self):
        mean, sd = np.random.rand(2)
        g = Gaussian(mean, sd)
        bg = BoundedGaussian(mean, sd, -1, 2)
        self.assertEqual(bg.prob(1), g.prob(1))
        self.assertEqual(bg.prob(-1), g.prob(-1))
        self.assertEqual(bg.prob(2), g.prob(2))
        self.assertEqual(bg.prob(-2), 0)
        self.assertEqual(bg.prob(3), 0)

    @attr("fast")
    def test_sample(self):
        n_samples = 1000
        bound = 0.1
        np.random.seed(1502)
        bg = BoundedGaussian(0, 1, -bound, bound)
        samples = bg.sample(n_samples)
        self.assertTrue(np.all(samples > -bound) and np.all(samples < bound))


class TestComplexPrior(unittest.TestCase):
    @attr("fast")
    def test_construction(self):
        parameters = {'real': Uniform(1, 2), 'imag': 3, 'name': 'a'}
        cp = ComplexPrior(**parameters)
        self.assertTrue(isinstance(cp, Prior))
        for key, val in parameters.items():
            self.assertEqual(getattr(cp, key), val)

    @attr("fast")
    def test_guess_2_priors(self):
        real = Uniform(*(np.random.rand(2) + np.array([0, 1])))
        imag = Uniform(*(np.random.rand(2) + np.array([0, 1])))
        cp = ComplexPrior(real, imag)
        self.assertEqual(cp.guess, real.guess + 1.0j * imag.guess)

    @attr("fast")
    def test_guess_fixed_imag(self):
        real = Uniform(*(np.random.rand(2) + np.array([0, 1])))
        imag = np.random.rand()
        cp = ComplexPrior(real, imag)
        self.assertEqual(cp.guess, real.guess + 1.0j * imag)

    @attr("fast")
    def test_guess_fixed_real(self):
        real = np.random.rand()
        imag = Uniform(*(np.random.rand(2) + np.array([0, 1])))
        cp = ComplexPrior(real, imag)
        self.assertEqual(cp.guess, real + 1.0j * imag.guess)

    @attr("fast")
    def test_lnprob_2_priors(self):
        real = Uniform(*np.random.rand(2) + np.array([0, 1]))
        imag = Uniform(*np.random.rand(2) + np.array([0, 1]))
        cp = ComplexPrior(real, imag)
        self.assertEqual(cp.lnprob(1), -np.inf)
        self.assertEqual(cp.lnprob(1.0j), -np.inf)
        self.assertEqual(cp.lnprob(1 + 1.0j), real.lnprob(1) + imag.lnprob(1))

    @attr("fast")
    def test_lnprob_fixed_imag(self):
        real = Uniform(*np.random.rand(2) + np.array([0, 1]))
        imag = np.random.rand()
        cp = ComplexPrior(real, imag)
        self.assertEqual(cp.lnprob(1), real.lnprob(1))
        self.assertEqual(cp.lnprob(1.0j), -np.inf)
        self.assertEqual(cp.lnprob(1 + 1.0j), real.lnprob(1))

    @attr("fast")
    def test_lnprob_fixed_real(self):
        real = np.random.rand()
        imag = Uniform(*(np.random.rand(2) + np.array([0, 1])))
        cp = ComplexPrior(real, imag)
        self.assertEqual(cp.lnprob(1), -np.inf)
        self.assertEqual(cp.lnprob(1.0j), imag.lnprob(1))
        self.assertEqual(cp.lnprob(1 + 1.0j), imag.lnprob(1))

    @attr("fast")
    def test_prob_2_priors(self):
        real = Uniform(*(np.random.rand(2) + np.array([0, 1])))
        imag = Uniform(*(np.random.rand(2) + np.array([0, 1])))
        cp = ComplexPrior(real, imag)
        self.assertEqual(cp.prob(1), 0)
        self.assertEqual(cp.prob(1.0j), 0)
        self.assertAlmostEqual(cp.prob(1 + 1.0j), real.prob(1) * imag.prob(1))

    @attr("fast")
    def test_sample_2_priors(self):
        n_samples = 10
        real = Uniform(*(np.random.rand(2) + np.array([0, 1])))
        imag = Uniform(*(np.random.rand(2) + np.array([0, 1])))
        cp = ComplexPrior(real, imag)
        samples = cp.sample(n_samples)
        self.assertTrue(np.all(samples.real < real.upper_bound))
        self.assertTrue(np.all(samples.real > real.lower_bound))
        self.assertTrue(np.all(samples.imag < imag.upper_bound))
        self.assertTrue(np.all(samples.imag > imag.lower_bound))

    @attr("fast")
    def test_sample_fixed_real(self):
        n_samples = 10
        real = np.random.rand()
        imag = Uniform(*(np.random.rand(2) + np.array([0, 1])))
        cp = ComplexPrior(real, imag)
        samples = cp.sample(n_samples)
        self.assertTrue(np.all(samples.real == real))
        self.assertTrue(np.all(samples.imag < imag.upper_bound))
        self.assertTrue(np.all(samples.imag > imag.lower_bound))

    @attr("fast")
    def test_sample_fixed_imag(self):
        n_samples = 10
        real = Uniform(*(np.random.rand(2) + np.array([0, 1])))
        imag = np.random.rand()
        cp = ComplexPrior(real, imag)
        samples = cp.sample(n_samples)
        self.assertTrue(np.all(samples.real < real.upper_bound))
        self.assertTrue(np.all(samples.real > real.lower_bound))
        self.assertTrue(np.all(samples.imag == imag))


class TestTransformedPrior(unittest.TestCase):
    @attr("fast")
    def test_single_base_prior_becomes_tuple(self):
        base_prior = Uniform(0, 2)
        transformed = TransformedPrior(np.sqrt, base_prior)
        self.assertEqual(transformed.base_prior, (base_prior,))

    @attr('fast')
    def test_transformation_must_be_function(self):
        transform = Uniform(0, 2)
        base_prior = Uniform(1, 3)
        self.assertRaises(TypeError, TransformedPrior, transform, base_prior)

    @attr('fast')
    def test_sample_single_prior_once(self):
        transformed = TransformedPrior(np.sqrt, Uniform(81, 100))
        sample = transformed.sample()
        self.assertTrue(9 < sample < 10)

    @attr('fast')
    def test_sample_single_prior_with_size(self):
        transformed = TransformedPrior(np.sqrt, Uniform(81, 100))
        sample = transformed.sample(100)
        self.assertEqual(len(sample), 100)
        self.assertTrue(isinstance(sample, np.ndarray))
        self.assertTrue((9 < sample).all() and (sample < 10).all())
        self.assertNotEqual(sample[0], sample[1])

    @attr('fast')
    def test_sample_multiple_priors_once(self):
        base_prior = [Uniform(0, 1), Uniform(2, 3)]
        transformed = TransformedPrior(np.maximum, base_prior)
        sample = transformed.sample()
        self.assertTrue(2 < sample < 3)

    @attr('fast')
    def test_sample_multiple_priors_with_size(self):
        base_prior = [Uniform(0, 1), Uniform(2, 3)]
        transformed = TransformedPrior(np.maximum, base_prior)
        sample = transformed.sample(100)
        self.assertEqual(len(sample), 100)
        self.assertTrue(isinstance(sample, np.ndarray))
        self.assertTrue((2 < sample).all() and (sample < 3).all())
        self.assertNotEqual(sample[0], sample[1])

    @attr("fast")
    def test_guess_with_single_prior(self):
        base_prior = Uniform(0, 10, guess=4)
        transformed = TransformedPrior(np.sqrt, base_prior)
        self.assertEqual(transformed.guess, 2)

    @attr('fast')
    def test_guess_with_multiple_priors(self):
        base_priors = [Uniform(0, 10, guess=4), Uniform(0, 2, guess=1)]
        transformed = TransformedPrior(np.maximum, base_priors)
        self.assertEqual(transformed.guess, 4)

    @attr('fast')
    def test_hierarchical_transformed_prior(self):
        base_prior = Uniform(10, 20, guess=16)
        transform = TransformedPrior(np.sqrt, base_prior)
        double = TransformedPrior(np.sqrt, transform)
        samples = double.sample(100)
        self.assertEqual(double.guess, 2)
        self.assertTrue(all(samples < np.sqrt(np.sqrt(20))) and
                        all(samples > np.sqrt(np.sqrt(10))))

    @attr('fast')
    def test_transformation_by_constant(self):
        base_prior = [Uniform(0, 5, guess=3), 2]
        transform = TransformedPrior(np.maximum, base_prior)
        samples = transform.sample(100)
        self.assertEqual(transform.guess, 3)
        self.assertTrue(all(samples < 5))
        self.assertTrue(all(samples >= 2))

    @attr('fast')
    def test_map_keys_indexing(self):
        base_priors = [Uniform(0, 10, guess=4), Uniform(0, 2, guess=1)]
        transformed = TransformedPrior(np.maximum, base_priors)
        mapper = Mapper()
        mapper.iterate_mapping('root', transformed.map_keys)
        self.assertEqual(mapper.parameter_names[-2:], ['root0', 'root1'])

    @attr('fast')
    def test_map_keys_single_prior(self):
        transformed = TransformedPrior(np.sqrt, Uniform(0, 2))
        mapper = Mapper()
        mapper.iterate_mapping('root', transformed.map_keys)
        self.assertEqual(mapper.parameter_names[-1], 'root')

def test_scale_factor():
    p1 = Gaussian(3, 1)
    assert_equal(p1.scale_factor, 3)
    p2 = Gaussian(0, 2)
    assert_equal(p2.scale_factor, 2)
    p4 = Uniform(-1, 1, 0)
    assert_equal(p4.scale_factor, 0.2)
    p5 = Uniform(1, 4)
    assert_equal(p5.scale_factor, 2.5)
    p6 = Uniform(0, np.inf)
    assert_equal(p6.scale_factor, 1)
    assert_equal(p2.scale(10), 5)
    assert_equal(p2.unscale(5), 10)


def test_updated():
    p = BoundedGaussian(1, 2, -1, 2)
    d = UncertainValue(1, 0.5, 1)
    u = updated(p, d)
    assert_equal(u.guess, 1)
    assert_allclose(u.lnprob(0), GOLD_SIGMA)


class TestPriorMath(unittest.TestCase):
    @property
    def u(self):
        return Uniform(1, 2)
    @property
    def g(self):
        return Gaussian(1, 2)
    @property
    def b(self):
        return BoundedGaussian(1, 2, 0, 3)
    @property
    def c(self):
        return ComplexPrior(self.u, self.g)

    @attr("fast")
    def test_my_properties(self):
        self.assertEqual(self.u, Uniform(1, 2))
        self.assertEqual(self.g, Gaussian(1, 2))
        self.assertEqual(self.b, BoundedGaussian(1, 2, 0, 3))
        self.assertEqual(self.c, ComplexPrior(self.u, self.g))

    @attr("fast")
    def test_no_zero_multiplication(self):
        with self.assertRaises(TypeError):
            self.u * 0

    @attr("fast")
    def test_guess_is_adjusted(self):
        u = Uniform(0, 1, 1)
        self.assertEqual((u + 1).guess, 2)
        self.assertEqual((u * 2).guess, 2)
        self.assertEqual((-u).guess, -1)

    @attr("fast")
    def test_addition_identities(self):
        self.assertEqual(1 + self.u, self.u + 1)
        self.assertEqual(self.u - 1, self.u + (-1))

    @attr('fast')
    def test_addition_sampling(self):
        self.assertTrue(2 < (1 + self.u).sample() < 3)

    @attr("fast")
    def test_multiplication_identities(self):
        self.assertEqual(2 * self.u, self.u * 2)
        self.assertEqual(self.u / 2, self.u * 0.5)
        self.assertEqual(-1 * self.u, -self.u)

    @attr('fast')
    def test_multiplication_sampling(self):
        self.assertTrue(2 < (2 * self.u).sample() < 4)

    @attr('fast')
    def test_rdiv(self):
        uniform = Uniform(1, 4, guess=3)
        reciprocal = 1 / uniform
        samples = reciprocal.sample(100)
        self.assertEqual(reciprocal.guess, 1/uniform.guess)
        self.assertEqual(2 * reciprocal, 2/uniform)
        self.assertTrue(all(samples < 1) and all(samples > 1/4))

    @attr('fast')
    def test_add_two_priors(self):
        composite = self.u + self.g
        self.assertEqual(composite.guess, self.u.guess + self.g.guess)

    @attr('fast')
    def test_multiply_two_priors(self):
        composite = self.u * self.g
        self.assertEqual(composite.guess, self.u.guess * self.g.guess)

    @attr('fast')
    def test_subtract_two_priors(self):
        composite = self.u - self.g
        self.assertEqual(composite.guess, self.u.guess - self.g.guess)

    @attr('fast')
    def test_divide_two_priors(self):
        composite = self.u / self.g
        self.assertEqual(composite.guess, self.u.guess / self.g.guess)

    @attr("fast")
    def test_prior_array_math(self):
        sum_array = self.g + np.array([0, 1])
        product_array = self.g * np.array([1, 2])
        for arr in [sum_array, product_array]:
            self.assertTrue(isinstance(arr, np.ndarray))
            self.assertTrue(all([isinstance(p, Prior) for p in arr]))

    @attr("fast")
    def test_single_arg_numpy_ufunc(self):
        base_prior = Uniform(2, 3)
        transformed = TransformedPrior(np.sqrt, base_prior)
        self.assertEqual(np.sqrt(base_prior), transformed)

    @attr("fast")
    def test_two_arg_numpy_ufunc_with_const(self):
        base_prior = Uniform(2, 4)
        transformed = TransformedPrior(np.maximum, [base_prior, 3])
        self.assertEqual(np.maximum(base_prior, 3), transformed)

    @attr("fast")
    def test_two_arg_numpy_ufunc_with_both_priors(self):
        prior_1 = Uniform(2, 4)
        prior_2 = Uniform(1, 3)
        transformed = TransformedPrior(np.maximum, [prior_1, prior_2])
        self.assertEqual(np.maximum(prior_1, prior_2), transformed)

    @attr("fast")
    def test_numpy_ufunc_passes_through_name(self):
        prior_1 = Uniform(2, 8, name='unused')
        new_name = 'name_from_numpy'
        transformed = np.sqrt(prior_1, name=new_name)
        self.assertEqual(transformed.name, new_name)

    @attr('fast')
    def test_pow_rpow(self):
        prior = Uniform(1, 4, guess=3)
        power = prior ** 2
        rpower = 2 ** prior
        self.assertEqual(power.guess, 9)
        self.assertEqual(rpower.guess, 8)


def test_generate_guess():
    gold1 = np.array([[-0.091949, 0.270532], [-1.463350, 0.691041],
        [1.081791, 0.220404], [-0.239325, 0.811950], [-0.491129, 0.010526]])
    gold2 = np.array([[-0.045974, 0.535266], [-0.731675, 0.745520],
        [0.540895, 0.510202], [-0.119662, 0.805975], [-0.245564, 0.405263]])
    pars = [Gaussian(0, 1), Uniform(0, 1, 0.8)]
    guess1 = generate_guess(pars, 5, seed=22)
    guess2 = generate_guess(pars, 5, scaling=0.5, seed=22)
    assert_allclose(guess1, gold1, atol=1e-5)
    assert_allclose(guess2, gold2, atol=1e-5)


class TestMakeCenterPriors(unittest.TestCase):
    @property
    def image(self):
        img = np.zeros([4, 4])
        img[:3, 1:] = np.pad(np.zeros([1,1]), 1, 'constant', constant_values=1)
        return data_grid(img, spacing=2)

    @attr("fast")
    def test_my_image(self):
        self.assertTrue(np.allclose(self.image.values, np.array(
            [[0, 1, 1, 1], [0, 1, 0, 1], [0, 1, 1, 1], [0, 0, 0, 0]])))

    @attr("fast")
    def test_basic(self):
        expected = [Gaussian(0, 2), Gaussian(2, 2), Uniform(0, 40)]
        evaluated = make_center_priors(self.image)
        self.assertEqual(evaluated, expected)

    @attr("fast")
    def test_z_range_extents(self):
        expected = [Gaussian(0, 2), Gaussian(2, 2), Uniform(0, 16)]
        evaluated = make_center_priors(self.image, z_range_extents=2)
        self.assertEqual(evaluated, expected)

    @attr("fast")
    def test_xy_uncertainty(self):
        expected = [Gaussian(0, 4), Gaussian(2, 4), Uniform(0, 40)]
        evaluated = make_center_priors(self.image, xy_uncertainty_pixels=2)
        self.assertEqual(evaluated, expected)

    @attr("fast")
    def test_z_range_units(self):
        expected = [Gaussian(0, 2), Gaussian(2, 2), Uniform(2, 10)]
        evaluated = make_center_priors(self.image, z_range_units=(2, 10))
        self.assertEqual(evaluated, expected)


if __name__ == '__main__':
    unittest.main()

