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


import warnings

import numpy as np
from numpy.testing import assert_equal, assert_raises, assert_allclose
from nose.plugins.attrib import attr

from holopy.scattering import Sphere, Spheres, calc_holo
from holopy.core import detector_grid
from holopy.core.tests.common import assert_obj_close
from holopy.scattering.errors import ParameterSpecificationError
from holopy.inference import NmpfitStrategy as Nmpfit
from holopy.inference import prior, AlphaModel


@attr('fast')
def test_minimizer():
    x = np.arange(-10, 10, .1)
    a = 5.3
    b = -1.8
    c = 3.4
    gold_list = [a, b, c]
    y = a*x**2 + b*x + c

    # This test does NOT handle scaling correctly -- we would need a Model
    # which knows the parameters to properly handle the scaling/unscaling
    def cost_func(pars):
        a, b, c = pars
        return a*x**2 + b*x + c - y

    # test basic usage
    parameters = [prior.Uniform(-np.inf, np.inf, name='a', guess=5),
                  prior.Uniform(-np.inf, np.inf, name='b', guess=-2),
                  prior.Uniform(-np.inf, np.inf, name='c', guess=3)]
    minimizer = Nmpfit()
    result, minimization_details = minimizer.minimize(parameters, cost_func)
    assert_obj_close(result, gold_list, context='basic_minimized_parameters')

    # now test limiting minimizer iterations
    minimizer = Nmpfit(maxiter=1)
    try:
        result, minimization_details = minimizer.minimize(parameters, cost_func)
    except MinimizerConvergenceFailed as cf: # the fit shouldn't converge
        result, minimization_details = cf.result, cf.details
    assert_equal(minimization_details.niter, 2) # there's always an offset of 1


@attr('slow')
def test_optimization_with_maxiter_of_2():
    gold_fit_dict = {
        '0:r': 0.52480509800531849,
        '1:center.1': 14.003687569304704,
        'alpha': 0.93045027963762217,
        '0:center.2': 19.93177549652841,
        '1:r': 0.56292664494653732,
        '0:center.1': 15.000340621607815,
        '1:center.0': 14.020984607646726,
        '0:center.0': 15.000222185576494,
        '1:center.2': 20.115613202192328}

    #calculate a hologram with known particle positions to do a fit against
    schema = detector_grid(shape = 100, spacing = .1)

    s1 = Sphere(center=(15, 15, 20), n = 1.59, r = 0.5)
    s2 = Sphere(center=(14, 14, 20), n = 1.59, r = 0.5)
    cluster = Spheres([s1, s2])
    holo = calc_holo(schema, cluster, 1.33, .66, illum_polarization=(1,0))

    #trying to do a fast fit:
    guess1 = Sphere(
        center=(prior.Uniform(5, 25, guess=15),
                prior.Uniform(5, 25, 15),
                prior.Uniform(5, 25, 20)),
        r=(prior.Uniform(.4, .6, guess=.45)),
        n=1.59)
    guess2 = Sphere(
        center=(prior.Uniform(5, 25, guess=14),
                prior.Uniform(5, 25, 14),
                prior.Uniform(5, 25, 20)),
        r=(prior.Uniform(.4, .6, guess = .45)),
        n=1.59)
    par_s = Spheres([guess1, guess2])

    model = AlphaModel(
        par_s, medium_index=1.33, illum_wavelen=.66, illum_polarization=(1, 0),
        alpha=prior.Uniform(.1, 1, .6))
    optimizer = Nmpfit(maxiter=2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = optimizer.fit(model, holo)
    assert_obj_close(gold_fit_dict, result.parameters, rtol=1e-5)


# TODO: There could be a test that the optimizer raises a warning when
# `maxiter` iterations have passed.

