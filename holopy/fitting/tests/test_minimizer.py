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
from ...scattering.scatterer import Sphere, Spheres
from ...core import detector_grid
from .. import fit, Parameter, par, Model
from ..minimizer import Nmpfit
from ..errors import ParameterSpecificationError, MinimizerConvergenceFailed
from ...core.tests.common import assert_obj_close
from holopy.scattering.calculations import calc_holo

def test_minimizer():
    x = np.arange(-10, 10, .1)
    a = 5.3
    b = -1.8
    c = 3.4
    gold_dict = dict((('a', a), ('b', b), ('c', c)))
    y = a*x**2 + b*x + c

    # This test does NOT handle scaling correctly -- we would need a Model
    # which knows the parameters to properly handle the scaling/unscaling
    def cost_func(pars):
        a = pars['a']
        b = pars['b']
        c = pars['c']
        return a*x**2 + b*x + c - y

    # test basic usage
    parameters = [Parameter(name='a', guess = 5),
                 Parameter(name='b', guess = -2),
                 Parameter(name='c', guess = 3)]
    minimizer = Nmpfit()
    result, minimization_details = minimizer.minimize(parameters, cost_func)
    assert_obj_close(gold_dict, result, context = 'basic_minimized_parameters')

    # test inadequate specification
    with assert_raises(ParameterSpecificationError):
        minimizer.minimize([Parameter(name = 'a')], cost_func)

    # now test limiting minimizer iterations
    minimizer = Nmpfit(maxiter=1)
    try:
        result, minimization_details = minimizer.minimize(parameters, cost_func)
    except MinimizerConvergenceFailed as cf: # the fit shouldn't converge
        result, minimization_details = cf.result, cf.details
    assert_equal(minimization_details.niter, 2) # there's always an offset of 1

    # now test parinfo argument passing
    parameters2 = [Parameter(name='a', guess = 5, mpside = 2),
                   Parameter(name='b', guess = -2, limit = [-4, 4.]),
                   Parameter(name='c', guess = 3, step = 1e-4, mpmaxstep = 2.,
                             limit = [0., 12.])]
    minimizer = Nmpfit()
    result2, details2, parinfo = minimizer.minimize(parameters2, cost_func,
                                                    debug = True)
    assert_equal(parinfo[0]['mpside'], 2)
    assert_equal(parinfo[2]['limits'], np.array([0., 12.])/3.)
    assert_allclose(parinfo[2]['step'], 1e-4/3.)
    assert_equal(parinfo[2]['limited'], [True, True])
    assert_obj_close(gold_dict, result2, context = 'minimized_parameters_with_parinfo')

def test_iter_limit():
    gold_fit_dict={'0:Sphere.r': 0.52480509800531849, '1:Sphere.center[1]': 14.003687569304704, 'alpha': 0.93045027963762217, '0:Sphere.center[2]': 19.93177549652841, '1:Sphere.r': 0.56292664494653732, '0:Sphere.center[1]': 15.000340621607815, '1:Sphere.center[0]': 14.020984607646726, '0:Sphere.center[0]': 15.000222185576494, '1:Sphere.center[2]': 20.115613202192328}

    #calculate a hologram with known particle positions to do a fit against
    schema = detector_grid(shape = 100, spacing = .1, illum_wavelen = .660,
                         medium_index = 1.33, illum_polarization = (1,0))

    s1 = Sphere(center=(15, 15, 20), n = 1.59, r = 0.5)
    s2 = Sphere(center=(14, 14, 20), n = 1.59, r = 0.5)
    cluster = Spheres([s1, s2])
    holo = calc_holo(schema, cluster, 1.33, .66, illum_polarization=(1,0))

    #trying to do a fast fit:
    guess1 = Sphere(center = (par(guess = 15, limit = [5,25]), par(15, [5, 25]), par(20, [5, 25])), r = (par(guess = .45, limit=[.4,.6])), n = 1.59)
    guess2 = Sphere(center = (par(guess = 14, limit = [5,25]), par(14, [5, 25]), par(20, [5, 25])), r = (par(guess = .45, limit=[.4,.6])), n = 1.59)
    par_s = Spheres([guess1,guess2])

    model = Model(par_s, calc_holo, 1.33, .66, illum_polarization=(1, 0), alpha = par(.6, [.1, 1]))
    warnings.simplefilter("always")
    result = fit(model, holo, minimizer = Nmpfit(maxiter=2))
    assert_obj_close(gold_fit_dict,result.parameters,rtol=1e-6)
