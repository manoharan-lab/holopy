# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca
# W. Perry, Jerome Fung, and Ryan McGorty
#
# This file is part of Holopy.
#
# Holopy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Holopy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Holopy.  If not, see <http://www.gnu.org/licenses/>.

'''
Proposal for new function structure for fitting in the form of tests.

Don't expect these tests to pass for a while

'''

import numpy as np
import holopy as hp
from numpy.testing import assert_array_almost_equal
from nose.tools import with_setup
import os
import string
from nose.plugins.attrib import attr

from holopy.model.scatterer import Sphere, SphereCluster
from holopy.analyze.fit import fit

def setup_optics():
    # set up optics class for use in several test functions
    global optics
    wavelen = 658e-9
    polarization = [0., 1.0]
    divergence = 0
    pixel_scale = [.1151e-6, .1151e-6]
    index = 1.33
    
    optics = hp.optics.Optics(wavelen=wavelen, index=index,
                              pixel_scale=pixel_scale,
                              polarization=polarization,
                                  divergence=divergence)
    
def teardown_optics():
    global optics
    del optics

gold_single = np.array([1.582, 1.000, 6.484, 5.534, 5.792, 1.415, 6.497])

@attr('medium')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_fit_mie_single():
    path = os.path.abspath(hp.__file__)
    path = os.path.join(os.path.split(path)[0],'tests', 'exampledata')
    holo = hp.process.normalize(hp.load(os.path.join(path, 'image0001.npy'),
                                        optics=optics))
    
    s = Sphere(n=1.59+1e-4j, r=8.5e-7, x=.567e-5, y=.576e-5, z=15e-6)
    alpha = .6
    lb = Sphere.make_from_parameter_list([1.0, 1e-4, 1e-8, 0., 0., 0.]), .1
    ub = Sphere.make_from_parameter_list([2.0, 1e-4, 1e-5, 1e-5, 1e-5, 1e-4]), 1.0

    fitresult = fit(holo, (s,alpha), hp.model.theory.Mie, 'nmpfit',
                    lb, ub)

    assert_array_almost_equal(fitresult * [1,10**4,10**7,
            10**6,10**6,10**5,10], gold_single, decimal=2)

@attr('medium')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_fit_mie_single_ralg():
    path = os.path.abspath(hp.__file__)
    path = os.path.join(os.path.split(path)[0],'tests', 'exampledata')
    holo = hp.process.normalize(hp.load(os.path.join(path, 'image0001.npy'),
                                        optics=optics))
    
    s = Sphere(n=1.59+1e-4j, r=8.5e-7, x=.567e-5, y=.576e-5, z=15e-6)
    alpha = .6
    lb = Sphere.make_from_parameter_list([1.0, 1e-4, 1e-8, 0., 0., 0.]), .1
    ub = Sphere.make_from_parameter_list([2.0, 1e-4, 1e-5, 1e-5, 1e-5, 1e-4]), 1.0

    fitresult = fit(holo, (s,alpha), hp.model.theory.Mie, 'ralg',
                    lb, ub, plot=False)

    assert_array_almost_equal(fitresult * [1,10**4,10**7,
            10**6,10**6,10**5,10], gold_single, decimal=2)

    
@attr('slow')
def test_fit_superposition():
    # Make a test hologram
    optics = hp.Optics(wavelen=6.58e-07, index=1.33, polarization=[0.0, 1.0],
                    divergence=0, pixel_size=None, train=None, mag=None,
                    pixel_scale=[2.302e-07, 2.302e-07])

    s1 = Sphere(n=1.5891+1e-4j, r = .65e-6, center=(1.56e-05, 1.44e-05, 15e-6))
    s2 = Sphere(n=1.5891+1e-4j, r = .65e-6, center=(3.42e-05, 3.17e-05, 10e-6))
    sc = SphereCluster([s1, s2])
    alpha = .629
    
    theory = hp.model.theory.Mie(imshape=200, optics=optics)

    holo = hp.process.normalize(theory.calc_holo(sc, alpha))

    # Now fit it
    s1 = Sphere(n=1.5891+1e-4j, r = .65e-6, center=(1.56e-05, 1.44e-05, 15e-6))
    s2 = Sphere(n=1.5891+1e-4j, r = .65e-6, center=(3.42e-05, 3.17e-05, 10e-6))
    sc = SphereCluster([s1, s2])
    alpha = .629
    
    lb1 = Sphere(1+1e-4j, 1e-8, 0, 0, 0)
    ub1 = Sphere(2+1e-4j, 1e-5, 1e-4, 1e-4, 1e-4)
    lb = SphereCluster([lb1, lb1]), .1
    ub = SphereCluster([ub1, ub1]), 1

    fitresult = fit(holo, (sc, alpha), hp.model.theory.Mie, 'nmpfit', lb, ub)

    gold = np.array([1.5891, 1.000, 6.500, 1.560, 1.440, 1.500, 1.5891, 1.000, 6.50,
                  3.420, 3.170, 1.000, 6.26])
    assert_array_almost_equal(fitresult * [1, 10**4, 10**7, 10**5, 10**5,
                                           10**5,1,10**4, 10**7, 10**5,10**5,
                                           10**5, 10], gold, decimal=2)
    
'''
def test_fit_cluster():
    path = os.path.abspath(hp.__file__)
    path = string.rstrip(path, chars='__init__.pyc')+'tests/exampledata/'
    holo = normalize(hp.load(path + image0002))

    sc = hp.model.scatterer.Cluster(
'''
    
