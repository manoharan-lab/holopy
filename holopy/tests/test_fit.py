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
from nose.tools import with_setup, assert_raises
import os
import string
from nose.plugins.attrib import attr

from scatterpy.scatterer import Sphere, SphereCluster
from scatterpy.errors import ScattererOverlap
import scatterpy
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

# TODO: This test is obseleted by checking for overlap in scatterer definition.
# Check if there is anything important in it, then remove. 
#@attr('fast')
#@with_setup(setup=setup_optics, teardown=teardown_optics)
#def test_overlap_rejection():
#    path = os.path.abspath(hp.__file__)
#    path = os.path.join(os.path.split(path)[0],'tests', 'exampledata')
#    holo = hp.process.normalize(hp.load(os.path.join(path, 'image0001.npy'),
#                                        optics=optics))
#    
#    sc = SphereCluster([Sphere(n=1.59+1e-4j, r=8.5e-7, x=.567e-5, y=.576e-5,
#                               z=15e-6),
#                        Sphere(n=1.59+1e-4j, r=8.5e-7, x=.667e-5, y=.576e-5,
#                               z=15e-6)]), .6
#    lb = SphereCluster([Sphere(n=1.59+1e-4j, r=8.5e-7, x=.56e-5, y=.57e-5,
#                               z=10e-6),
#                        Sphere(n=1.59+1e-4j, r=8.5e-7, x=.66e-5, y=.57e-5,
#                               z=10e-6)]), .1
#    ub = SphereCluster([Sphere(n=1.59+1e-4j, r=8.5e-7, x=.57e-5, y=.58e-5,
#                               z=20e-6),
#                        Sphere(n=1.59+1e-4j, r=8.5e-7, x=.67e-5, y=.58e-5,
#                               z=20e-6)]), 1
#
#    assert_raises(ScattererOverlap, lambda: fit(holo, sc, scatterpy.theory.Mie,
#                                                'nmpfit', lb, ub))

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

    fitresult = fit(holo, (s,alpha), scatterpy.theory.Mie, 'nmpfit',
                    lb, ub)

    assert_array_almost_equal(np.concatenate((fitresult[0].parameter_list, 
                                             np.array([fitresult[1]]))) * 
                                             np.array([1,10**4,10**7,
                                                       10**6,10**6,10**5,10]), 
                                             gold_single, decimal=2)


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

    fitresult = fit(holo, (s,alpha), scatterpy.theory.Mie, 'ralg',
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
    
    theory = scatterpy.theory.Mie(imshape=200, optics=optics)

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

    fitresult = fit(holo, (sc, alpha), theory, 'nmpfit', lb, ub)

    fit_sc = fitresult[0]
    fit_alpha = fitresult[1]
    fitres_unpacked = np.array([fit_sc.n[0].real, fit_sc.n[0].imag, 
                                fit_sc.r[0], fit_sc.x[0], fit_sc.y[0],
                                fit_sc.z[0], fit_sc.n[1].real, fit_sc.n[1].imag,
                                fit_sc.r[1], fit_sc.x[1], fit_sc.y[1], 
                                fit_sc.z[1], fit_alpha])

    gold = np.array([1.5891, 1.000, 6.500, 1.560, 1.440, 1.500, 1.5891, 1.000, 6.50,
                  3.420, 3.170, 1.000, 6.29])
    assert_array_almost_equal(fitres_unpacked * [1, 10**4, 10**7, 10**5, 10**5,
                                           10**5,1,10**4, 10**7, 10**5,10**5,
                                           10**5, 10], gold, decimal=2)


@attr('slow')
def test_fit_multisphere_noisydimer_slow():
    optics = hp.Optics(wavelen=658e-9, polarization = [0., 1.0], 
                       divergence = 0., pixel_scale = [0.345e-6, 0.345e-6], 
                       index = 1.334)

    path = os.path.abspath(hp.__file__)
    path = os.path.join(os.path.split(path)[0],'tests', 'exampledata')
    holo = hp.process.normalize(hp.load(os.path.join(path, 'image0002.npy'),
                                        optics=optics))
    
    # gold results
    gold = np.array([1.603, 1.000, 6.857, 1.642, 1.725, 2.058, 1.603, 1.000, 
                     6.964, 1.758, 1.753, 2.127, 1.000])
    
    # initial guess
    s1 = Sphere(n=1.6026+1e-5j, r = .6856e-6, center=(1.64155e-05, 1.7247e-05, 20.582e-6))
    s2 = Sphere(n=1.6026+1e-5j, r = .695e-6, center=(1.758e-05, 1.753e-05, 21.2698e-6))
    sc = SphereCluster([s1, s2])
    alpha = 0.99

    lb1 = Sphere(1+1e-5j, 1e-8, 0, 0, 0)
    ub1 = Sphere(2+1e-5j, 1e-5, 1e-4, 1e-4, 1e-4)
    step1 = Sphere(1e-4+1e-4j, 1e-8, 0, 0, 0)
    lb = SphereCluster([lb1, lb1]), .1
    ub = SphereCluster([ub1, ub1]), 1    
    step = SphereCluster([step1, step1]), 0

    fitresult = fit(holo, (sc, alpha), 
                    scatterpy.theory.Multisphere(imshape = 100, 
                                                 optics = optics), 'nmpfit', 
                    lb, ub, step = step)

    fit_sc = fitresult[0]
    fit_alpha = fitresult[1]
    fitres_unpacked = np.array([fit_sc.n[0].real, fit_sc.n[0].imag, 
                                fit_sc.r[0], fit_sc.x[0], fit_sc.y[0],
                                fit_sc.z[0], fit_sc.n[1].real, fit_sc.n[1].imag,
                                fit_sc.r[1], fit_sc.x[1], fit_sc.y[1], 
                                fit_sc.z[1], fit_alpha])

    assert_array_almost_equal(fitres_unpacked * [1, 10**5, 10**7, 10**5, 10**5,
                                           10**5,1,10**5, 10**7, 10**5,10**5,
                                           10**5, 1], gold, decimal=2)


@attr('slow')
def test_six_mie_superposition():
    '''
    Right now Mie Superposition is only being tested for 2 simulated particles.
    Test against real data fitted by calling nmpfit directly.
    This links to data on the group share.

    Original data file: /group/manoharan/jerome/jf072511/13/image0095.tif
    with bg jerome/jf072611/bg_13.npy

    fnorm should be 6.216.  (Updated: got a better fit w/lower fnorm).

    Very slow -- takes about an hour to run.
    '''
    optics = hp.Optics(wavelen = 662.3e-9, polarization = [0., 1.0], 
                       divergence = 0., pixel_scale = [0.10678e-6, 0.10678e-6],
                       index = 1.4105)
    holo = hp.load('/group/manoharan/holopy/test_image_six_droplet.npy', 
                   optics = optics)
    gold = np.array([1.526, 1.0, 4.372, 1.3798, 1.5089, 1.3256, 
                     1.526, 1.0, 4.372, 1.4461, 1.2384, 1.5213,
                     1.526, 1.0, 4.372, 1.5164, 1.4277, 1.6675,
                     1.526, 1.0, 4.372, 1.4045, 1.5979, 1.4805,
                     1.526, 1.0, 4.372, 1.4152, 1.4171, 1.2691,
                     1.526, 1.0, 4.372, 1.7079, 1.5628, 1.4302, 2.3607])

    # set up initial guess
    s1 = Sphere(n = 1.515+1e-4j, r = 0.472e-6, center = (1.38e-05, 1.51e-05, 
                                                         1.33e-5))
    s2 = Sphere(n = 1.515+1e-4j, r = 0.472e-6, center = (1.45e-5, 1.24e-5, 1.52e-5))
    s3 = Sphere(n = 1.515+1e-4j, r = 0.472e-6, center = (1.52e-5, 1.43e-5, 1.67e-5))
    s4 = Sphere(n = 1.515+1e-4j, r = 0.472e-6, center = (1.40e-5, 1.60e-5, 1.48e-5))
    s5 = Sphere(n = 1.515+1e-4j, r = 0.472e-6, center = (1.42e-5, 1.42e-5, 1.27e-5))
    s6 = Sphere(n = 1.515+1e-4j, r = 0.472e-6, center = (1.71e-5, 1.56e-5, 1.43e-5))
    sc = SphereCluster([s1, s2, s3, s4, s5, s6])
    alpha = 0.24

    # bounds and step
    lb1 = Sphere(1+1e-4j, 1e-8, 0, 0, 0)
    ub1 = Sphere(2+1e-4j, 1e-5, 1e-4, 1e-4, 1e-4)
    lb = SphereCluster([lb1, lb1, lb1, lb1, lb1, lb1]), .1
    ub = SphereCluster([ub1, ub1, ub1, ub1, ub1, ub1]), 1. 
    step1 = Sphere(1e-4+1e-4j, 1e-7, 0, 0, 0)
    step = SphereCluster([step1, step1, step1, step1, step1, step1]), 0

    tie1 = Sphere(n = 1, r = 2, center = None)
    tie = SphereCluster([tie1, tie1, tie1, tie1, tie1, tie1]), None

    fitresult = fit(holo, (sc, alpha), scatterpy.theory.Mie, 'nmpfit', 
                    lb, ub, step = step, tie=tie)       
    fitres_unpacked = np.concatenate((fitresult[0].parameter_list, 
                                      np.array([fitresult[1]])))

    assert_array_almost_equal(fitres_unpacked, gold, decimal=2)

    
'''
def test_fit_cluster():
    path = os.path.abspath(hp.__file__)
    path = string.rstrip(path, chars='__init__.pyc')+'tests/exampledata/'
    holo = normalize(hp.load(path + image0002))

    sc = hp.model.scatterer.Cluster(
'''

@attr('slow')
def test_tie():
    s1 = Sphere(n=1.59, r = .5, center=(10,10,10))
    s2 = Sphere(n=1.59, r = .5, center=(10,11,11))
    sc = SphereCluster([s1, s2])

    optics = hp.Optics(wavelen=.66, index=1.33, pixel_scale=.1)
    theory = scatterpy.theory.Mie(optics)

    holo = theory.calc_holo(sc)

    igs1 = Sphere(n=1.59, r = .5, center=(10.1,10,10))
    igs2 = Sphere(n=1.59, r = .5, center=(10.1,11,11))
    ig = SphereCluster([igs1, igs2]), 1

    lb1 = Sphere(n=1.59, r = .5, center=(10,10,10))
    lb2 = Sphere(n=1.59, r = .5, center=(10,11,11))
    lb = SphereCluster([lb1, lb2]), 1
    
    ub1 = Sphere(n=1.59, r = .5, center=(10.5,10,10))
    ub2 = Sphere(n=1.59, r = .5, center=(10.5,11,11))
    ub = SphereCluster([ub1, ub2]), 1

    tie1 = Sphere(n=0+0j, r = None, center=[1, None, None])
    tie = SphereCluster([tie1, tie1]), None
    
    fitresult = fit(holo, ig, theory, 'ralg', lb, ub, tie = tie) 
    #fitresult = fit(holo, ig, theory, 'nmpfit', lb, ub, tie = None)

    assert_array_almost_equal(fitresult[0].parameter_list, sc.parameter_list)
