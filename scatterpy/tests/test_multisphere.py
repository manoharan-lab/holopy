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
Test T-matrix sphere cluster calculations and python interface.  

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
'''

import sys
import os
import numpy as np
hp_dir = (os.path.split(sys.path[0])[0]).rsplit(os.sep, 1)[0]
sys.path.append(hp_dir)

from nose.tools import assert_raises, with_setup
from numpy.testing import assert_equal, assert_array_almost_equal, assert_almost_equal


from nose.plugins.attrib import attr

import holopy

from scatterpy.theory.multisphere import Multisphere
from scatterpy.theory.mie import Mie
from scatterpy.scatterer import Sphere, SphereCluster
from scatterpy.errors import UnrealizableScatterer, TheoryNotCompatibleError
import scatterpy


def setup_model():
    global xoptics, yoptics, scaling_alpha, radius, n_particle_imag
    global n_particle_real, x, y, z, imshape, wavelen
    
    # define optical train
    wavelen = 658e-9
    ypolarization = [0., 1.0] # y-polarized
    xpolarization = [1.0, 0.] # x-polarized
    divergence = 0
    pixel_scale = [.1151e-6, .1151e-6]
    index = 1.33

    yoptics = holopy.optics.Optics(wavelen=wavelen, index=index,
                                   pixel_scale=pixel_scale,
                                   polarization=ypolarization,
                                   divergence=divergence)

    xoptics = holopy.optics.Optics(wavelen=wavelen, index=index,
                                   pixel_scale=pixel_scale,
                                   polarization=xpolarization,
                                   divergence=divergence)

    scaling_alpha = .6
    radius = .85e-6
    n_particle_real = 1.59
    n_particle_imag = 1e-4
    x = .576e-05
    y = .576e-05
    z = 15e-6

    imshape = 128

def teardown_model():
    global xoptics, yoptics, scaling_alpha, radius, n_particle_imag
    global n_particle_real, x, y, z, imshape, wavelen

    del xoptics, yoptics, scaling_alpha, radius, n_particle_imag
    del n_particle_real, x, y, z, imshape, wavelen
    
@attr('fast')
@with_setup(setup=setup_model, teardown=teardown_model)
def test_construction():
    # test constructor to make sure it works properly and calls base
    # class constructor properly
    theory = Multisphere(imshape=128, optics=xoptics, niter=100, eps=1e-6,
                         meth=0, qeps1=1e-5, qeps2=1e-8)

    assert_equal(theory.imshape, (128,128))
    assert_equal(theory.optics.wavelen, wavelen)
    assert_equal(theory.niter, 100)
    assert_equal(theory.eps, 1e-6)
    assert_equal(theory.meth, 0)
    assert_equal(theory.qeps1, 1e-5)
    assert_equal(theory.qeps2, 1e-8)


@with_setup(setup=setup_model, teardown=teardown_model)
def test_polarization():
    # test holograms for orthogonal polarizations; make sure they're
    # not the same, nor too different from one another.

    sphere = Sphere(n=n_particle_real + n_particle_imag*1j, r=radius, 
                    x=x, y=y, z=z)
    sc = SphereCluster([sphere])
    xmodel = Multisphere(imshape = imshape, optics=xoptics)
    ymodel = Multisphere(imshape = imshape, optics=yoptics)

    xholo = xmodel.calc_holo(sc, alpha=scaling_alpha)
    yholo = ymodel.calc_holo(sc, alpha=scaling_alpha)

    # the two arrays should not be equal
    try:
        assert_array_almost_equal(xholo, yholo)
    except AssertionError:
        pass
    else:
        raise AssertionError("Holograms computed for both x- and y-polarized light are too similar.")

    # but their max and min values should be close
    assert_almost_equal(xholo.max(), yholo.max())
    assert_almost_equal(xholo.min(), yholo.min())
    return xholo, yholo
    

@with_setup(setup=setup_model, teardown=teardown_model)
def test_2_sph():
    sc = SphereCluster(spheres=[Sphere(center=[7.1e-6, 7e-6, 10e-6],
                                       n=1.5811+1e-4j, r=5e-07),
                                Sphere(center=[6e-6, 7e-6, 10e-6],
                                       n=1.5811+1e-4j, r=5e-07)])


    theory = Multisphere(xoptics, imshape)

    holo = theory.calc_holo(sc, .6)

    assert_almost_equal(holo.max(), 1.4140292298443309)
    assert_almost_equal(holo.mean(), 0.9955420925817654)
    assert_almost_equal(holo.std(), 0.09558537595025796)

@attr('fast')
@with_setup(setup=setup_model, teardown=teardown_model)
def test_invalid():
    sc = SphereCluster(spheres=[Sphere(center=[7.1, 7e-6, 10e-6],
                                       n=1.5811+1e-4j, r=5e-07),
                                Sphere(center=[6e-6, 7e-6, 10e-6],
                                       n=1.5811+1e-4j, r=5e-07)])

    theory = Multisphere(xoptics, imshape)

    assert_raises(UnrealizableScatterer, theory.calc_holo, sc)
    
    sc = SphereCluster(spheres=[Sphere(center=[7.1, 7e-6, 10e-6],
                                       n=1.5811+1e-4j, r=5e-01),
                                Sphere(center=[6e-6, 7e-6, 10e-6],
                                       n=1.5811+1e-4j, r=5e-07)])
    
    assert_raises(UnrealizableScatterer, theory.calc_holo, sc)

    sc.scatterers[0].r = -1

    assert_raises(UnrealizableScatterer, theory.calc_holo, sc)

    cs = scatterpy.scatterer.CoatedSphere()

    assert_raises(TheoryNotCompatibleError, theory.calc_holo, cs)

"""
def test_single_sphere():
    # single spheres hologram (only tests that functions return)
    holo = mie.forward_holo(imshape, xoptics, n_particle_real,
                            n_particle_imag, radius, x, y, z, 
                            scaling_alpha)

    xfield, yfield, zfield = mie.calc_mie_fields(imshape, xoptics,
                                                 n_particle_real,
                                                 n_particle_imag,
                                                 radius, x, y, z)

def test_linearity():
    # look at superposition of scattering from two point particles;
    # make sure that this is sum of holograms from individual point
    # particles (scattered intensity should be negligible for this
    # case)

    x2 = x*2
    y2 = y*2
    z2 = z*2
    scaling_alpha = 1.0

    r = 1e-2*wavelen    # something much smaller than wavelength

    nrarr = np.array([n_particle_real, n_particle_real])
    niarr = np.array([n_particle_imag, n_particle_imag])
    rarr = np.array([r, r])
    xarr = np.array([x, x2])
    yarr = np.array([y, y2])
    zarr = np.array([z, z2])

    holo_1 = mie.forward_holo(imshape, xoptics, nrarr[0],
                              niarr[0], rarr[0], 
                              xarr[0], yarr[0], zarr[0],  
                              scaling_alpha)
    holo_2 = mie.forward_holo(imshape, xoptics, nrarr[1],
                              niarr[1], rarr[1], 
                              xarr[1], yarr[1], zarr[1],  
                              scaling_alpha)
    holo_super = mie.forward_holo(imshape, xoptics, nrarr,
                                     niarr, rarr, xarr, yarr, zarr, 
                                     scaling_alpha)

    # make sure we're not just looking at uniform arrays (could
    # happen if the size is set too small)
    try:
        assert_array_almost_equal(holo_1, holo_2, decimal=12)
    except AssertionError:
        pass    # no way to do "assert array not equal" in numpy.testing
    else:
        raise AssertionError("Hologram computed for point particle" +
                             " looks suspiciously close to having" +
                             " no fringes")

    # test linearity by subtracting off individual holograms
    # should recover the other hologram
    assert_array_almost_equal(holo_super - holo_1 + 1, holo_2)
    assert_array_almost_equal(holo_super - holo_2 + 1, holo_1)

    # uncomment to debug
    #return holo_1, holo_2, holo_super

def test_nonlinearity():
    # look at superposition of scattering from two large particles;
    # make sure that this is *not equal* to sum of holograms from
    # individual scatterers (scattered intensity should be
    # non-negligible for this case)

    x2 = x*2
    y2 = y*2
    z2 = z*2
    scaling_alpha = 1.0

    r = wavelen    # order of wavelength

    nrarr = np.array([n_particle_real, n_particle_real])
    niarr = np.array([n_particle_imag, n_particle_imag])
    rarr = np.array([r, r])
    xarr = np.array([x, x2])
    yarr = np.array([y, y2])
    zarr = np.array([z, z2])

    holo_1 = mie.forward_holo(imshape, xoptics, nrarr[0],
                              niarr[0], rarr[0], 
                              xarr[0], yarr[0], zarr[0],  
                              scaling_alpha)
    holo_2 = mie.forward_holo(imshape, xoptics, nrarr[1],
                              niarr[1], rarr[1], 
                              xarr[1], yarr[1], zarr[1],  
                              scaling_alpha)
    holo_super = mie.forward_holo(imshape, xoptics, nrarr,
                                     niarr, rarr, xarr, yarr, zarr, 
                                     scaling_alpha)

    # test nonlinearity by subtracting off individual holograms
    try:
        assert_array_almost_equal(holo_super - holo_1 + 1, holo_2)
    except AssertionError:
        pass    # no way to do "assert array not equal" in numpy.testing
    else:
        raise AssertionError("Holograms computed for " +
                             "wavelength-scale scatterers should " +
                             "not superpose linearly")

    # uncomment to debug
    #return holo_1, holo_2, holo_super

def test_two_spheres_samez():
    # put a second sphere in the same plane as the first.  This only
    # tests that the function returns.
    x2 = x*2
    y2 = y*2
    z2 = z

    nrarr = np.array([n_particle_real, n_particle_real])
    niarr = np.array([n_particle_imag, n_particle_imag])
    rarr = np.array([radius, radius])
    xarr = np.array([x, x2])
    yarr = np.array([y, y2])
    zarr = np.array([z, z2])
    holo = mie.forward_holo(imshape, xoptics, nrarr,
                            niarr, rarr, xarr, yarr, zarr, 
                            scaling_alpha)

    # uncomment to debug
    #return holo

def test_multiple_spheres():
    # test superposition from many spheres.  This only tests that the
    # function returns
    N = 10
    # this generates some random coordinates distributed uniformly
    # across the image
    xarr = np.random.random(N)*imshape*pixel_scale[0]
    yarr = np.random.random(N)*imshape*pixel_scale[0]
    zarr = np.random.random(N)*5e-6 + z # spread over 5-um in z
    rarr = np.ones(N)*radius
    nrarr = np.ones(N)*n_particle_real
    niarr = np.ones(N)*n_particle_imag
    narr = nrarr + 1j*niarr

    holo = mie.forward_holo(imshape, xoptics, nrarr,
                            niarr, rarr, xarr, yarr, zarr, 
                            scaling_alpha)
    # uncomment to debug
    #return holo

"""
