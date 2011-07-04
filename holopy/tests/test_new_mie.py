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
Some code to test the refactoring of the mie module.  Compares output
of the functions from the new class structure to output from the
legacy (non-method) functions.  Can delete this file once refactoring
is complete.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
'''

import numpy as np
import os
import string
import pylab

from nose.tools import raises, assert_raises
from numpy.testing import assert_, assert_equal, assert_array_almost_equal
from nose.tools import with_setup

import holopy
from holopy.model.theory import mie
from holopy.model.theory import Mie
from holopy.model.scatterer import Sphere, SphereCluster
from holopy.third_party import nmpfit
from holopy.process import normalize

# define optical train
wavelen = 658e-9
#polarization = [0., 1.0]        # y-polarized
polarization = [1.0, 0.]       # x-polarized
divergence = 0
pixel_scale = [.1151e-6, .1151e-6]
index = 1.33

optics = holopy.optics.Optics(wavelen=wavelen, index=index,
                              pixel_scale=pixel_scale,
                              polarization=polarization,
                              divergence=divergence)

# initial guess
scaling_alpha = .6
radius = .85e-6
n_particle_real = 1.59
n_particle_imag = 1e-4
x = .576e-05
y = .576e-05
z = 15e-6

imshape = 128

def test_single_sphere():
    sphere = Sphere(n=n_particle_real + n_particle_imag*1j, r=radius, 
                    x=x, y=y, z=z)
    theory = Mie(imshape = imshape, optics=optics)

    # compare holograms
    hnew = theory.calc_holo(sphere, alpha=scaling_alpha)
    hold = mie.forward_holo(imshape, optics, n_particle_real,
                            n_particle_imag, radius, x, y, z, 
                            scaling_alpha)

    assert_array_almost_equal(hnew, hold)

def test_single_sphere_polarization():
    # optics for x-polarization
    xopts = holopy.optics.Optics(wavelen=wavelen, index=index,
                                pixel_scale=pixel_scale,
                                polarization=[1.0, 0],
                                divergence=divergence)
    # optics for y-polarization
    yopts = holopy.optics.Optics(wavelen=wavelen, index=index,
                                pixel_scale=pixel_scale,
                                polarization=[0, 1.0],
                                divergence=divergence)
    sphere = Sphere(n=n_particle_real + n_particle_imag*1j, r=radius, 
                    x=x, y=y, z=z)

    ytheory = Mie(imshape = imshape, optics=yopts)
    xtheory = Mie(imshape = imshape, optics=xopts)

    yh = ytheory.calc_holo(sphere, alpha=scaling_alpha)
    xh = xtheory.calc_holo(sphere, alpha=scaling_alpha)

    # holograms should *not* be the same
    try:
        assert_array_almost_equal(xh, yh)
    except AssertionError:
        pass    # no way to do "assert array not equal" in numpy.testing
    else:
        raise AssertionError("Holograms calculated for x- and " +
                             "y-polarizations look suspiciously "+
                             "similar")

    return xh, yh


def test_two_spheres_samez():
    # put a second sphere at twice x and y
    x2 = x*2
    y2 = y*2
    z2 = z

    s1 = Sphere(n=n_particle_real + n_particle_imag*1j, r=radius, 
                    x=x, y=y, z=z)
    s2 = Sphere(n=n_particle_real + n_particle_imag*1j, r=radius, 
                    x=x2, y=y2, z=z2)
    sc = SphereCluster(spheres=[s1,s2])
    theory = Mie(imshape = imshape, optics=optics)
    inew = theory.calc_intensity(sc)
    hnew = theory.calc_holo(sc, alpha=scaling_alpha)
    
    nrarr = np.array([n_particle_real, n_particle_real])
    niarr = np.array([n_particle_imag, n_particle_imag])
    rarr = np.array([radius, radius])
    xarr = np.array([x, x2])
    yarr = np.array([y, y2])
    zarr = np.array([z, z2])
    iold = mie.forward_holo(imshape, optics, nrarr,
                            niarr, rarr, xarr, yarr, zarr, 
                            scaling_alpha, intensity=True)
    hold = mie.forward_holo(imshape, optics, nrarr,
                            niarr, rarr, xarr, yarr, zarr, 
                            scaling_alpha)

    assert_array_almost_equal(inew, iold)
    assert_array_almost_equal(hnew, hold)

def test_multiple_spheres():
    N = 10
    # this generates some random coordinates distributed uniformly
    # across the image
    xarr = np.random.random(N)*imshape*pixel_scale[0]
    yarr = np.random.random(N)*imshape*pixel_scale[0]
    zarr = np.random.random(N)*5e-6 + z
    rarr = np.ones(N)*radius
    nrarr = np.ones(N)*n_particle_real
    niarr = np.ones(N)*n_particle_imag
    narr = nrarr + 1j*niarr

    sc = SphereCluster(n=narr, r=rarr, x=xarr, y=yarr, z=zarr)
    theory = Mie(imshape = imshape, optics=optics)
    inew = theory.calc_intensity(sc)
    hnew = theory.calc_holo(sc, alpha=scaling_alpha)
    
    iold = mie.forward_holo(imshape, optics, nrarr,
                            niarr, rarr, xarr, yarr, zarr, 
                            scaling_alpha, intensity=True)
    hold = mie.forward_holo(imshape, optics, nrarr,
                            niarr, rarr, xarr, yarr, zarr, 
                            scaling_alpha)
    assert_array_almost_equal(inew, iold)
    assert_array_almost_equal(hnew, hold)

