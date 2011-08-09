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
Test fortran-based Mie calculations and python interface.  

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
'''

import sys
import os
hp_dir = (os.path.split(sys.path[0])[0]).rsplit(os.sep, 1)[0]
sys.path.append(hp_dir)
import numpy as np
from numpy.linalg import norm
import os
import string
import pylab

from nose.tools import raises, assert_raises, assert_true
from numpy.testing import assert_, assert_equal, \
    assert_array_almost_equal, assert_array_equal, assert_almost_equal
from nose.tools import with_setup
from nose.plugins.attrib import attr

import holopy
from holopy.model.scatterer import Sphere, SphereCluster
from holopy.model.theory import mie
from holopy.model.theory import Mie



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

@attr('fast')
def test_mie_polarization():
    # test holograms for orthogonal polarizations; make sure they're
    # not the same, nor too different from one another.
    sphere = Sphere(n=n_particle_real + n_particle_imag*1j, r=radius, 
                    x=x, y=y, z=z)
    xmodel = Mie(imshape = imshape, optics=xoptics)
    ymodel = Mie(imshape = imshape, optics=yoptics)

    xholo = xmodel.calc_holo(sphere, alpha=scaling_alpha)
    yholo = ymodel.calc_holo(sphere, alpha=scaling_alpha)

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

@attr('fast')
def test_single_sphere():
    # single sphere hologram (only tests that functions return)
    sphere = Sphere(n=n_particle_real + n_particle_imag*1j, r=radius, 
                    x=x, y=y, z=z)
    model = Mie(imshape = imshape, optics=xoptics)

    holo = model.calc_holo(sphere, alpha=scaling_alpha)
    xfield, yfield, zfield = model.calc_field(sphere)

@attr('fast')
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

    sphere1 = Sphere(n=n_particle_real + n_particle_imag*1j, r=r, 
                     x=x, y=y, z=z)
    sphere2 = Sphere(n=n_particle_real + n_particle_imag*1j, r=r, 
                     x=x2, y=y2, z=z2)

    sc = SphereCluster(spheres = [sphere1, sphere2])
    model = Mie(imshape=imshape, optics=xoptics)
    
    holo_1 = model.calc_holo(sphere1, alpha=scaling_alpha)
    holo_2 = model.calc_holo(sphere2, alpha=scaling_alpha)
    holo_super = model.calc_holo(sc)

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

    # Test linearity by subtracting off individual holograms.
    # This should recover the other hologram
    assert_array_almost_equal(holo_super - holo_1 + 1, holo_2)
    assert_array_almost_equal(holo_super - holo_2 + 1, holo_1)

    # uncomment to debug
    #return holo_1, holo_2, holo_super

@attr('fast')
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

    sphere1 = Sphere(n=n_particle_real + n_particle_imag*1j, r=r, 
                     x=x, y=y, z=z)
    sphere2 = Sphere(n=n_particle_real + n_particle_imag*1j, r=r, 
                     x=x2, y=y2, z=z2)

    sc = SphereCluster(spheres = [sphere1, sphere2])
    model = Mie(imshape=imshape, optics=xoptics)
    
    holo_1 = model.calc_holo(sphere1, alpha=scaling_alpha)
    holo_2 = model.calc_holo(sphere2, alpha=scaling_alpha)
    holo_super = model.calc_holo(sc)

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

@attr('fast')
def test_two_spheres_samez():
    # put a second sphere in the same plane as the first.  This only
    # tests that the function returns.
    x2 = x*2
    y2 = y*2
    z2 = z
    sphere1 = Sphere(n=n_particle_real + n_particle_imag*1j, r=radius, 
                     x=x, y=y, z=z)
    sphere2 = Sphere(n=n_particle_real + n_particle_imag*1j, r=radius, 
                     x=x2, y=y2, z=z2)

    sc = SphereCluster(spheres = [sphere1, sphere2])
    model = Mie(imshape=imshape, optics=xoptics)
    
    holo = model.calc_holo(sc, alpha=scaling_alpha)

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

    sc = SphereCluster(n = nrarr + niarr*1j, r = rarr, x=xarr, y=yarr,
                       z=zarr)
    model = Mie(imshape=imshape, optics=xoptics)
    holo = model.calc_holo(sc, alpha=scaling_alpha)

    # uncomment to debug
    #return holo
