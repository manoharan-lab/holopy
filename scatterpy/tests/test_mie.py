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
.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''

import sys
import os
hp_dir = (os.path.split(sys.path[0])[0]).rsplit(os.sep, 1)[0]
sys.path.append(hp_dir)
from nose.tools import with_setup, assert_raises

import numpy as np
from numpy.testing import assert_equal
from numpy.testing import (assert_array_almost_equal, assert_almost_equal,
                           assert_raises, assert_allclose)
from nose.plugins.attrib import attr

from holopy import Optics
import scatterpy
from scatterpy.scatterer import (Sphere, SphereCluster, CoatedSphere, Ellipsoid)
from scatterpy.theory import Mie

from scatterpy.theory.mie import UnrealizableScatterer
from scatterpy.errors import TheoryNotCompatibleError
import common

# nose setup/teardown methods
def setup_model():
    global xoptics, yoptics, xmodel, ymodel, scaling_alpha, radius, n
    global n_particle_real, n_particle_imag, x, y, z, optics

    scaling_alpha = .6
    radius = .85e-6
    n = 1.59+1e-4j
    n_particle_real = 1.59
    n_particle_imag = 1e-4
    x = .576e-05
    y = .576e-05
    z = 15e-6

    imshape = 128
    
    # set up optics class for use in several test functions
    xoptics, yoptics = common.xoptics, common.yoptics
    optics = common.optics
    xmodel = Mie(imshape = imshape, optics=xoptics)
    ymodel = Mie(imshape = imshape, optics=yoptics)

def teardown_model():
    global xoptics, yoptics, xmodel, ymodel, scaling_alpha, radius, n
    global n_particle_real, n_particle_imag, x, y, z, optics
    del xoptics, yoptics, xmodel, ymodel, scaling_alpha, radius, n
    del n_particle_real, n_particle_imag, x, y, z, optics


@attr('fast')
@with_setup(setup=setup_model, teardown=teardown_model)
def test_Mie_construction():
    theory = Mie(optics)
    assert_equal(theory.imshape, (256,256))
    theory = Mie(optics, imshape=(100,100))
    assert_equal(theory.imshape , (100,100))

    # test with single value instead of tuple
    theory = Mie(optics, imshape=128)
    assert_equal(theory.imshape , (128,128))

    # construct with optics
    theory = Mie(imshape=256, optics=optics)
    assert_equal(theory.optics.index, 1.33)

    # construct with a dict as optics
    theory = Mie({'wavelen': .66, 'pixel_scale': .1, 'index': 1.33})

    common.assert_obj_close(theory, eval(repr(theory)))

    


@attr('fast')
@with_setup(setup=setup_model, teardown=teardown_model)
def test_single_sphere():
    # single sphere hologram (only tests that functions return)
    sphere = Sphere(n=n, r=radius, center=(x, y, z))

    holo = xmodel.calc_holo(sphere, alpha=scaling_alpha)
    field = xmodel.calc_field(sphere)

    intensity = xmodel.calc_intensity(sphere)

    common.verify(holo, 'single_holo')
    common.verify(field, 'single_field')

    # now test some invalid scatterers and confirm that it rejects calculating
    # for them

    # large radius (calculation not attempted because it would take forever
    assert_raises(UnrealizableScatterer, xmodel.calc_holo, Sphere(r=1))
 


@attr('fast')
@with_setup(setup=setup_model, teardown=teardown_model)
def test_Mie_multiple():
    s1 = Sphere(n = 1.59, r = 5e-7, center = (1e-6, -1e-6, 10e-6))
    s2 = Sphere(n = 1.59, r = 1e-6, center=[8e-6,5e-6,5e-6])
    s3 = Sphere(n = 1.59+0.0001j, r = 5e-7, center=[5e-6,10e-6,3e-6])
    sc = SphereCluster(spheres=[s1, s2, s3])
    theory = Mie(imshape=128, optics=optics)

    fields = theory.calc_field(sc)

    common.verify(fields, 'mie_multiple_fields')
    theory.calc_intensity(sc)

    holo = theory.calc_holo(sc)
    common.verify(holo, 'mie_multiple_holo')

    # should throw exception when fed a ellipsoid
    el = Ellipsoid(n = 1.59, r = (1e-6, 2e-6, 3e-6), center=[8e-6,5e-6,5e-6])
    with assert_raises(TheoryNotCompatibleError) as cm:
        theory.calc_field(el)
    assert_equal(str(cm.exception), "The implementation of the Mie scattering "
                 "theory doesn't know how to handle scatterers of type "
                 "Ellipsoid")
    
    assert_raises(TheoryNotCompatibleError, theory.calc_field, el)
    assert_raises(TheoryNotCompatibleError, theory.calc_intensity,
                  el)
    assert_raises(TheoryNotCompatibleError, theory.calc_holo, el)
    # and when the list of scatterers includes a coated sphere
    sc.add(el)
    assert_raises(TheoryNotCompatibleError, theory.calc_field, sc)
    assert_raises(TheoryNotCompatibleError, theory.calc_intensity, sc)
    assert_raises(TheoryNotCompatibleError, theory.calc_holo, sc)
    
@attr('fast')
@with_setup(setup=setup_model, teardown=teardown_model)
def test_mie_polarization():
    # test holograms for orthogonal polarizations; make sure they're
    # not the same, nor too different from one another.
    sphere = Sphere(n=n, r=radius, center=(x, y, z))

    xholo = xmodel.calc_holo(sphere, alpha=scaling_alpha)
    yholo = ymodel.calc_holo(sphere, alpha=scaling_alpha)

    # the two arrays should not be equal
    try:
        assert_array_almost_equal(xholo, yholo)
    except AssertionError:
        pass
    else:
        raise AssertionError("Holograms computed for both x- and y-polarized "
                             "light are too similar.")

    # but their max and min values should be close
    assert_almost_equal(xholo.max(), yholo.max())
    assert_almost_equal(xholo.min(), yholo.min())
    return xholo, yholo

@attr('fast')
@with_setup(setup=setup_model, teardown=teardown_model)
def test_linearity():
    # look at superposition of scattering from two point particles;
    # make sure that this is sum of holograms from individual point
    # particles (scattered intensity should be negligible for this
    # case)

    x2 = x*2
    y2 = y*2
    z2 = z*2
    scaling_alpha = 1.0
    r = 1e-2*xoptics.wavelen    # something much smaller than wavelength

    sphere1 = Sphere(n=n, r=r, center = (x, y, z))
    sphere2 = Sphere(n=n, r=r, center = (x2, y2, z2))

    sc = SphereCluster(spheres = [sphere1, sphere2])
    model = xmodel
    
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
@with_setup(setup=setup_model, teardown=teardown_model)
def test_nonlinearity():
    # look at superposition of scattering from two large particles;
    # make sure that this is *not equal* to sum of holograms from
    # individual scatterers (scattered intensity should be
    # non-negligible for this case)

    x2 = x*2
    y2 = y*2
    z2 = z*2
    scaling_alpha = 1.0
    r = xoptics.wavelen    # order of wavelength

    sphere1 = Sphere(n=n, r=r, center = (x, y, z))
    sphere2 = Sphere(n=n, r=r, center = (x2, y2, z2))

    sc = SphereCluster(spheres = [sphere1, sphere2])
    model = xmodel
    
    holo_1 = model.calc_holo(sphere1, alpha=scaling_alpha)
    holo_2 = model.calc_holo(sphere2, alpha=scaling_alpha)
    holo_super = model.calc_holo(sc)

    # test nonlinearity by subtracting off individual holograms
    try:
        assert_array_almost_equal(holo_super - holo_1 + 1, holo_2)
    except AssertionError:
        pass    # no way to do "assert array not equal" in numpy.testing
    else:
        raise AssertionError("Holograms computed for " 
                             "wavelength-scale scatterers should " 
                             "not superpose linearly")

    # uncomment to debug
    #return holo_1, holo_2, holo_super
