# Copyright 2011-2013, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang
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
'''
Test fortran-based Mie calculations and python interface.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''

import os
from nose.tools import with_setup, assert_raises, nottest
import yaml
import warnings

import numpy as np
from numpy.testing import assert_equal
from numpy.testing import (assert_array_almost_equal, assert_almost_equal,
                           assert_raises)
from nose.plugins.attrib import attr

from ..scatterer import Sphere, Spheres, Ellipsoid
from holopy.scattering.scatterer.sphere import LayeredSphere
from ..theory import Mie

from ..theory.mie import UnrealizableScatterer
from ..errors import TheoryNotCompatibleError
from ...core import (ImageSchema, Image, Optics, Angles, Schema, VolumeSchema,
                     subimage)
from .common import verify, sphere, xschema, scaling_alpha, optics, yschema
from .common import x, y, z, n, xoptics, radius
from ...core.tests.common import assert_allclose, assert_obj_close

@attr('fast')
def test_single_sphere():
    # single sphere hologram (only tests that functions return)
    thry = Mie(False)
    holo = thry.calc_holo(sphere, xschema, scaling=scaling_alpha)
    field = thry.calc_field(sphere, xschema)

    intensity = thry.calc_intensity(sphere, xschema)

    verify(holo, 'single_holo')
    verify(field, 'single_field')

    # now test some invalid scatterers and confirm that it rejects calculating
    # for them

    # large radius (calculation not attempted because it would take forever
    assert_raises(UnrealizableScatterer, Mie.calc_holo, Sphere(r=1, n = 1.59, center = (5,5,5)), xschema)

@attr('fast')
def test_subimaged():
    # make a dummy image so that we can pretend we are working with
    # data we want to subimage
    im = Image(np.zeros(xschema.shape), xschema.spacing, xschema.optics)
    h = Mie.calc_holo(sphere, im)
    sub = (60, 70), 30
    hs = Mie.calc_holo(sphere, subimage(im, *sub))

    assert_obj_close(subimage(h, *sub), hs)


@attr('fast')
def test_Mie_multiple():
    s1 = Sphere(n = 1.59, r = 5e-7, center = (1e-6, -1e-6, 10e-6))
    s2 = Sphere(n = 1.59, r = 1e-6, center=[8e-6,5e-6,5e-6])
    s3 = Sphere(n = 1.59+0.0001j, r = 5e-7, center=[5e-6,10e-6,3e-6])
    sc = Spheres(scatterers=[s1, s2, s3])
    thry = Mie(False)

    schema = yschema
    fields = thry.calc_field(sc, schema)

    verify(fields, 'mie_multiple_fields')
    thry.calc_intensity(sc, schema)

    holo = thry.calc_holo(sc, schema)
    verify(holo, 'mie_multiple_holo')

    # should throw exception when fed a ellipsoid
    el = Ellipsoid(n = 1.59, r = (1e-6, 2e-6, 3e-6), center=[8e-6,5e-6,5e-6])
    with assert_raises(TheoryNotCompatibleError) as cm:
        thry.calc_field(el, schema)
    assert_equal(str(cm.exception), "The implementation of the Mie scattering "
                 "theory doesn't know how to handle scatterers of type "
                 "Ellipsoid")

    assert_raises(TheoryNotCompatibleError, Mie.calc_field, el, schema)
    assert_raises(TheoryNotCompatibleError, Mie.calc_intensity,
                  el, schema)
    assert_raises(TheoryNotCompatibleError, Mie.calc_holo, el, schema)

@attr('fast')
def test_mie_polarization():
    # test holograms for orthogonal polarizations; make sure they're
    # not the same, nor too different from one another.
    thry = Mie(False)
    xholo = thry.calc_holo(sphere, xschema, scaling=scaling_alpha)
    yholo = thry.calc_holo(sphere, yschema, scaling=scaling_alpha)

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

def test_linearity():
    # look at superposition of scattering from two point particles;
    # make sure that this is sum of holograms from individual point
    # particles (scattered intensity should be negligible for this
    # case)

    x2 = x*2
    y2 = y*2
    z2 = z*2
    scaling_alpha = 1.0
    r = 1e-2*optics.wavelen    # something much smaller than wavelength

    sphere1 = Sphere(n=n, r=r, center = (x, y, z))
    sphere2 = Sphere(n=n, r=r, center = (x2, y2, z2))

    sc = Spheres(scatterers = [sphere1, sphere2])

    holo_1 = Mie.calc_holo(sphere1, xschema, scaling=scaling_alpha)
    holo_2 = Mie.calc_holo(sphere2, xschema, scaling=scaling_alpha)
    holo_super = Mie.calc_holo(sc, xschema, scaling=scaling_alpha)

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
    r = optics.wavelen    # order of wavelength

    sphere1 = Sphere(n=n, r=r, center = (x, y, z))
    sphere2 = Sphere(n=n, r=r, center = (x2, y2, z2))

    sc = Spheres(scatterers = [sphere1, sphere2])

    holo_1 = Mie.calc_holo(sphere1, xschema, scaling=scaling_alpha)
    holo_2 = Mie.calc_holo(sphere2, xschema, scaling=scaling_alpha)
    holo_super = Mie.calc_holo(sc, xschema, scaling=scaling_alpha)

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


@attr('fast')
def test_radiometric():
    cross_sects = Mie.calc_cross_sections(sphere, xoptics)
    # turn cross sections into efficiencies
    cross_sects[0:3] = cross_sects[0:3] / (np.pi * radius**2)

    # create a dict from the results
    result = {}
    result_keys = ['qscat', 'qabs', 'qext', 'costheta']
    for key, val in zip(result_keys, cross_sects):
        result[key] = val

    location = os.path.split(os.path.abspath(__file__))[0]
    gold_name = os.path.join(location, 'gold',
                             'gold_mie_radiometric')
    gold = yaml.load(file(gold_name + '.yaml'))
    for key, val in gold.iteritems():
        assert_almost_equal(gold[key], val, decimal = 5)

@attr('fast')
def test_farfield():
    schema = Schema(positions = Angles(np.linspace(0, np.pi/2)), optics =
                    Optics(wavelen=.66, index = 1.33, polarization = (1, 0)))
    sphere = Sphere(r = .5, n = 1.59+0.1j)

    matr = Mie.calc_scat_matrix(sphere, schema)
    verify(matr, 'farfield_matricies', rtol = 1e-6)

@attr('medium')
def test_radialEscat():
    thry_1 = Mie()
    thry_2 = Mie(False)

    sphere = Sphere(r = 1e-6, n = 1.4 + 0.01j, center = [10e-6, 10e-6,
                                                         1.2e-6])
    h1 = thry_1.calc_holo(sphere, xschema)
    h2 = thry_2.calc_holo(sphere, xschema)

    try:
        assert_array_almost_equal(h1, h2, decimal=12)
    except AssertionError:
        pass    # no way to do "assert array not equal" in numpy.testing
    else:
        raise AssertionError("Holograms w/ and w/o full radial fields" +
                             " are exactly equal")

def test_calc_xz_plane():
    s = Sphere(n = 1.59, r = .5, center = (0, 0, 5))
    sch = VolumeSchema((50, 1, 50), .1, Optics(.66, 1.33, (1,0)))
    e = Mie.calc_field(s, sch)

# TODO: finish internal fields
def test_internal_fields():
    s = Sphere(1.59, .5, (5, 5, 0))
    sch = ImageSchema((100, 100), .1, Optics(.66, 1.33, (1, 0)))
    # TODO: actually test correctness

def test_1d():
    s = Sphere(1.59, .5, (5, 5, 0))
    sch = ImageSchema((10, 10), .1, Optics(.66, 1.33, (1, 0)))
    holo = Mie.calc_holo(s, sch)
    field = Mie.calc_field(s, sch)
    flatsch = Schema(positions=sch.positions.xyz(), optics=sch.optics)

    flatholo = Mie.calc_holo(s, flatsch)
    flatfield = Mie.calc_field(s, flatsch)

    assert_equal(holo.ravel(), flatholo)
    assert_equal(flatfield, field.reshape(flatfield.shape))

def test_layered():
    l = LayeredSphere(n = (1, 2), t = (1, 1), center = (2, 2, 2))
    s = Sphere(n = (1,2), r = (1, 2), center = (2, 2, 2))
    sch = ImageSchema((10, 10), .2, Optics(.66, 1, (1, 0)))
    hl = Mie.calc_holo(l, sch)
    hs = Mie.calc_holo(s, sch)
    assert_equal(hl, hs)
