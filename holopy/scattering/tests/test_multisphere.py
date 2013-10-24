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
Test T-matrix sphere cluster calculations and python interface.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
'''
from __future__ import division

import sys
import os
import numpy as np

import warnings
from nose.tools import assert_raises, with_setup, nottest
from numpy.testing import (assert_equal, assert_array_almost_equal,
                           assert_almost_equal, assert_allclose)

from nose.plugins.attrib import attr

from ...core import Optics, ImageSchema, Schema, Angles
from ..theory import Multisphere
from ..theory.multisphere import MultisphereExpansionNaN, ConvergenceFailureMultisphere
from ..scatterer import Sphere, Spheres
from ..errors import UnrealizableScatterer, TheoryNotCompatibleError
from .common import assert_allclose, verify, xschema, yschema
from .common import scaling_alpha, sphere

schema = xschema

@attr('fast')
def test_construction():
    # test constructor to make sure it works properly and calls base
    # class constructor properly
    theory = Multisphere(niter=100, eps=1e-6, meth=0, qeps1=1e-5, qeps2=1e-8)

    assert_equal(theory.niter, 100)
    assert_equal(theory.eps, 1e-6)
    assert_equal(theory.meth, 0)
    assert_equal(theory.qeps1, 1e-5)
    assert_equal(theory.qeps2, 1e-8)


def test_polarization():
    # test holograms for orthogonal polarizations; make sure they're
    # not the same, nor too different from one another.

    sc = Spheres([sphere])

    xholo = Multisphere.calc_holo(sc, xschema, scaling=scaling_alpha)
    yholo = Multisphere.calc_holo(sc, yschema, scaling=scaling_alpha)

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


def test_2_sph():
    sc = Spheres(scatterers=[Sphere(center=[7.1e-6, 7e-6, 10e-6],
                                       n=1.5811+1e-4j, r=5e-07),
                                Sphere(center=[6e-6, 7e-6, 10e-6],
                                       n=1.5811+1e-4j, r=5e-07)])


    holo = Multisphere.calc_holo(sc, schema, scaling=.6)

    assert_almost_equal(holo.max(), 1.4140292298443309)
    assert_almost_equal(holo.mean(), 0.9955420925817654)
    assert_almost_equal(holo.std(), 0.09558537595025796)


def test_radial_holos():
    # Check that holograms computed with and w/o radial part of E_scat differ
    sc = Spheres(scatterers=[Sphere(center=[7.1e-6, 7e-6, 10e-6],
                                       n=1.5811+1e-4j, r=5e-07),
                                Sphere(center=[6e-6, 7e-6, 10e-6],
                                       n=1.5811+1e-4j, r=5e-07)])
    thry_nonrad = Multisphere()
    thry_rad = Multisphere(compute_escat_radial = True)

    holo_nonrad = thry_nonrad.calc_holo(sc, schema)
    holo_rad = thry_rad.calc_holo(sc, schema)

    # the two arrays should not be equal
    try:
        assert_allclose(holo_nonrad, holo_rad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Holograms computed with and without radial component of scattered electric field are too similar.")


@attr('fast')
def test_invalid():
    sc = Spheres(scatterers=[Sphere(center=[7.1, 7e-6, 10e-6],
                                       n=1.5811+1e-4j, r=5e-07),
                                Sphere(center=[6e-6, 7e-6, 10e-6],
                                       n=1.5811+1e-4j, r=5e-07)])


    assert_raises(UnrealizableScatterer, Multisphere.calc_holo, sc, schema)

    sc = Spheres(scatterers=[Sphere(center=[7.1, 7e-6, 10e-6],
                                       n=1.5811+1e-4j, r=5e-01),
                                Sphere(center=[6e-6, 7e-6, 10e-6],
                                       n=1.5811+1e-4j, r=5e-07)])

    assert_raises(UnrealizableScatterer, Multisphere.calc_holo, sc, schema)

    sc.scatterers[0].r = -1

    assert_raises(UnrealizableScatterer, Multisphere.calc_holo, sc, schema)

    cs = Sphere(center = (0, 0, 0))

    assert_raises(TheoryNotCompatibleError, Multisphere.calc_holo, cs, schema)

    # try a coated sphere
    sc2 = Spheres([Sphere(center = [0., 0., 0.],
                          n = [1.+0.1j, 1.2],
                          r = [4e-7, 5e-7])])
    assert_raises(TheoryNotCompatibleError, Multisphere.calc_cross_sections,
                  sc2, schema.optics)


def test_overlap():
    # should raise a warning
    with warnings.catch_warnings(True) as w:
        warnings.simplefilter("always")
        sc = Spheres(scatterers=[Sphere(center=[3e-6, 3e-6, 10e-6],
                                           n=1.59, r=.5e-6),
                                    Sphere(center=[3.4e-6, 3e-6, 10e-6],
                                           n=1.59, r=.5e-6)])
        assert len(w) > 0

    # should fail to converge
    assert_raises(MultisphereExpansionNaN, Multisphere.calc_holo, sc, schema)

    # but it should succeed with a small overlap, after raising a warning
    with warnings.catch_warnings(True) as w:
        warnings.simplefilter("always")
        sc = Spheres(scatterers=[Sphere(center=[3e-6, 3e-6, 10e-6],
                                           n=1.59, r=.5e-6),
                                    Sphere(center=[3.9e-6, 3.e-6, 10e-6],
                                           n=1.59, r=.5e-6)])
        assert len(w) > 0
    holo = Multisphere.calc_holo(sc, schema)

    verify(holo, '2_sphere_allow_overlap')


@attr('fast')
def test_niter():
    sc = Spheres(scatterers=[Sphere(center=[7.1e-6, 7e-6, 10e-6],
                                          n=1.5811+1e-4j, r=5e-07),
                                   Sphere(center=[6e-6, 7e-6, 10e-6],
                                          n=1.5811+1e-4j, r=5e-07)])
    multi = Multisphere(niter = 2)

    assert_raises(ConvergenceFailureMultisphere, multi.calc_holo, sc, schema)

def test_cross_sections():
    opt = Optics(wavelen = 1., index = 1., polarization = [1., 0])
    a = 1./(2 * np.pi) # size parameter 1
    n = 1.5 + 0.1j
    sc = Spheres([Sphere(n = n, r = a, center = [0., 0., a]),
                  Sphere(n = n, r = a, center = [0., 0., -a])])
    thry = Multisphere()
    # this ends up testing the angular dependence of scattering
    # as well as all the scattering coefficients
    xsects = thry.calc_cross_sections(sc, opt)

    gold_xsects = np.array([0.03830316, 0.04877015, 0.08707331])
    # calculated directly by SCSMFO. Efficiencies normalized
    # in funny way by SCSMFO, based on "volume mean radius".
    assert_allclose(xsects[:3], gold_xsects, rtol = 1e-3)

def test_farfield():
    schema = Schema(positions = Angles(np.linspace(0, np.pi/2),
                                       phi = np.zeros(50)),
                    optics = Optics(wavelen=.66, index = 1.33,
                                    polarization = (1, 0)))
    n = 1.59+0.01j
    r = 0.5

    cluster = Spheres([Sphere(n = n, r = r, center = [0., 0., r]),
                       Sphere(n = n, r = r, center = [0., 0., -r])])

    matr = Multisphere.calc_scat_matrix(cluster, schema)
