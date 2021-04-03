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
'''
Test T-matrix sphere cluster calculations and python interface.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
'''


import sys
import os
import warnings
import unittest

import numpy as np
from numpy.testing import (assert_equal, assert_array_almost_equal,
                           assert_allclose, assert_array_equal, assert_raises)

from nose.plugins.attrib import attr
from nose.plugins.skip import SkipTest
import scipy

from holopy.core.metadata import detector_points
from holopy.scattering import (
    calc_holo, calc_scat_matrix, calc_cross_sections, Multisphere, Sphere,
    Spheres)
from holopy.scattering.errors import (
    InvalidScatterer, TheoryNotCompatibleError, MultisphereFailure,
    OverlapWarning)
from holopy.scattering.tests.common import (
    xschema, yschema, index, wavelen, xpolarization, ypolarization,
    scaling_alpha, sphere)
from holopy.core.tests.common import assert_obj_close, verify

schema = xschema

SCATTERERS_LARGE_OVERLAP = [
    Sphere(center=[3e-6, 3e-6, 10e-6], n=1.59, r=.5e-6),
    Sphere(center=[3.4e-6, 3e-6, 10e-6], n=1.59, r=.5e-6),
    ]
SCATTERERS_SMALL_OVERLAP = [
    Sphere(center=[3e-6, 3e-6, 10e-6], n=1.59, r=.5e-6),
    Sphere(center=[3.9e-6, 3.e-6, 10e-6], n=1.59, r=.5e-6),
    ]


class TestMultisphere(unittest.TestCase):
    @attr('fast')
    def test_theory_from_parameters(self):
        np.random.seed(1332)
        kwargs = {
            'niter': np.random.randint(400),
            'eps': np.random.rand() * 1e-6,
            'meth': 1,
            'qeps1': np.random.rand() * 1e-5,
            'qeps2': np.random.rand() * 1e-8,
            'compute_escat_radial': np.random.choice([True, False]),
            'suppress_fortran_output': np.random.choice([True, False]),
            }
        theory_in = Multisphere(**kwargs)
        pars = {}
        theory_out = theory_in.from_parameters(pars)

        for k, v in kwargs.items():
            self.assertEqual(getattr(theory_out, k), v)



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


@attr('medium')
def test_polarization():
    # test holograms for orthogonal polarizations; make sure they're
    # not the same, nor too different from one another.

    sc = Spheres([sphere])

    xholo = calc_holo(xschema, sc, index, wavelen, xpolarization, scaling=scaling_alpha)
    yholo = calc_holo(yschema, sc, index, wavelen, ypolarization, scaling=scaling_alpha)

    # the two arrays should not be equal
    try:
        assert_array_almost_equal(xholo, yholo)
    except AssertionError:
        pass
    else:
        raise AssertionError("Holograms computed for both x- and y-polarized light are too similar.")

    # but their max and min values should be close
    assert_obj_close(xholo.max(), yholo.max())
    assert_obj_close(xholo.min(), yholo.min())
    return xholo, yholo


@attr('medium')
def test_2_sph():
    sc = Spheres(scatterers=[Sphere(center=[7.1e-6, 7e-6, 10e-6],
                                       n=1.5811+1e-4j, r=5e-07),
                                Sphere(center=[6e-6, 7e-6, 10e-6],
                                       n=1.5811+1e-4j, r=5e-07)])


    holo = calc_holo(schema, sc, theory=Multisphere, scaling=.6)

    assert_obj_close(holo.max(), 1.4140292298443309)
    assert_obj_close(holo.mean(), 0.9955420925817654)
    assert_obj_close(holo.std(), 0.09558537595025796)


@attr('medium')
def test_radial_holos():
    # Check that holograms computed with and w/o radial part of E_scat differ
    sc = Spheres(scatterers=[Sphere(center=[7.1e-6, 7e-6, 10e-6],
                                       n=1.5811+1e-4j, r=5e-07),
                                Sphere(center=[6e-6, 7e-6, 10e-6],
                                       n=1.5811+1e-4j, r=5e-07)])
    thry_nonrad = Multisphere()
    thry_rad = Multisphere(compute_escat_radial = True)

    holo_nonrad = calc_holo(schema, sc, index, wavelen, xpolarization, theory=thry_nonrad)
    holo_rad = calc_holo(schema, sc, index, wavelen, xpolarization, theory=thry_rad)

    # the two arrays should not be equal
    try:
        assert_allclose(holo_nonrad, holo_rad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Holograms computed with and without radial component of scattered electric field are too similar.")


class TestErrors(unittest.TestCase):
    args = (index, wavelen, xpolarization)
    kwargs = {'theory': Multisphere()}

    @attr('medium')
    def test_invalid_when_spheres_very_far_apart(self):
        sphere1 = Sphere(center=[7e0, 7e-6, 10e-6],  n=1.5811+1e-4j, r=5e-07)
        sphere2 = Sphere(center=[6e-6, 7e-6, 10e-6], n=1.5811+1e-4j, r=5e-07)
        scatterer = Spheres(scatterers=[sphere1, sphere2])

        self.assertRaises(
            InvalidScatterer,
            calc_holo,
            schema, scatterer, *self.args, **self.kwargs)

    @attr('medium')
    def test_invalid_when_spheres_too_big(self):
        sphere1 = Sphere(center=[7e-1, 7e-6, 10e-6], n=1.5811+1e-4j, r=5e-02)
        sphere2 = Sphere(center=[6e-1, 7e-6, 10e-6], n=1.5811+1e-4j, r=5e-07)
        scatterer = Spheres(scatterers=[sphere1, sphere2])
        self.assertRaises(
            InvalidScatterer,
            calc_holo,
            schema, scatterer, *self.args, **self.kwargs)

    @attr('fast')
    def test_invalid_when_scatterer_is_sphere(self):
        sphere = Sphere(center=(0, 0, 11e-6))
        self.assertRaises(
            TheoryNotCompatibleError,
            calc_holo,
            schema, sphere, *self.args, **self.kwargs)

    @attr('fast')
    def test_invalid_when_coated_spheres(self):
        # try a coated sphere
        sphere1 = Sphere(
            center=[7e-6, 7e-6, 10e-6],
            n=1.5811+1e-4j,
            r=5e-07)
        sphere2 = Sphere(
            center=[6e-6, 5e-6, 10e-6],
            n=[1.5, 1.6],
            r=[2e-7, 5e-07])
        scatterer = Spheres(scatterers=[sphere1, sphere2])
        self.assertRaises(
            TheoryNotCompatibleError,
            calc_holo,
            schema, scatterer, *self.args, **self.kwargs)


class TestOverlap(unittest.TestCase):
    @attr('fast')
    def test_spheres_raises_warning_when_overlapping(self):
        self.assertWarns(
            OverlapWarning,
            Spheres,
            scatterers=SCATTERERS_LARGE_OVERLAP)

    @attr('fast')
    def test_calc_holo_fails_to_converge_when_overlap_is_large(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc = Spheres(scatterers=SCATTERERS_LARGE_OVERLAP)
            self.assertRaises(
                MultisphereFailure,
                calc_holo,
                schema, sc, index, wavelen, xpolarization)

    @attr('medium')
    def test_calc_holo_succeeds_but_warns_with_small_overlap(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc = Spheres(scatterers=SCATTERERS_SMALL_OVERLAP)
            holo = calc_holo(schema, sc, index, wavelen, xpolarization)
        verify(holo, '2_sphere_allow_overlap')


@attr('fast')
def test_niter():
    sc = Spheres(scatterers=[Sphere(center=[7.1e-6, 7e-6, 10e-6],
                                          n=1.5811+1e-4j, r=5e-07),
                                   Sphere(center=[6e-6, 7e-6, 10e-6],
                                          n=1.5811+1e-4j, r=5e-07)])
    multi = Multisphere(niter = 2)

    assert_raises(MultisphereFailure, calc_holo, schema, sc, index, wavelen, xpolarization, multi)


@attr('medium')
def test_cross_sections():
    wavelen = 1.
    index = 1.
    polarization = [1., 0]
    a = 1./(2 * np.pi) # size parameter 1
    n = 1.5 + 0.1j
    sc = Spheres([Sphere(n = n, r = a, center = [0., 0., a]),
                  Sphere(n = n, r = a, center = [0., 0., -a])])
    thry = Multisphere()
    # this ends up testing the angular dependence of scattering
    # as well as all the scattering coefficients
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', scipy.integrate.IntegrationWarning)
        xsects = calc_cross_sections(sc, illum_wavelen=wavelen, medium_index=index, illum_polarization=ypolarization)

    gold_xsects = np.array([0.03830316, 0.04877015, 0.08707331])
    # calculated directly by SCSMFO. Efficiencies normalized
    # in funny way by SCSMFO, based on "volume mean radius".

    if 'TRAVIS' in os.environ:
        #This test fails on travis for unknown reasons. See github issue #194
        raise SkipTest()
    else:
        assert_allclose(xsects[:3], gold_xsects, rtol = 1e-3)


@attr("fast")
def test_farfield():
    schema = detector_points(theta = np.linspace(0, np.pi/2), phi = np.zeros(50))
    n = 1.59+0.01j
    r = 0.5

    cluster = Spheres([Sphere(n = n, r = r, center = [0., 0., r]),
                       Sphere(n = n, r = r, center = [0., 0., -r])])

    matr = calc_scat_matrix(schema, cluster, illum_wavelen=.66, medium_index=index, theory=Multisphere)


@attr('medium')
def test_wrap_sphere():
    sphere=Sphere(center=[7.1e-6, 7e-6, 10e-6],n=1.5811+1e-4j, r=5e-07)
    sphere_w=Spheres([sphere])
    holo=calc_holo(schema, sphere, theory=Multisphere, scaling=.6)
    holo_w=calc_holo(schema, sphere_w, theory=Multisphere, scaling=.6)
    assert_array_equal(holo,holo_w)


if __name__ == '__main__':
    unittest.main()

