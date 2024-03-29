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
Test construction and manipulation of Scatterer objects.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_raises
from holopy.core.tests.common import assert_obj_close

import pytest

from holopy.scattering.scatterer import Sphere, Ellipsoid, Spheres, RigidCluster
from holopy.scattering.errors import InvalidScatterer, OverlapWarning

import warnings

@pytest.mark.fast
def test_Spheres_construction():

    # cluster of multiple spheres
    s1 = Sphere(n = 1.59, r = 5e-7, center = (1e-6, -1e-6, 10e-6))
    s2 = Sphere(n = 1.59, r = 1e-6, center=[0,0,0])
    s3 = Sphere(n = 1.59+0.0001j, r = 5e-7, center=[5e-6,0,0])
    sc = Spheres(scatterers=[s1, s2, s3])

    # test attribute access
    sc.n
    sc.n_real
    sc.n_imag
    sc.r
    sc.x
    sc.y
    sc.z
    sc.centers
    sc.center


@pytest.mark.fast
def test_Spheres_construction_typechecking():
    # heterogeneous composite should raise exception, since a
    # sphere cluster must contain only Spheres
    s1 = Sphere(n = 1.59, r = 5e-7, center = (1e-6, -1e-6, 10e-6))
    s2 = Sphere(n = 1.59, r = 1e-6, center=[0,0,0])
    s3 = Sphere(n = 1.59+0.0001j, r = 5e-7, center=[5e-6,0,0])
    cs = Ellipsoid(n=1.59+0.0001j, r=(5e-7, 5e-7, 8e-7),
                      center=[-5e-6, 0,0])
    with pytest.raises(InvalidScatterer):
        sc = Spheres(scatterers=[s1, s2, s3, cs])

@pytest.mark.fast
def test_Spheres_ovelap_checking():
    s1 = Sphere(n = 1.59, r = 5e-7, center=(1e-6, -1e-6, 10e-6))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', OverlapWarning)
        sc = Spheres([s1, s1, s1])
        assert len(w) > 0


@pytest.mark.fast
def test_Spheres_parameters():
    s1 = Sphere(n = 1.59, r = 5e-7, center=[1e-6, -1e-6, 10e-6])
    s2 = Sphere(n = 1.59, r = 1e-6, center=[0,0,0])
    sc = Spheres(scatterers = [s1, s2])

    assert_equal(sc.parameters, dict([('0:center', [1e-6, -1e-6, 1.0e-05]),
                                      ('0:n', 1.59), ('0:r', 5e-07),
                                      ('1:center', [0, 0, 0]),
                                      ('1:n', 1.59), ('1:r', 1e-06)]))

    sc2 = sc.from_parameters(sc.parameters)

    assert_equal(sc.scatterers[0].r, sc2.scatterers[0].r)
    assert_equal(sc.scatterers[1].r, sc2.scatterers[1].r)
    assert_equal(sc.scatterers[0].n, sc2.scatterers[0].n)
    assert_equal(sc.scatterers[1].n, sc2.scatterers[1].n)
    assert_equal(sc.scatterers[0].center, sc2.scatterers[0].center)
    assert_equal(sc.scatterers[1].center, sc2.scatterers[1].center)


@pytest.mark.fast
def test_Spheres_translation():
    s1 = Sphere(n = 1.59, r = 5, center=[1, -1, 10])
    s2 = Sphere(n = 1.59, r = 1, center=[0,0,0])
    sc = Spheres(scatterers = [s1, s2])

    sc2 = sc.translated(1, 1, 1)

    assert_equal(sc.scatterers[0].r, sc2.scatterers[0].r)
    assert_equal(sc.scatterers[1].r, sc2.scatterers[1].r)
    assert_equal(sc.scatterers[0].n, sc2.scatterers[0].n)
    assert_equal(sc.scatterers[1].n, sc2.scatterers[1].n)
    assert_equal([2, 0, 11], sc2.scatterers[0].center)
    assert_equal([1, 1, 1], sc2.scatterers[1].center)

@pytest.mark.fast
def test_Spheres_rotation():
    s1 = Sphere(n = 1.59, r = 1, center = [1, 0, 0])
    s2 = Sphere(n = 1.59, r = 1, center = [-1, 0, 1])
    sc = Spheres(scatterers = [s1, s2])

    sc2 = sc.rotated(-np.pi/2, 0, 0)

    assert_equal(sc.scatterers[0].r, sc2.scatterers[0].r)
    assert_equal(sc.scatterers[1].r, sc2.scatterers[1].r)
    assert_equal(sc.scatterers[0].n, sc2.scatterers[0].n)
    assert_equal(sc.scatterers[1].n, sc2.scatterers[1].n)
    assert_almost_equal([0, -1, 0], sc2.scatterers[0].center)
    assert_almost_equal([0, 1, 1], sc2.scatterers[1].center)


@pytest.mark.fast
def test_RigidCluster():
    # test construction
    s1 = Sphere(n = 1, center = (1, 0, 0))
    s2 = Sphere(n = 2, center = (-1, 0, 0))
    s3 = Sphere(n = 3, center = (0, 1,0))
    s4 = Sphere(n = 4, center = (0, -1, 0))
    base = Spheres([s1, s2, s3, s4])
    rc = RigidCluster(base)
    assert_obj_close(rc.scatterers, base.scatterers)
    assert_raises(InvalidScatterer, RigidCluster,s1)

    # test transformation
    ts1 = Sphere(n = 1, center = [-1/np.sqrt(2)+1, 2., -1/np.sqrt(2)+3])
    ts2 = Sphere(n = 2, center = [1/np.sqrt(2)+1, 2., 1/np.sqrt(2)+3])
    ts3 = Sphere(n = 3, center = [-1/np.sqrt(2)+1,2., +1/np.sqrt(2)+3])
    ts4 = Sphere(n = 4, center = [1/np.sqrt(2)+1, 2., -1/np.sqrt(2)+3])
    trans = Spheres([ts1,ts2,ts3,ts4])
    trc = RigidCluster(base, rotation=(np.pi/4,np.pi/2,np.pi/2),translation=(1,2,3))
    assert_obj_close(trc.scatterers,trans.scatterers)

    # test parameters, from_parameters
    assert_obj_close(trc.from_parameters(trc.parameters), trans)
