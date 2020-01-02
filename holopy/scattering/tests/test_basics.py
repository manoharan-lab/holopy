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
import numpy as np
import xarray as xr
from numpy.testing import assert_allclose, assert_equal
from nose.plugins.attrib import attr

from holopy.core import detector_grid, detector_points
from holopy.core.tests.common import assert_obj_close
from holopy.core.metadata import update_metadata
from holopy.scattering.scatterer import Sphere, Difference
from holopy.scattering.theory import Mie
from holopy.scattering.interface import calc_intensity, calc_holo, calc_field

# small tests against results from the previous version of holopy


@attr("fast")
def test_calc_field():
    s = Sphere(n=1.59, r=.5, center=(0,0,1))
    t = update_metadata(detector_grid(shape = (2,2), spacing = .1), illum_wavelen = 0.66, medium_index=1.33, illum_polarization = (1,0))
    thry = Mie(False)
    f = calc_field(t, s, 1.33, .66, (1,0), theory=thry)
    assert_obj_close(t.attrs, f.attrs)
    gold = xr.DataArray(np.array([[[ -3.95866810e-01 +2.47924378e+00j,
                                     0.00000000e+00 +0.00000000e+00j,
                                     0.00000000e+00 -0.00000000e+00j],
                                  [ -4.91260953e-01 +2.32779296e+00j,
                                     9.21716363e-20 -5.72226912e-19j,
                                     2.99878926e-18 -1.41959276e-17j]],

                                 [[ -4.89755627e-01 +2.31844748e+00j,
                                     0.00000000e+00 +0.00000000e+00j,
                                     4.89755627e-02 -2.31844748e-01j],
                                  [ -5.71886751e-01 +2.17145168e+00j,
                                     1.72579090e-03 -8.72241140e-03j,
                                     5.70160960e-02 -2.16272927e-01j]]]), dims=['x', 'y', 'vector'], coords={'x':t.x, 'y': t.y, 'vector': ['x', 'y', 'z']})
    assert abs((f - gold).max()) < 5e-9

@attr("fast")
def test_detector_points():
    s = Sphere(n=1.59, r=.5, center=(0,0,0))
    medium_index = 1.33
    illum_wavelen = 0.660
    illum_polarization = (1, 0)

    points_cartesian = detector_points(
        x=[1, 0, -1, 0],
        y=[0, 1, 0, -1],
        z=1)
    points_spherical = detector_points(
        theta=3*np.pi/4,
        phi=[0, np.pi/2, np.pi, 3*np.pi/2],
        r=np.sqrt(2))
    field_cartesian = calc_field(
        points_cartesian, s, medium_index, illum_wavelen, illum_polarization)
    field_spherical = calc_field(
        points_spherical, s, medium_index, illum_wavelen, illum_polarization)
    assert_allclose(field_cartesian, field_spherical)


@attr("fast")
def test_calc_holo():
    s = Sphere(n=1.59, r=.5, center=(0,0,1))
    t = detector_grid(shape = (2,2), spacing = .1)
    thry = Mie(False)
    h = calc_holo(t, s, 1.33, .66, (1,0), theory=thry)
    assert_allclose(h, np.array([[[ 6.51162661],[  5.67743548]],
                                 [[ 5.63554802],[  4.89856241]]]))

@attr("fast")
def test_calc_intensity():
    s = Sphere(n=1.59, r=.5, center=(0,0,1))
    t = detector_grid(shape = (2,2), spacing = .1)
    thry = Mie(False)
    i = calc_intensity(t, s, illum_wavelen=.66, medium_index=1.33, illum_polarization = (1, 0), theory=thry)
    assert_allclose(i, np.array([[[ 6.30336023],  [5.65995739]],
                                 [[ 5.61505927],  [5.04233591]]]))

@attr("fast")
def test_csg_construction():
    s = Sphere(n = 1.6, r=.5, center=(0, 0, 0))
    st = s.translated(.4, 0, 0)
    pacman = Difference(s, st)
    assert_allclose(pacman.bounds, [(-.5, .5), (-.5, .5), (-.5, .5)])

