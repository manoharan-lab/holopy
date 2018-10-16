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
"""
Test the ability of the mielens calculations to interface with holopy.

.. moduleauthor:: Brian D. Leahy <bleahy@seas.harvard.edu>
"""

import os
import yaml
import unittest

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_almost_equal,
                           assert_raises, assert_equal, assert_allclose)
from nose.plugins.attrib import attr

from ..scatterer import Sphere, Spheres, Ellipsoid
from ..theory import Mie, MieLens, mielensfunctions

from ..errors import TheoryNotCompatibleError, InvalidScatterer
from ...core.metadata import (detector_grid, detector_points, to_vector,
                              sphere_coords, update_metadata)
from ...core.process import subimage
from .common import (sphere, xschema, scaling_alpha, yschema, xpolarization,
                     ypolarization)
from .common import x, y, z, n, radius, wavelen, index
from ...core.tests.common import assert_obj_close, verify

from ..calculations import (calc_field, calc_holo, calc_intensity,
                            calc_scat_matrix, calc_cross_sections)


TOLS = {'atol': 1e-13, 'rtol': 1e-13}
MEDTOLS = {"atol": 1e-6, "rtol": 1e-6}


class TestMieLens(unittest.TestCase):
    @attr("fast")
    def test_does_not_crash(self):
        theory = MieLens()
        holo = calc_holo(xschema, sphere, index, wavelen, xpolarization,
                         theory=theory)
        self.assertTrue(holo is not None)

    @attr("fast")
    def test_holopy_hologram_equal_to_exact_calculation(self):
        # Checks that phase shifts and wrappers for mielens are correct
        illum_wavelength = 0.66  # 660 nm red light
        k = 2 * np.pi / illum_wavelength
        center = (10, 10, 5.)

        kwargs = {'particle_kz': center[2] * k,
                  'index_ratio': 1.2,
                  'size_parameter': 0.5 * k,
                  'lens_angle': 1.0} # using the default mielens lens angle
        detector = detector_grid(10, 2.0)
        x = detector.x.values.reshape(-1, 1) - center[0]
        y = detector.y.values.reshape(1, -1) - center[1]

        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        calculator = mielensfunctions.MieLensCalculator(**kwargs)
        scatterer = Sphere(n=kwargs['index_ratio'],
                           r=kwargs['size_parameter'] / k,
                           center=center)

        holo_calculate = calculator.calculate_total_intensity(k * rho, phi)
        holo_holopy = calc_holo(
            detector, scatterer, illum_wavelen=illum_wavelength,
            medium_index=1., illum_polarization=(1, 0), theory=MieLens())

        is_ok = np.allclose(holo_calculate, holo_holopy.values.squeeze(),
                            **TOLS)
        self.assertTrue(is_ok)

    @attr('fast')  #??
    def _test_large_sphere(self):
        scatterer = Sphere(n=1.2, r=5.0, center=(10, 10, 10))
        detector = detector_grid(100, 0.2)
        holo = calc_holo(detector, scatterer, illum_wavelen=0.66,
                         medium_index=1.0, illum_polarization=(1, 0),
                         theory=MieLens())
        assert_obj_close(np.array(hl[0:2,0:2]),large_sphere_gold)

    @attr('fast')
    def test_mielens_is_close_to_mieonly(self):
        """Tests that a mielens hologram is similar to a mie-only hologram."""
        theory_mielens = MieLens()
        theory_mieonly = Mie()

        holo_mielens = calc_holo(
            xschema, sphere, index, wavelen, xpolarization,
            theory=theory_mielens)
        holo_mieonly = calc_holo(
            xschema, sphere, index, wavelen, xpolarization,
            scaling=scaling_alpha, theory=theory_mieonly)

        # the two arrays should not be equal
        self.assertFalse(np.allclose(holo_mielens, holo_mieonly, **TOLS))

        # but their max and min values should be close:
        ptp_close_ish = np.isclose(
            holo_mielens.values.ptp(), holo_mieonly.values.ptp(), atol=0.1)
        # and their median should be close:
        median_close_ish = np.isclose(
            np.median(holo_mielens), np.median(holo_mieonly), atol=0.1)

        self.assertTrue(ptp_close_ish)
        self.assertTrue(median_close_ish)

    def test_mielens_x_polarization_differs_from_y(self):
        # test holograms for orthogonal polarizations; make sure they're
        # not the same, nor too different from one another.
        theory = MieLens()
        holo_x = calc_holo(xschema, sphere, index, wavelen,
                           illum_polarization=xpolarization, theory=theory)
        holo_y = calc_holo(yschema, sphere, index, wavelen,
                           illum_polarization=ypolarization, theory=theory)

        # the two arrays should not be equal
        self.assertFalse(np.allclose(holo_x, holo_y, **MEDTOLS))

        # but their max and min values should be very close
        self.assertFalse(np.isclose(holo_x.max(), holo_y.max(), **MEDTOLS))
        self.assertFalse(np.isclose(holo_x.min(), holo_y.min(), **MEDTOLS))


@attr('fast')
def _test_Mie_multiple():
    s1 = Sphere(n = 1.59, r = 5e-7, center = (1e-6, -1e-6, 10e-6))
    s2 = Sphere(n = 1.59, r = 1e-6, center=[8e-6,5e-6,5e-6])
    s3 = Sphere(n = 1.59+0.0001j, r = 5e-7, center=[5e-6,10e-6,3e-6])
    sc = Spheres(scatterers=[s1, s2, s3])
    thry = Mie(False)

    schema = yschema
    fields = calc_field(schema, sc, index, wavelen, ypolarization, thry)

    verify(fields, 'mie_multiple_fields')
    calc_intensity(schema, sc, index, wavelen, ypolarization, thry)

    holo = calc_holo(schema, sc, index, wavelen, theory=thry)
    verify(holo, 'mie_multiple_holo')
    # should throw exception when fed a ellipsoid
    el = Ellipsoid(n = 1.59, r = (1e-6, 2e-6, 3e-6), center=[8e-6,5e-6,5e-6])
    with assert_raises(TheoryNotCompatibleError) as cm:
        calc_field(schema, el, index, wavelen, theory=Mie)
    assert_equal(str(cm.exception), "Mie scattering theory can't handle "
                 "scatterers of type Ellipsoid")
    assert_raises(TheoryNotCompatibleError, calc_field, schema, el, index, wavelen, xpolarization, Mie)
    assert_raises(TheoryNotCompatibleError, calc_intensity,
                  schema, el, index, wavelen, xpolarization, Mie)
    assert_raises(TheoryNotCompatibleError, calc_holo, schema, el, index, wavelen, xpolarization, Mie)


if __name__ == '__main__':
    unittest.main()
