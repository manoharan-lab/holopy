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
Base class for scattering theories.  Implements python-based
calc_intensity and calc_holo, based on subclass's calc_field

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""
import unittest

import numpy as np
import xarray as xr
from nose.plugins.attrib import attr

from holopy.scattering import Sphere, Spheres, calc_holo
from holopy.scattering.interface import prep_schema
from holopy.core.metadata import detector_grid, update_metadata, to_vector
from holopy.inference import prior, AlphaModel
from holopy.core.tests.common import (
    assert_equal, assert_obj_close, assert_allclose)


class TestHologramCalculation(unittest.TestCase):
    @attr("medium")
    def test_calc_holo_with_twocolor_index(self):
        indices = dict([('red',1.5),('green',2)])
        radius = 0.5
        center = (1, 1, 1)
        illum_wavelen = dict([('red', 0.66), ('green', 0.52)])

        sphere_red = Sphere(n=indices['red'], r=radius, center=center)
        sphere_green = Sphere(n=indices['green'], r=radius, center=center)
        sphere_both = Sphere(n=indices, r=radius, center=center)

        schema_single_color = update_metadata(
            detector_grid(shape=2, spacing=1),
            illum_polarization=(0,1),
            medium_index=1.3)
        schema_two_colors = update_metadata(
            detector_grid(
                shape=2,spacing=1,extra_dims={'illumination':['red','green']}),
            illum_polarization=(0,1),
            medium_index=1.3)

        red_hologram = calc_holo(
            schema_single_color, sphere_red, illum_wavelen=illum_wavelen['red'])
        green_hologram = calc_holo(
            schema_single_color, sphere_green,
            illum_wavelen=illum_wavelen['green'])
        both_hologram = calc_holo(
            schema_two_colors,sphere_both, illum_wavelen=illum_wavelen)

        joined = np.concatenate([
            np.array([red_hologram.values]),
            np.array([green_hologram.values])])
        assert_equal(both_hologram.values, joined)

    @attr("fast")
    def test_calc_holo_with_twocolor_alpha(self):
        detector = detector_grid(
            5, 1, extra_dims={'illumination': ['red', 'green']})
        scatterer = Sphere(
            r=0.5, n={'red': 1.5, 'green': 1.6}, center=(2, 2, 2))
        alpha = {'red': 0.8, 'green': 0.9}
        result = calc_holo(
            detector, scatterer, scaling=alpha, illum_polarization=(0, 1),
            illum_wavelen={'red': 0.66, 'green': 0.52}, medium_index=1.33)
        assert result is not None

    @attr("fast")
    def test_calc_holo_with_twocolor_priors(self):
        detector = detector_grid(
            5, 1, extra_dims={'illumination': ['red', 'green']})
        index = {
            'red': prior.Uniform(1.5, 1.6),
            'green': prior.Uniform(1.5, 1.6)}
        scatterer = Sphere(r=0.5, n=index, center=(2,2,2))
        alpha = {'red': prior.Uniform(0.6, 1), 'green': prior.Uniform(0.6, 1)}
        model = AlphaModel(scatterer, alpha, illum_polarization=(0, 1),
                           illum_wavelen={'red': 0.66, 'green': 0.52},
                           medium_index=1.33)
        result = model.forward(model.initial_guess, detector)
        assert result is not None

@attr("medium")
def test_prep_schema():
    sch_f = detector_grid(shape=5,spacing=1)
    sch_x = detector_grid(shape=5,spacing=1,extra_dims={'illumination':['red','green','blue']})

    wl_f = 0.5
    wl_l = [0.5,0.6,0.7]
    wl_d = dict([('red', 0.5), ('green', 0.6), ('blue', 0.7)])
    wl_x = xr.DataArray([0.5,0.6,0.7],dims='illumination',coords={'illumination':['red','green','blue']})

    pol_f = (0,1)
    pol_d = dict([('red', (0,1)), ('green', (1,0)), ('blue', (0.5,0.5))])

    pol_x = xr.concat([to_vector((0,1)),to_vector((1,0)),to_vector((0.5,0.5))], wl_x.illumination)

    all_in = prep_schema(sch_x,1,wl_x,pol_x)

    assert_obj_close(prep_schema(sch_x,1,wl_d,pol_d),all_in)
    assert_obj_close(prep_schema(sch_x,1,wl_l,pol_d),all_in)
    assert_obj_close(prep_schema(sch_f,1,wl_x,pol_x),all_in)


if __name__ == '__main__':
    unittest.main()

