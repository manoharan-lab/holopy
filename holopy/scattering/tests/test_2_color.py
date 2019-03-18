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
from collections import OrderedDict

import numpy as np
import xarray as xr
from nose.plugins.attrib import attr

from .. import Sphere, Spheres, calc_holo
from ...core.metadata import detector_grid, update_metadata, to_vector
from ...inference import prior
from ..calculations import prep_schema
from ...core.tests.common import assert_equal, assert_obj_close, assert_allclose


@attr("medium")
def test_hologram():
    r_sph = Sphere(n = 1.5, r=.5, center=(1,1,1))
    g_sph = Sphere(n = 2, r=.5, center=(1,1,1))
    b_sph = Sphere(n = OrderedDict([('red',1.5),('green',2)]), r=.5, center=(1,1,1))

    sch1 = update_metadata(detector_grid(shape=2, spacing=1), illum_polarization=(0,1), medium_index=1.3)
    sch2 = update_metadata(detector_grid(shape=2,spacing=1,extra_dims={'illumination':['red','green']}),
                illum_polarization=(0,1),medium_index=1.3)
    
    red = calc_holo(sch1, r_sph, illum_wavelen = .66).values
    grn = calc_holo(sch1, g_sph, illum_wavelen = .52).values
    joined = np.concatenate([np.array([red]),np.array([grn])])
    both = calc_holo(sch2,b_sph, illum_wavelen=OrderedDict([('red',0.66),('green',0.52)]))
    assert_equal(both.values, joined)


def test_select():
    s = Sphere(n=xr.DataArray([1.5,1.7],dims='ill',coords={'ill':['r','g']}),center=[0,0,0],r=0.5)
    assert_equal(s.select({'ill':'g'}),Sphere(n=1.7,center=[0,0,0],r=0.5))

    ss = Spheres([s, s.translated([1,1,1])])
    assert_equal(ss.select({'ill':'g'}),Spheres([Sphere(n=1.7,center=[0,0,0],r=0.5),Sphere(n=1.7,center=[1,1,1],r=0.5)]))


@attr("medium")
def test_prep_schema():
    sch_f = detector_grid(shape=5,spacing=1)
    sch_x = detector_grid(shape=5,spacing=1,extra_dims={'illumination':['red','green','blue']})
    
    wl_f = 0.5
    wl_l = [0.5,0.6,0.7]
    wl_d = OrderedDict([('red', 0.5), ('green', 0.6), ('blue', 0.7)])
    wl_x = xr.DataArray([0.5,0.6,0.7],dims='illumination',coords={'illumination':['red','green','blue']})
    
    pol_f = (0,1)
    pol_d = OrderedDict([('red', (0,1)), ('green', (1,0)), ('blue', (0.5,0.5))])

    pol_x = xr.concat([to_vector((0,1)),to_vector((1,0)),to_vector((0.5,0.5))], wl_x.illumination)

    all_in = prep_schema(sch_x,1,wl_x,pol_x)

    assert_obj_close(prep_schema(sch_x,1,wl_d,pol_d),all_in)
    assert_obj_close(prep_schema(sch_x,1,wl_l,pol_d),all_in)
    assert_obj_close(prep_schema(sch_f,1,wl_x,pol_x),all_in)
    
