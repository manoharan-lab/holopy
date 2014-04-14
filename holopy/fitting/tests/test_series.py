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
from __future__ import division

import tempfile

import numpy as np
import warnings

from nose.tools import nottest, assert_raises
from nose.plugins.attrib import attr
from numpy.testing import assert_equal, assert_approx_equal, assert_allclose
from ...scattering.scatterer import Sphere, Spheres, Scatterer
from ...scattering.theory import Mie, Multisphere, DDA
from ...core import Optics, ImageSchema, load, save
from ...core.process import normalize
from ...core.helpers import OrderedDict
from .. import fit, Parameter, ComplexParameter, par, Parametrization, Model
from ..fit_series import fit_series
from ...core.tests.common import (assert_obj_close, get_example_data_path)

from ..errors import InvalidMinimizer


gold_sphere = Sphere(n=1.58, r=5e-7, center=[5.5308956e-6,5.7935436e-6,1.337183988e-5])
gold_alpha = .6497

@attr('medium')
def test_fit_series():

    par_s = Sphere(center = (par(guess = 5.5e-6, limit = [0,10e-6]), par(5.8e-6, [0, 10e-6]), par(13.3e-6, [5e-6, 15e-6])),
               r = .5e-6, n = 1.58)
    model = Model(par_s, Mie.calc_holo, alpha = gold_alpha)

    opticsinfo = Optics(wavelen = .658e-6, polarization = [1, 0], index = 1.33)
    px_size = .1151e-6

    inf = [get_example_data_path('image0001.yaml'),
            get_example_data_path('image0001.yaml')]

    np.random.seed(40)

    with warnings.catch_warnings() as w:
        #test with no saving
        warnings.simplefilter('ignore')
        res = fit_series(model, inf, opticsinfo, px_size, use_random_fraction=.01)

    assert_obj_close(res[-1].scatterer, gold_sphere, rtol = 1e-2)
