# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
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

from holopy.core import Optics
from holopy.core.tests.common import assert_pickle_roundtrip, get_example_data

from holopy.inference.mcmc import Emcee
from holopy.inference.noise_model import AlphaModel
from holopy.inference.timeseries import TimeIndependent, TimeSeriesAlphaModel
from holopy.inference import prior
from holopy.scattering.scatterer import Sphere
from holopy.scattering.theory import Mie

import numpy as np

def test_prior():
    g = prior.Gaussian(1, 1)
    assert_pickle_roundtrip(g)
    assert_pickle_roundtrip(g.lnprob)

def test_AlphaModelholo_likelihood():
    holo = get_example_data('image0001.yaml')
    s = Sphere(prior.Gaussian(.5, .1), prior.Gaussian(1.6, .1),
               (prior.Gaussian(5, 1), prior.Gaussian(5, 1), prior.Gaussian(5, 1)))
    model = AlphaModel(s, alpha = prior.Gaussian(.7, .1), noise_sd=.01, medium_index=1.33, wavelen=.66, optics=holo.optics)
    assert_pickle_roundtrip(model)


def test_emcee():
    holo = get_example_data('image0001.yaml')
    s = Sphere(prior.Gaussian(.5, .1), prior.Gaussian(1.6, .1),
               (prior.Gaussian(5, 1), prior.Gaussian(5, 1), prior.Gaussian(5, 1)))
    model = AlphaModel(s, alpha = prior.Gaussian(.7, .1), noise_sd=.01, medium_index=1.33, wavelen=.66, optics=holo.optics)
    emcee = Emcee(model, holo)
    assert_pickle_roundtrip(emcee)

def test_TimeSeriesAlphaModel():
    n = TimeIndependent(prior.Gaussian(5, .5))
    assert_pickle_roundtrip(n)
    st = Sphere(n=n, r=TimeIndependent(prior.BoundedGaussian(1.6,.1, 0, np.inf)), center=(prior.Gaussian(10, 1), prior.Gaussian(10, 1), prior.BoundedGaussian(1.6, .1, 0, np.inf)))
    assert_pickle_roundtrip(st)
    noise_sd = .1
    mt = TimeSeriesAlphaModel(st, noise_sd, alpha=prior.Uniform(0, 1), n_frames=2, medium_index=1.33, wavelen=.66, optics=Optics(polarization=(0, 1)))
    assert_pickle_roundtrip(mt)
