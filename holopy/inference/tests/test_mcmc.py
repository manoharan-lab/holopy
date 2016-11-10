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



import warnings
import numpy as np
from numpy.testing import assert_equal
from holopy.inference import prior, mcmc
from holopy.core.process import normalize
from holopy.core.tests.common import assert_obj_close, get_example_data
from holopy.scattering import Sphere, Mie
from holopy.fitting import model
from holopy.inference.noise_model import AlphaModel,NoiseModel

#GOLD:log(sqrt(0.5/pi))-1/2
gold_sigma=-1.4189385332

#GOLD:Gaussian distribution - depends on np seed
prior_dist=np.array([[-0.6075477],[-0.12613641],[-0.68460636]])

#GOLD:inference result - depends on both seeds
gold_alpha=np.array([0.695876])
gold_nsteps=10
gold_frac=0.925


def test_Emcee_Class():
    np.random.seed(40)
    scat = model.Parametrization(0,[prior.Gaussian(0,1)])
    mod = model.BaseModel(scat)
    e = mcmc.Emcee(mod,[],nwalkers=3)
    assert_equal(e.nwalkers,3)
    assert_obj_close(e.make_guess(),prior_dist)

def test_NoiseModel_lnprior():
    scat=Sphere(r=prior.Gaussian(1,1),n=prior.Gaussian(1,1),center=[10,10,10])
    mod=NoiseModel(scat, noise_sd=.1)
    assert_obj_close(mod.lnprior([0,0]),gold_sigma*2)

def test_subset_tempering():
    np.random.seed(40)
    holo = normalize(get_example_data('image0001'))
    scat = Sphere(r=0.65e-6,n=1.58,center=[5.5e-6,5.8e-6,14e-6])
    mod = AlphaModel(scat,noise_sd=.1, alpha=prior.Gaussian(0.7,0.1))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        inf=mcmc.subset_tempering(mod,holo,final_len=10,nwalkers=4,stages=1,stage_len=10,threads=None, verbose=False,seed=40)
    assert_obj_close(inf.most_probable_values(),gold_alpha, rtol=1e-5)
    assert_equal(inf.n_steps,gold_nsteps)
    assert_obj_close(inf.acceptance_fraction,gold_frac)
    assert_obj_close(float(inf.data_frame(burn_in=6)[1:2].alpha),gold_alpha, rtol=1e-1)
