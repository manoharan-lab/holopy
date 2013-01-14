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
Test fitting a single-particle hologram using OpenOpt solvers
'''

import numpy as np
import os
import string
import pylab

import holopy
from scatterpy.theory import mie
from holopy.third_party import nmpfit
from holopy.process import normalize
from nose.plugins.attrib import attr

from openopt import NLP, NLLSP, GLP
# tested with openopt and DerApproximator 0.33, installed using
# easy_install on Ubuntu x64

# these are the "gold" values; should correspond to fit results
# In order: real index, imag index, radius, x, y, z, alpha, objective
# function, fit status
gold_single = np.array([1.5768, 0.0001, 6.62e-7, 5.54e-6, 5.79e-6,
                        14.2e-6, 0.6398, 7.119, 2]) 

# define optical train
wavelen = 658e-9
polarization = [0., 1.0]
divergence = 0
pixel_scale = [.1151e-6, .1151e-6]
index = 1.33

optics = holopy.optics.Optics(wavelen=wavelen, index=index,
                              pixel_scale=pixel_scale,
                              polarization=polarization,
                              divergence=divergence)

# initial guess
scaling_alpha = .6
radius = .85e-6
n_particle_real = 1.59
n_particle_imag = 1e-4
x = .576e-05
y = .576e-05
z = 15e-6

path = os.path.abspath(holopy.__file__)
path = string.rstrip(path, chars='__init__.pyc')+'tests/exampledata/'

# load the target file
# be sure to normalize, or fit will fail
holo = normalize(holopy.load(path + 'image0001.npy'))

# or fit a calculated hologram
#holo = mie.forward_holo(100, optics, gold_single[0], gold_single[1],
#                        gold_single[2], gold_single[3],
#                        gold_single[4], gold_single[5], gold_single[6])

# THIS IS IMPORTANT!
# specify typical orders of magnitude of variables passed to forward
# function through 'scale'.  This is so all variables passed to the
# forward function will be order(1), which is important for certain
# solvers that implicitly assume this.  If we don't do this, the
# solver might try to calculate a derivative by making an order(1)
# change in, say, the radius, which would stymie the forward function
# as it takes forever to calculate Mie scattering from a 1-meter
# large bead.
#
# OpenOpt lets you specify this through p.scale, but some solvers
# (e.g. scipy_cobyla) ignore this.  So I found that it's better to
# specify it and incorporate it into the objective function
# explicitly. 
scale = np.array([10, 1e7, 1e6, 1e6, 1e6, 10])

# scaled initial guess
x0 = np.array([n_particle_real, radius, x, y, z, scaling_alpha])*scale

# scaled lower and upper bounds
# I specify tighter bounds than for NMPFIT, in case I want to try a
# global minimizer.
#lb = [1.0, 0., -np.Inf, -np.Inf, -np.Inf, 0.0] # NMPFIT bounds
#ub = [np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, 1.0]     # NMPFIT bounds
lb = [1.0, 1e-8, 0., 0., 0., 1e-1]*scale
ub = [2.0, 1e-5, 1e-5, 1e-5, 1e-4, 1.0]*scale

# define the residual function
def residual(x):

    x = x/scale

    # uncomment to try fitting for imaginary index too
    #calculated = mie.forward_holo(holo.shape[0], optics, x[0], x[1],
    #                              x[2], x[3], x[4], x[5], x[6])
    calculated = mie.forward_holo(holo.shape[0], optics,
                                  x[0], n_particle_imag, 
                                  x[1], x[2], x[3], x[4], x[5])

    derivates = holo - calculated
    resid = derivates.ravel()
    # print sum of squares and param values for debugging
    print "sum of squares: ", np.dot(resid, resid)
    print "params: ", x

    # use np.dot(resid, resid) with solvers in NLP or GLP classes.
    # use return resid with solvers in NLLSP
    return np.dot(resid,resid)
    #return resid

@attr('slow')
def test_openopt():
    p = NLP(residual, x0, lb=lb, ub=ub,
            iprint=1, plot=True)
    # uncomment the following (and set scale = 1 above) to use openopt's
    # scaling mechanism.  This only seems to work with a few solvers, though.
    #p.scale = np.array([1, 1e6, 1e6, 1e6, 1e6, 1])

    r = p.solve('ralg')             # OpenOpt solver, seems to work well,
                                    # though not too fast (~50 seconds)

    # Or try some other solvers.  Time estimates are relative

    # r = p.solve('scipy_cobyla')     # derivative-free but really slow
                                    # convergence (>5 minutes)
    # r = p.solve('scipy_lbfgsb')     # seems to work well (~20 sec)
    # r = p.solve('scipy_tnc')        # truncated Newton, converges but
                                    # gets wrong answer
    # r = p.solve('scipy_slsqp')      # sequential least squares; works
                                    # and converges pretty quickly (~8 sec)
    # r = p.solve('sqlcp')            # does not converge
    # r = p.solve('gsubg')            # converges but slowly (~6 min)
    # r = p.solve('lincher')          # doesn't work; throws error

    # Global solvers
    # r = p.solve('galileo')          # genetic algorithm
    # r = p.solve('de')               # differential evolution; could not
                                    # get to converge within iteration limit


    # comment out lb and ub in the NLP statement above to try these
    # unconstrained solvers
    # r = p.solve('scipy_powell')     # no derivatives; converges in ~30 s
    # r = p.solve('scipy_fmin')       # Nelder-Mead simplex; converges in
                                    # ~30 s
    # r = p.solve('scipy_bfgs')       # Broyden method; converges faster
                                    # than above two (~13 s)
    # r = p.solve('scipy_cg')         # conjugate gradient; converges in ~45 s
    # r = p.solve('scipy_ncg')        # could not get to converge

    # redefine objective function to return a vector to use unconstrained
    # Levenberg-Marquardt.  Also change from NLP to NLLSP above
    # r = p.solve('scipy_leastsq')    # fast (2-3 seconds)
