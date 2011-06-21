# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca
# W. Perry, Jerome Fung, and Ryan McGorty
#
# This file is part of Holopy.
#
# Holopy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Holopy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Holopy.  If not, see <http://www.gnu.org/licenses/>.

'''
Proposal for new function structure for fitting in the form of tests.

Don't expect these tests to pass for a while

'''

import numpy as np
import holopy
import nose
from nose.tools import raises, assert_raises
from numpy.testing import assert_, assert_equal, assert_array_almost_equal
from nose.tools import with_setup
import os
import string
from nose.plugins.attrib import attr

from holopy.model.scatterer import Sphere, Cluster, Composite

def setup_optics():
    # set up optics class for use in several test functions
    global optics
    wavelen = 658e-9
    polarization = [0., 1.0]
    divergence = 0
    pixel_scale = [.1151e-6, .1151e-6]
    index = 1.33
    
    optics = holopy.optics.Optics(wavelen=wavelen, index=index,
                                  pixel_scale=pixel_scale,
                                  polarization=polarization,
                                  divergence=divergence)
    
def teardown_optics():
    global optics
    del optics

    
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_fit_mie_single():
    path = os.path.abspath(holopy.__file__)
    path = string.rstrip(path, chars='__init__.pyc')+'tests/exampledata/'
    holo = normalize(holopy.load(path + 'image0001.npy'))
    
    sc = holopy.model.scatterer.Sphere(n=1.59, r=8.5e-7, x=.567e-5, y=.576e-5, z=15e-6)

    minimizer = holopy.minimizer.nmpfit()

    constraints = holopy.constraints(lower = [1.0, 1e-8, 0., 0., 0., 1e-1],
                                     upper = [2.0, 1e-5, 1e-5, 1e-5, 1e-4, 1.0])

    # Do we specify the initial guess for alpha with the theory?  It doesn't
    # quite make sense here, but it makes less sense other places - tgd
    theory = holopy.model.theory.Mie(holo.shape)
    

    fitresult = fit(holo, sc, theory, minimizer, constraints)


def test_fit_dimer():
    path = os.path.abspath(holopy.__file__)
    path = string.rstrip(path, chars='__init__.pyc')+'tests/exampledata/'
    holo = normalize(holopy.load(path + image0002))

    sc = Cluster(com=[17.3e-6,17.3e-6,20.7e-6], n1=1.59, n2=1.59, r1=.65e-6,
                 r2=.65e-6, gap=10e-9, beta=-28.5, gamma=-14.87)

    theory = holopy.model.theory.TMatrix(holo.shape)


    minimizer = holopy.minimizer.nmpfit()

    # How should we specify constraints for more general things?  Does the
    # variable constraint belong in with the object?
    # I don't see a good way to define constraints decoupled from the geometry
    # in general -tgd
    constraints = []
    
    fitresult = fit(holo, sc, theory, minimizer, constraints)

    # Maybe do constraints as
    fitresult = fit(holo, sc, theory, minimizer, sc.get_constraints())
    # Though in this case the fit could just get constraints from sc directly

def test_fit_superposition():
    # TODO: connect this with real data
    holo = load('something.tif')

    sc = Composite(Sphere(n=1.59, r=8.5e-7, x=.567e-5, y=.576e-5, z=15e-6),
                   Sphere(n=1.59, r=8.5e-7, x=.587e-5, y=.586e-5, z=16e-6))

    theory = holopy.model.theory.Mie(holo.shape)

    minmizer = holopy.minimizer.nmpfit()

    fitresult = fit(holo, sc, theory, minimizer, sc.get_constraints())

def test_fit_general():
    # TODO: connect this with some kind of data
    holo = load('something.tif')

    sc = Composite(Cluster(com=[17.3e-6,17.3e-6,20.7e-6], n1=1.59, n2=1.59, r1=.65e-6,
                           r2=.65e-6, gap=10e-9, beta=-28.5, gamma=-14.87),
                   CoatedSphere(n1=1.59, n2=1.4, r1=6.5e-7, r2=1e-6, x=.567e-5,
                                y=.576e-5, z=15e-6))

    theory = holopy.model.theory.RayleighGansDiscritize(holo.shape,
                                                        grid=[100,100,100])

    minmizer = holopy.minimizer.Genetic()

    fitresult = fit(holo, sc, theory, minimizer, sc.get_constraints)

    
    

    
    
