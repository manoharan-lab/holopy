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
The tests here are for fitting holograms of single particles, dimers,
and trimers

Mie (single-particle, Fortran based)
T-Matrix (dimer and trimer, Fortran based)

In the process of testing the fitting, we vicariously test:
- io (loading holograms)
- calculating holograms with various scattering models
- writing the results out to a file

.. moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>
'''

import sys
import os
hp_dir = (os.path.split(sys.path[0])[0]).rsplit(os.sep, 1)[0]
sys.path.append(hp_dir)
import numpy as np
import holopy
import nose
from numpy.testing import assert_, assert_equal, assert_array_almost_equal, assert_array_less
import string
from nose.plugins.attrib import attr

gold_single = np.array([1.5768,0.0001,6.62e-7,5.54e-6,5.79e-6,14.2e-6,0.6398,7.119])
gold_dimerslow = np.array([1.6026,1.6026,0.00001,0.00001,6.857e-7,6.964e-7,
    1.6998e-5,1.739e-5,2.0927e-5,1,-29.78,-13.83,5.447e-25,16.722])
gold_dimerfast = np.array([1.5999,1.5999,0.00001,0.00001,6.773e-7,6.651e-7,
    1.7186e-5,1.735e-5,2.0670e-5,0.787,-30.791,-14.038,0.00,53.655,5.000,0.000])
gold_trimerslow = np.array([1.5894,1.598,1.599,1.00e-5,1.00e-5,1.00e-5,5.00e-7,5.00e-7,
    5.00e-7,6.00e-6,6.00e-6,7.02e-6,0.60948,40.678,12.0806,4.438,4.795])
gold_trimerfast = np.array([1.5898,1.598,1.599,1.00e-5,1.00e-5,1.00e-5,5.00e-7,5.00e-7,
    5.00e-7,6.00e-6,6.00e-6,7.02e-6,0.60865,41.4538,11.113,3.638,4.8098,5.00,0.00])

class TestFit:

    def test_mie_noisysingle(self):
        path = os.path.abspath(holopy.__file__)
        path = string.rstrip(path, chars='__init__.pyc')+'tests/exampledata/'
        input_path = path + 'Mie_input_deck.yaml'
        result_path = path + 'Mie/fit_result.tsv'       
        holopy.fit(input_path)
        fit_result = np.loadtxt(result_path,skiprows=2,
            usecols=[1,2,3,4,5,6,7,8,9])
        assert_array_almost_equal(fit_result[0:8],gold_single,
            decimal=3,err_msg='Mie fit results from the single particle are not approx. equal to the standard fit results.')
        assert_array_less(fit_result[8],5)      

    @attr('slow')
    def test_tmatrix_noisydimer_slow(self):
        path = os.path.abspath(holopy.__file__)
        path = string.rstrip(path, chars='__init__.pyc')+'tests/exampledata/'
        input_path = path + 'TmatDimer_input_deck_slow.yaml'
        result_path = path + 'TmatDimerSlow/fit_result.tsv'    
        holopy.fit(input_path)
        fit_result = np.loadtxt(result_path,skiprows=2,
            usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
        assert_array_almost_equal(fit_result[0:14],gold_dimerslow,
            decimal=3,err_msg='Fit results from the dimer are not approx. equal to the standard fit results.')
        assert_array_less(fit_result[14],5)
        assert_equal(fit_result[15],0)

    def test_tmatrix_noisydimer_fast(self): #this fit stops after 5 iterations to be a faster fit test
        path = os.path.abspath(holopy.__file__)
        path = string.rstrip(path, chars='__init__.pyc')+'tests/exampledata/'
        input_path = path + 'TmatDimer_input_deck_fast.yaml'
        result_path = path + 'TmatDimerFast/fit_result.tsv'    
        holopy.fit(input_path)
        fit_result = np.loadtxt(result_path,skiprows=2,
            usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
        assert_array_almost_equal(fit_result,gold_dimerfast,
            decimal=3,err_msg='Fit results from the dimer are not approx.' 
            +'equal to the standard fit results.')

    @attr('slow')  
    def test_tmatrix_noisytrimer_slow(self):
        path = os.path.abspath(holopy.__file__)
        path = string.rstrip(path, chars='__init__.pyc')+'tests/exampledata/'
        input_path = path + 'TmatTrimer_input_deck_slow.yaml'
        result_path = path + 'TmatTrimerSlow/fit_result.tsv'    
        holopy.fit(input_path)
        fit_result = np.loadtxt(result_path,skiprows=2,
            usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]) 
        assert_array_almost_equal(fit_result[0:17],gold_trimerslow,
            decimal=3,err_msg='Fit results from the trimer are not approx. equal to the standard fit results.')
        assert_array_less(fit_result[17],5)
        assert_equal(fit_result[18],0)   
        
    def test_tmatrix_noisytrimer_fast(self):
        path = os.path.abspath(holopy.__file__)
        path = string.rstrip(path, chars='__init__.pyc')+'tests/exampledata/'
        input_path = path + 'TmatTrimer_input_deck_fast.yaml'
        result_path = path + 'TmatTrimerFast/fit_result.tsv'    
        holopy.fit(input_path)
        fit_result = np.loadtxt(result_path,skiprows=2,
            usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]) 
        assert_array_almost_equal(fit_result,gold_trimerfast,
            decimal=3,err_msg='Fit results from the trimer are not approx. equal to the standard fit results.')
