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

#The gold standards are the fit results normalized to be between 0 and 10
gold_single = np.array([1.577,1.000,6.623,5.536,5.794,1.419,6.398,7.119])
gold_dimerslow = np.array([1.603,1.603,1.000,1.000,6.857,6.964,
    1.700,1.739,2.093,1,-2.978,-1.383,0.000,1.672])
gold_dimerfast = np.array([1.600,1.600,1.000,1.000,6.773,6.651,
    1.719,1.735,2.067,7.876,-3.079,-1.404,0.000,5.366,5.000,0.000])
gold_trimerslow = np.array([1.589,1.598,1.599,1.000,1.000,1.000,5.000,5.000,
    5.000,6.001,5.999,7.018,6.095,4.068,1.208,4.438,4.795])
gold_trimerfast = np.array([1.590,1.598,1.599,1.000,1.000,1.000,5.000,5.000,
    5.000,6.001,5.999,7.020,6.087,4.145,1.111,3.638,4.810,5.000,0.000])

class TestFit:

    def test_mie_noisysingle(self):
        path = os.path.abspath(holopy.__file__)
        path = string.rstrip(path, chars='__init__.pyc')+'tests/exampledata/'
        input_path = path + 'Mie_input_deck.yaml'
        result_path = path + 'Mie/fit_result.tsv'       
        holopy.fit(input_path)
        fit_result = np.loadtxt(result_path,skiprows=2,
            usecols=[1,2,3,4,5,6,7,8,9])
        #multiply results by powers of 10 to make them be of the same magnitude
        #so that we may use just one tolerance for all array values.
        assert_array_almost_equal(fit_result[0:8]*[1,10**4,10**7,
            10**6,10**6,10**5,10,1],gold_single,
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
        #multiply results by powers of 10 to make them be of the same magnitude
        #so that we may use just one tolerance for all array values.
        assert_array_almost_equal(fit_result[0:14]*[1,1,10**5,10**5,
            10**7,10**7,10**5,10**5,10**5,1,10**-1,10**-1,10**9,10**-1],gold_dimerslow,
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
        #multiply results by powers of 10 to make them be of the same magnitude
        #so that we may use just one tolerance for all array values.
        assert_array_almost_equal(fit_result*[1,1,10**5,10**5,
            10**7,10**7,10**5,10**5,10**5,10,10**-1,10**-1,10**9,10**-1,1,1],gold_dimerfast,
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
        #multiply results by powers of 10 to make them be of the same magnitude
        #so that we may use just one tolerance for all array values.
        assert_array_almost_equal(fit_result[0:17]*[1,1,1,10**5,10**5,10**5,10**7,10**7,
            10**7,10**6,10**6,10**6,10,10**-1,10**-1,1,1],gold_trimerslow,
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
        #multiply results by powers of 10 to make them be of the same magnitude
        #so that we may use just one tolerance for all array values.
        assert_array_almost_equal(fit_result*[1,1,1,10**5,10**5,10**5,10**7,10**7,
            10**7,10**6,10**6,10**6,10,10**-1,10**-1,1,1,1,1],gold_trimerfast,
            decimal=3,err_msg='Fit results from the trimer are not approx. equal to the standard fit results.')
