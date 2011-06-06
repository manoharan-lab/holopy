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
The tests here test basic reconstruction capability

.. moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>
'''

import sys
import os
hp_dir = (os.path.split(sys.path[0])[0]).rsplit(os.sep, 1)[0]
sys.path.append(hp_dir)
import numpy as np
import scipy
import holopy
import nose
from numpy.testing import assert_array_equal
import string

class TestRecon:
    def test_reconNear(self):
        path = os.path.abspath(holopy.__file__)
        path = string.rstrip(path, chars='__init__.pyc')+'tests/exampledata/'
        image_path = path + 'image0003.npy'
        gold_path = path + 'recon_4.npy'
        gold = np.load(gold_path)
        opts = path + 'optical_train3.yaml'
        im = holopy.load(image_path,optics=opts)
        rec_im = holopy.reconstruct(im, 4e-6)
        rec_im = abs(rec_im[:,:,0,0] * scipy.conj(rec_im[:,:,0,0]))
        assert_array_equal(rec_im.astype('uint8'),gold)

    def test_reconMiddle(self):
        path = os.path.abspath(holopy.__file__)
        path = string.rstrip(path, chars='__init__.pyc')+'tests/exampledata/'
        image_path = path + 'image0003.npy'
        gold_path = path + 'recon_7.npy'
        gold = np.load(gold_path)
        opts = path + 'optical_train3.yaml'
        im = holopy.load(image_path,optics=opts)
        rec_im = holopy.reconstruct(im, 7e-6)
        rec_im = abs(rec_im[:,:,0,0] * scipy.conj(rec_im[:,:,0,0]))
        assert_array_equal(rec_im.astype('uint8'),gold)
    
    def test_reconFar(self):
        path = os.path.abspath(holopy.__file__)
        path = string.rstrip(path, chars='__init__.pyc')+'tests/exampledata/'
        image_path = path + 'image0003.npy'
        gold_path = path + 'recon_10.npy'
        gold = np.load(gold_path)
        opts = path + 'optical_train3.yaml'
        im = holopy.load(image_path,optics=opts)
        rec_im = holopy.reconstruct(im, 10e-6)
        rec_im = abs(rec_im[:,:,0,0] * scipy.conj(rec_im[:,:,0,0]))
        assert_array_equal(rec_im.astype('uint8'),gold)
