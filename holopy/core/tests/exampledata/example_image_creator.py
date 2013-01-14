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
"""
A script for generating and saving holograms to be used in unit tests.
This also contains code for generating some reconstructions to test
against.

.. moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>
.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
"""
# TODO: needs to be updated to use 2.0 methods for calculating holograms

import holopy
import holopy.model.mie
import scipy.ndimage
import numpy.random
import numpy as np

#single particle
#image0001.npy is real data

#dimer
#image0002.npy is real data

#trimer with all three particles in contact with each other
im_size = 100
optics_info = holopy.optics.Optics(wavelen = 658e-9, index = 1.33, pixel_scale = 0.1e-6)
noise_factor = 0.04 # noise factor: fraction of smoothed gaussian noise that's added
gauss_width = 1.
noise = scipy.ndimage.filters.gaussian_filter(numpy.random.randn(im_size, im_size), gauss_width)

holo = holopy.model.tmatrix_trimer.forward_holo(im_size,optics_info,1.59,1.59,1.59,.00001,.00001,.00001,.5e-6,.5e-6,.5e-6,6e-6,6e-6,7e-6,.6,45,10,0.0)
holo = holo + noise_factor * noise
holo = holo*255./holo.max()
out_name = 'image0003.npy'
numpy.save(out_name, holo.astype('uint8'))

#trimer reconstructions
image_path = 'image0003.npy'
opts = 'optical_train3.yaml'
im = holopy.load(image_path,optics=opts)
rec_im = holopy.reconstruct(im, 4e-6)
rec_im = abs(rec_im[:,:,0,0] * scipy.conj(rec_im[:,:,0,0]))
rec_im = np.around((rec_im-rec_im.min())/(rec_im-rec_im.min()).max()*255)
out_name = 'recon_4.npy'
numpy.save(out_name, rec_im.astype('uint8'))

rec_im = holopy.reconstruct(im, 7e-6)
rec_im = abs(rec_im[:,:,0,0] * scipy.conj(rec_im[:,:,0,0]))
rec_im = np.around((rec_im-rec_im.min())/(rec_im-rec_im.min()).max()*255)
out_name = 'recon_7.npy'
numpy.save(out_name, rec_im.astype('uint8'))

rec_im = holopy.reconstruct(im, 10e-6)
rec_im = abs(rec_im[:,:,0,0] * scipy.conj(rec_im[:,:,0,0]))
rec_im = np.around((rec_im-rec_im.min())/(rec_im-rec_im.min()).max()*255)
out_name = 'recon_10.npy'
numpy.save(out_name, rec_im.astype('uint8'))
