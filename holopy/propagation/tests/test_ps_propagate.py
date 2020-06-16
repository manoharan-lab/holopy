# Copyright 2011-2017, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, Anna Wang, Solomon Barkley, Aaron Goldfain
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
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from nose.plugins.attrib import attr

from holopy.core.io import get_example_data_path, load_image
from holopy.propagation import ps_propagate
from holopy.core.process import bg_correct, subimage
from holopy.core.tests.common import verify
from warnings import simplefilter


@attr("medium")
def test_ps():

    imagepath = get_example_data_path('ps_image01.jpg')
    bgpath = get_example_data_path('ps_bg01.jpg')
    L = 0.0407 # distance from light source to screen
    cam_spacing = 12e-6 # linear size of camera pixels
    mag = 9.0 # magnification
    npix_out = 1020 # linear size of output image (pixels)
    zstack = [1.08e-3, 1.18e-3] # distances from camera to reconstruct

    holo = load_image(imagepath, spacing=cam_spacing, illum_wavelen=406e-9, medium_index=1) # load hologram
    bg = load_image(bgpath, spacing=cam_spacing) # load background image
    holo = bg_correct(holo, bg+1, bg) # subtract background
    holo = subimage(holo,[250,500],300)
    beam_c = center_of_mass(bg.values.squeeze()) # get beam center
    simplefilter('ignore')
    recons = ps_propagate(holo, zstack, L, beam_c) # do propagation

    verify(recons, 'ps_recon')

