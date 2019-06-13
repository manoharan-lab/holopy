import holopy as hp
import numpy as np
from holopy.core.io import get_example_data_path
from holopy.propagation import ps_propagate
from scipy.ndimage.measurements import center_of_mass

imagepath = get_example_data_path('ps_image01.jpg')
bgpath = get_example_data_path('ps_bg01.jpg')
L = 0.0407 # distance from light source to screen/camera
cam_spacing = 12e-6 # linear size of camera pixels
mag = 9.0 # magnification
npix_out = 1020 # linear size of output image (pixels)
zstack = np.arange(1.08e-3, 1.18e-3, 0.01e-3) # distances from camera to reconstruct

holo = hp.load_image(imagepath, spacing=cam_spacing, illum_wavelen=406e-9, medium_index=1) # load hologram
bg = hp.load_image(bgpath, spacing=cam_spacing) # load background image
holo = hp.core.process.bg_correct(holo, bg+1, bg) # subtract background (not divide)
beam_c = center_of_mass(bg.values.squeeze()) # get beam center
out_schema = hp.core.detector_grid(shape=npix_out, spacing=cam_spacing/mag) # set output shape

recons = ps_propagate(holo, zstack, L, beam_c, out_schema) # do propagation
hp.show(abs(recons[:,350:550,450:650])) # display result
