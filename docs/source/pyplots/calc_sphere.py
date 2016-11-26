import holopy as hp
from holopy.core import detector_grid
from holopy.scattering import calc_holo, Sphere

detector = detector_grid(shape = 100, spacing = .1, medium_index = 1.33, illum_wavelen = 0.66, illum_polarization = (1,0))

sphere = Sphere(n = 1.59, r = .5, center = (4, 4, 5))

holo = calc_holo(detector, sphere)
hp.show(holo)
