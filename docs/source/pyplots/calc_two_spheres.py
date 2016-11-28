import holopy as hp
import numpy as np
from holopy.scattering import calc_holo,Sphere, Spheres

illum_wavelen = 0.66
illum_polarization = (1, 0)
medium_index = 1.33
detector = hp.detector_grid(shape = 100, spacing = .1)

s1 = Sphere(center=(5, 5, 5), n = 1.59, r = .5)
s2 = Sphere(center=(4, 4, 5), n = 1.59, r = .5)
collection = Spheres([s1, s2])
holo = calc_holo(schema, collection, medium_index illum_wavelen, illum_polarization)
hp.show(holo)
