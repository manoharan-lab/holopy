import holopy as hp
import numpy as np
from holopy.core import ImageSchema
from holopy.scattering import calc_holo, Sphere, Spheres

wavelen = 0.66
polarization = (1, 0)
index = 1.33
schema = ImageSchema(shape = 100, spacing = .1, index = index, wavelen = wavelen, polarization = polarization)

s1 = Sphere(center=(5, 5, 5), n = 1.59, r = .5)
s2 = Sphere(center=(4, 4, 5), n = 1.59, r = .5)
collection = Spheres([s1, s2])
holo = calc_holo(schema, collection)
hp.show(holo)
