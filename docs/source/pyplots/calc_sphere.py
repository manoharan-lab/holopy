import holopy as hp
from holopy.core import ImageSchema
from holopy.scattering import calc_holo, Sphere

wavelen = 0.66
polarization = (1, 0)
index = 1.33
schema = ImageSchema(shape = 100, spacing = .1, index = index, wavelen = wavelen, polarization = polarization)

sphere = Sphere(n = 1.59, r = .5, center = (4, 4, 5))
holo = calc_holo(schema, sphere)
hp.show(holo)
