import holopy as hp
from holopy.core import ImageSchema
from holopy.scattering import calc_holo, Sphere

wavelen = 0.66
polarization = (1, 0)
index = 1.33
schema = ImageSchema(shape = 100, spacing = .1, index = index, wavelen = wavelen, polarization = polarization)

coated_sphere = Sphere(center=(2.5, 5, 5), n=(1.59, 1.42), r=(0.3, 0.6))
holo = calc_holo(schema, coated_sphere)
hp.show(holo)

