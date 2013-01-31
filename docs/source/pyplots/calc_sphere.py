import holopy as hp
from holopy.scattering.scatterer import Sphere
from holopy.core import ImageSchema, Optics
from holopy.scattering.theory import Mie

sphere = Sphere(n = 1.59+.0001j, r = .5, center = (4, 3, 5))

schema = ImageSchema(shape = 100, spacing = .1,
                     optics = Optics(wavelen = .660, index = 1.33,
                                     polarization = [1,0]))

holo = Mie.calc_holo(sphere, schema)
hp.show(holo)
