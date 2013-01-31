import holopy as hp
from holopy.scattering.scatterer import Sphere
from holopy.scattering.theory import Mie
from holopy.core import ImageSchema, Optics
schema = ImageSchema(shape = 100, spacing = .1,
                     optics = Optics(wavelen = .660, index = 1.33,
                                     polarization = [1,0]))
cs = Sphere(center=(2.5, 5, 5), n = (1.59, 1.42),\
            r = (0.3, 0.6))
holo = Mie.calc_holo(cs, schema)
hp.show(holo)
