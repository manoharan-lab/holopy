import holopy as hp
from holopy.scattering.scatterer import Axisymmetric, Spheroid
from holopy.core import ImageSchema, Optics
from holopy.scattering.theory import Tmatrix, DDA
from holopy.core.math import rotation_matrix

schema = ImageSchema(shape = 200, spacing = .1,
                     optics = Optics(wavelen = .660, index = 1.33,
                                     polarization = [1,0]))

ps = Spheroid(n = 1.585, r = [.4,1.5], rotation=[pi/2, 2], center = [10,10,20])
holo = DDA.calc_holo(ps,schema)
hp.show(holo)
holo = Tmatrix.calc_holo(ps,schema)
hp.show(holo)

