import holopy as hp
import numpy as np
from holopy.scattering import calc_holo, calc_scat_matrix
from holopy.scattering.scatterer import Axisymmetric, Spheroid, Sphere
from holopy.scattering.theory import Tmatrix, DDA
from holopy.core.math import rotation_matrix

#ps = Spheroid(n = 1.585, r = [.4,1.5], rotation=[np.pi/2, 2], center = [10,10,20])
#holo = DDA.calc_holo(ps,schema)

ps = Axisymmetric(n = 1.585, r = [.4, 1.5], rotation=[np.pi/2, 2], center = [10,10,20])
sphere = Sphere(n = 1.59, r = 0.9, center=(10, 10, 20))

medium_index = 1.33
illum_wavelen = 0.660
illum_polarization = (1,0)
detector = hp.detector_grid(shape = 100, spacing = .1)
holo_sphere = calc_holo(detector, sphere, medium_index, illum_wavelen, illum_polarization, theory=Tmatrix)
holo_spheroid = calc_holo(detector, ps, medium_index, illum_wavelen, illum_polarization)
#print(calc_scat_matrix(detector, ps, medium_index, illum_wavelen))

holo = holo_sphere
hp.show(holo)

