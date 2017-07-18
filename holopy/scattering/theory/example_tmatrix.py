import holopy as hp
import numpy as np
from holopy.scattering import calc_holo, calc_scat_matrix, Sphere, Spheroid, Tmatrix, DDA

ps = Spheroid(n = 1.585, r = [.4,1.5], rotation=[0, 90, 120], center = [10,10,20])

#ps = Axisymmetric(n = 1.585, r = [.4, 1.5], rotation=[np.pi/2, 2], center = [10,10,20])
sphere = Sphere(n = 1.59, r = 0.9, center=(10, 10, 20))

medium_index = 1.33
illum_wavelen = 0.660
illum_polarization = (1,0)
detector = hp.detector_grid(shape = 200, spacing = .1)
holo_dda = calc_holo(detector, ps, medium_index, illum_wavelen, illum_polarization, theory=DDA)
holo_sphere = calc_holo(detector, sphere, medium_index, illum_wavelen, illum_polarization, theory=Tmatrix)
holo_spheroid = calc_holo(detector, ps, medium_index, illum_wavelen, illum_polarization)
print(calc_scat_matrix(detector, ps, medium_index, illum_wavelen))

hp.show(holo_sphere)
hp.show(holo_dda)
hp.show(holo_spheroid)
