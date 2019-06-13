import matplotlib.pyplot as plt
import numpy as np
import holopy as hp
from holopy.scattering import calc_scat_matrix, Sphere

medium_index = 1.33
illum_wavelen = 0.660
illum_polarization = (1,0)

detector = hp.detector_points(theta = np.linspace(0, np.pi, 100), phi = 0)
distant_sphere = Sphere(r=0.5, n=1.59)
matr = calc_scat_matrix(detector, distant_sphere, medium_index, illum_wavelen)

import matplotlib.pyplot as plt
plt.figure()
plt.semilogy(np.linspace(0, np.pi, 100), abs(matr[:,0,0])**2)
plt.semilogy(np.linspace(0, np.pi, 100), abs(matr[:,1,1])**2)
plt.show()
