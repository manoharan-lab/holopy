import matplotlib.pyplot as plt
import numpy as np
from holopy.core import Schema, Angles, Optics
from holopy.scattering.scatterer import Sphere
from holopy.scattering.theory import Mie
schema = Schema(positions = Angles(np.linspace(0, np.pi, 100)), optics =
                Optics(wavelen=.66, index = 1.33, polarization = (1, 0)))

sphere = Sphere(r = .5, n = 1.59)

matr = Mie.calc_scat_matrix(sphere, schema)
# It is typical to look at scattering matrices on a semilog plot.
# You can make one as follows:
plt.figure()
plt.semilogy(np.linspace(0, np.pi, 100), abs(matr[:,0,0])**2)
plt.semilogy(np.linspace(0, np.pi, 100), abs(matr[:,1,1])**2)
plt.show()
