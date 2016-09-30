import matplotlib.pyplot as plt
import numpy as np
from holopy.core import Angles
from holopy.scattering import calc_scat_matrix, Sphere

wavelen = 0.66
index = 1.33

angle_list = Angles(np.linspace(0, np.pi, 100))
distant_sphere = Sphere(r=0.5, n=1.59)
matr = calc_scat_matrix(angle_list, distant_sphere, index, wavelen)


import matplotlib.pyplot as plt
plt.figure()
plt.semilogy(np.linspace(0, np.pi, 100), abs(matr[:,0,0])**2)
plt.semilogy(np.linspace(0, np.pi, 100), abs(matr[:,1,1])**2)
plt.show()
