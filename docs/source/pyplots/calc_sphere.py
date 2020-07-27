import holopy as hp
from holopy.scattering import calc_holo, Sphere


sphere = Sphere(n=1.59, r=0.5, center=(4, 4, 5))

medium_index = 1.33
illum_wavelen = 0.660
illum_polarization = (1, 0)
detector = hp.detector_grid(shape=100, spacing=0.1)

holo = calc_holo(detector, sphere, medium_index, illum_wavelen,
                 illum_polarization, theory='auto')
hp.show(holo)
