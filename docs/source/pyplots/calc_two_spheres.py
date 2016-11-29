import holopy as hp
from holopy.scattering import calc_holo, Sphere

sphere = Sphere(n = 1.59, r = .5, center = (4, 4, 5))
detector = hp.detector_grid(shape = 100, spacing = .1)
medium_index = 1.33
illum_wavelen = 0.660
illum_polarization = (1,0)

from holopy.core.io import get_example_data_path
imagepath = get_example_data_path('image01.jpg')
exp_img = hp.load_image(imagepath, spacing=0.0851, medium_index=medium_index, illum_wavelen=illum_wavelen, illum_polarization=illum_polarization)
exp_img = exp_img[{'x':slice(0,100),'y':slice(0,100)}]

from holopy.scattering import Spheres
s1 = Sphere(center=(5, 5, 5), n = 1.59, r = .5)
s2 = Sphere(center=(4, 4, 5), n = 1.59, r = .5)
collection = Spheres([s1, s2])
holo = calc_holo(exp_img, collection)
hp.show(holo)
