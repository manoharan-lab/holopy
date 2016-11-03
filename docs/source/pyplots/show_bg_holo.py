import holopy as hp
from holopy.core.io import get_example_data_path
imagepath = get_example_data_path('image01.jpg')
raw_holo = hp.load_image(imagepath, spacing = 0.0851)
bgpath = get_example_data_path('bg01.jpg')
bg = hp.load_image(bgpath, spacing = 0.0851)
holo = hp.core.bg_correct(raw_holo, bg)
hp.show(holo)
