import numpy as np
import holopy as hp
from holopy.core.io import get_example_data_path, load_average

imagepath = get_example_data_path('image01.jpg')
raw_holo = hp.load_image(imagepath, spacing = 0.0851, illum_wavelen = 0.66, medium_index = 1.33)
bgpath = get_example_data_path(['bg01.jpg','bg02.jpg','bg03.jpg'])
bg = load_average(bgpath, refimg = raw_holo)
holo = hp.core.process.bg_correct(raw_holo, bg)

zstack = np.linspace(1, 15, 8)
rec_vol = hp.propagate(holo, zstack)
hp.show(rec_vol)
