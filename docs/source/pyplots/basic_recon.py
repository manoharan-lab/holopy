import numpy as np
import holopy as hp
from holopy import propagate
from holopy.core.io import get_example_data_path

imagepath = get_example_data_path('image01.jpg')
raw_holo = hp.load_image(imagepath, spacing = 0.0851, wavelen = 0.66, index = 1.33)
bgpath = get_example_data_path('bg01.jpg')
bg = hp.load_image(bgpath)
holo = raw_holo / bg

zstack = np.linspace(1, 15, 8)
rec_vol = propagate(holo, zstack)
hp.show(rec_vol)
