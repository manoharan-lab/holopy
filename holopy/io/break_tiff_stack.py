'''
Load a multilayer TIFF, extract a given frame, return ndarray

Jerome Fung <fung@physics.harvard.edu>
'''

import numpy as np
from holopy.third_party.tifffile import TIFFfile

def extract_frame(fname, num):
    tif = TIFFfile(fname)
    return tif[num].asarray()
