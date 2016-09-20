# Copyright 2011-2013, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang
#
# This file is part of HoloPy.
#
# HoloPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HoloPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HoloPy.  If not, see <http://www.gnu.org/licenses/>.
"""
Routines that operate on files, including routines that import various
image file formats used for holograms.

.. moduleauthor:: Jerome Fung <jfung@physics.harvard.edu>
.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.harvard.edu>
"""


import numpy as np
import scipy as sp
# some ways of installing PIL give Image, others give image, this can
# handle both
# Hopefully this will clean up once we switch to pillow
try:
    import PIL.Image as PILImage
except ImportError: # pragma: no cover
    try:
        import Image as PILImage
    except ImportError:
        import image as PILImage
import os
import warnings
from copy import copy
import json
from scipy.misc import fromimage, bytescale
from holopy.core import Image

def save_image(filename, im, scaling='auto', depth=8):
    """Save an ndarray or image as a tiff.

    Parameters
    ----------
    im : ndarray or :class:`holopy.image.Image`
        image to save.
    filename : basestring
        filename in which to save image. If im is an image the
        function should default to the image's name field if no
        filename is specified
    scaling : 'auto', None, or (None|Int, None|Int)
        How the image should be scaled for saving. Ignored for float
        output. It defaults to auto, use the full range of the output
        format. Other options are None, meaning no scaling, or a pair
        of integers specifying the values which should be set to the
        maximum and minimum values of the image format.
    depth : 8, 16 or 'float'
        What type of image to save. Options other than 8bit may not be supported
        for many image types. You probably don't want to save 8bit images without
        some kind of scaling.

    """
    # if we don't have an extension, default to tif
    if os.path.splitext(filename)[1] is '':
        filename += '.tif'

    # to replicate old behavior from using sp.misc.toimage
    if depth == 8:
        if scaling == 'auto':
            cmin, cmax = None, None
        else:
            cmin, cmax = scaling
        im = bytescale(im)
    elif depth != 'float':
        if scaling is not None:
            if scaling == 'auto':
                min = im.min()
                max = im.max()
            elif len(scaling) == 2:
                min, max = scaling
            else:
                raise Error("Invalid image scaling")
            if min is not None:
                im = im - min
            if max is not None:
                im = im * ((2**depth-1)/max)
        if depth == 8:
            im = (im+.4999).astype('uint8')
        elif depth == 16:
            # PIL can't handle uint16, but seems to do the right thing
            # with int16, so go ahead and use it
            im = (im+.4999).astype('int16')
        else:
            raise Error("Unknown image depth")

    PILImage.fromarray(im).save(filename, autoscale=False)
