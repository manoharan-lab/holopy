# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca
# W. Perry, Jerome Fung, and Ryan McGorty
#
# This file is part of Holopy.
#
# Holopy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Holopy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Holopy.  If not, see <http://www.gnu.org/licenses/>.
"""
Routines that operate on files, including routines that import various
image file formats used for holograms. S

.. moduleauthor:: Jerome Fung <jfung@physics.harvard.edu>
.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.harvard.edu>
"""
from __future__ import division

import numpy as np
import scipy as sp
import Image as PILImage
import os
import warnings
from scipy.misc.pilutil import fromimage
from ..third_party.tifffile import TIFFfile

def save_image(filename, im):
    """
    Saves an ndarray or image as a tiff.

    If the array is complex, it will save the magnitude by default.

    Parameters
    ----------
    im : ndarray or :class:`holopy.image.Image`
        image to save.
    filename : basestring
        filename in which to save image. If im is a image the
        function should default to the image's name field if no
        filename is specified
    """
    # if we don't have an extension, default to tif
    if os.path.splitext(filename)[1] is '':
        filename += '.tif'

    sp.misc.pilutil.toimage(im).save(filename)

def load_image(filename, channel=0):
    """
    Handler for opening various types of image image files.

    Parameters
    ----------
    filename : String
        name of file to open.
    channel : int (optional)
        number of channel in color image (in general 0=red,
        1=green, 2=blue)

    Returns
    -------
    arr : numpy.array
        array with data in double precision (type 'd')

    Raises
    ------
    LoadError
        if there is a problem loading a file
    """


    # The most reliable way to determine what type of image file we are working
    # with is to just try opening it with various loaders.  Extensions could be
    # missing or wrong, but, ie, if np.load succeeds, it was a np file
    try:
        return np.load(filename)
    except IOError:
        pass

    might_be_color = True
    try:
        arr, might_be_color = _read_tiff(filename)
    except (ValueError, TypeError):
        im = PILImage.open(filename)
        arr = fromimage(im).astype('d')


    # pick out only one channel of a color image
    if len(arr.shape) > 2 and might_be_color:
        if channel >= arr.shape[2]:
            raise LoadError(filename,
                            "The image doesn't have a channel number " + channel)
        else:
            arr = arr[:, :, channel]
    elif channel > 0:
        warnings.warn("Warning: not a color image (channel number ignored)")

    # we choose a convention that the large dimension of an image is
    # always x
    # here we rotate if the file does not obey this convention
    if arr.shape[0] < arr.shape[1]:
        arr = np.transpose(arr)


    return arr

def _read_tiff(filename):
    """
    Reads a TIFF and returns the image as a Numpy array (double
    precision).

    Uses tifffile.py (by Christoph Gohlke) to detect size and depth of
    image.

    Notes
    -----
    TOFIX: The library doesn't convert our Photon Focus 12-bit tiffs
    correctly, so we call a special decoder for all 12-bit tiffs
    (should fix this in the future so that all tiffs can be opened by
    tifffile)
    """
    tif = TIFFfile(filename)

    if len(tif.pages) > 1:
        try:
            return tif.asarray().transpose(), False
        except Exception:
            print('failed to read multipage tiff, attempting to read a single page')

    # assuming a one-page tiff here...
    depth = tif[0].tags.bits_per_sample.value
    width = tif[0].tags.image_width.value
    height = tif[0].tags.image_length.value
    # I think the "samples per pixel" corresponds to the number of
    # channels; check on a 24-bit tiff to make sure
    channels = tif[0].tags.samples_per_pixel.value

    if depth == 8:
        tif.close()
        # use PIL to open it
        # TOFIX: see if tifffile will open 8-bit tiffs from our
        # cameras correctly
        im = PILImage.open(filename)
        arr = fromimage(im).astype('d')
    elif depth == 12:
        tif.close()
        if width == height:
            arr = _read_tiff_12bit(filename, height)
        else:
            raise NotImplementedError("Read non-square 12 bit tiff")
    else:
        # use the tifffile representation
        arr = tif.asarray().astype('d')
        tif.close()

    return arr, True

def _read_tiff_12bit(filename, size):
    """
    Reads a 12 bit greyscale TIFF file and returns image as a Numpy array.

    Parameters
    ----------
    filename : String
        valid path to the file.
    size : Integer
        number of pixels along one dimension (assumed square)

        """
    f = open(filename, 'rb')
    data = f.read()

    # dictionary of known offsets that work
    offsetdict = {256:194, 512:234, 1024:378}

    if size in offsetdict:
        im = PILImage.fromstring("F", (size, size), data[offsetdict[size]:],
                              "bit", 12, 0, 0, 0, 1)
    else:
        offset = 378 # it's got to be smaller for anything < 1024x1024
        done = False

        while not done:
            try:
                im = PILImage.fromstring("F", (size, size),
                                      data[offset:], "bit", 12, 0, 0, 0, 1)
            except ValueError:
                offset = offset - 1
            else:
                done = True

    imarray = fromimage(im).astype('d')
    return imarray


def _make_id(name):
    """ Construct an identifying name for a image based on the file
    or files it came from
    """

    if isinstance(name, np.ndarray):
        # we can't really define a name from just an ndarray
        return None

    # Many of us have images of the form image1234, the word image
    # adds no useful information and makes our already long names longer
    def strip(name):
        name = os.path.splitext(name)[0]
        if name[:5] == 'image':
            name = name[5:]
        return name

    if len(name) == 1:
        name = name[0]

    if np.isscalar(name):
        return strip(name)
    else:
        return '{0}_to_{1}'.format(strip(name[0]), strip(name[-1]))
