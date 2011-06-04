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
image file formats used for holograms. 

.. moduleauthor:: Jerome Fung <jfung@physics.harvard.edu>
.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.harvard.edu>
"""

import numpy as np
import scipy as sp
import Image
import os
import glob
import os.path
from scipy.misc.pilutil import fromimage
from holopy.third_party.tifffile import TIFFfile
from holopy.optics import Optics
from holopy.hologram import Hologram
from holopy.utility.errors import NotImplemented, LoadError, NoFilesFound
from holopy import process
from holopy.io.yaml_io import load_yaml

def load(im, optics=None, bg=None, bg_type='subtract',
        channel=0, time_scale=None): 
    """
    Loads image files and metadata to make a Hologram object.

    Parameters
    ----------
    im : ndarray, string, or list of strings
        if ndarray, contains the raw hologram data;
        if string or list of strings, specifies the filename or
        list of filenames 
    optics : :class:`holopy.optics.Optics` object or string (optional)
        Optical train parameters.  If string, specifies the filename
        of an optics yaml
    bg : string (optional)
        name of background file
    bg_type : string (optional)
        set to 'subtract' or 'divide' to specify how background is removed
    channel : int (optional)
        number of channel to load for a color image (in general 0=red,
        1=green, 2=blue) 
    time_scale : float or list (optional)
        time between frames or, if list, time at each frame

    Returns
    -------
    holo : :class:`holopy.hologram.Hologram` object
    
    Notes
    -----
    Currently only handles one channel of a color image.
    """
    
    # Handle the Optics File
    if isinstance(optics, basestring):
        try:
            # read from a yaml file.  Each line should be one of the
            # parameters of the optics.
            #
            # For Example
            #
            # wavelen: 785e-9
            # pixel_scale: [3.4e-7, 3.4e-7]
            #
            # would be a minimal file that would work
            optics = Optics(**load_yaml(optics))
        except LoadError as er:
            print("Could not load optics file: %s" % er.filename)
        if not isinstance(optics, Optics):
            print("Optics not provided, loading hologram without physical reference")
            optics = Optics(wavelen=1, pixel_scale=(1,1))
            
    def _guess_extension(filename):
        # most images will be tif files, this lets the user not
        # specify the extension 
        if os.path.splitext(filename)[1] == '' and os.path.exists(filename+'.tif'):
            filename = filename+'.tif'
        return filename

    if isinstance(im, np.ndarray):
        # we were given loaded data, use it
        input_array = im
    else:
        # load files given
        filenames = []
        try:
            im.__iter__() # I assume im could be a list/array?
        except AttributeError: # strings don't have __iter__
            im = [im]

        for f in im: 
            g = glob.glob(f)
            if g:
                filenames.extend(g)
            elif os.path.exists(f+'.tif'):
                filenames.append(f+'.tif')

        if len(filenames) < 1:
            raise NoFilesFound(im, os.getcwd())

        input_array = _read(filenames[0], channel=channel)
        for f in filenames[1:]:
            # For a list of files; open them as a 3D array 
            new_array = (_read(f, channel=channel))
            # stack along 3rd dimension
            input_array = np.dstack((input_array, new_array))

    holo = Hologram(input_array, optics=optics,
                    time_scale=time_scale, name=_make_id(im))
    
    if isinstance(bg, basestring):
        bg = load(_guess_extension(bg), optics, channel=channel)

    if isinstance(bg, np.ndarray):
        bg = Hologram(bg)

    if isinstance(bg, Hologram):
        return process.background(holo, bg, bg_type)
    else:
        return holo

def save_image(im, filename=None, phase=False):
    """
    Saves an ndarray or hologram as a tiff.

    If the array is complex, it will save the magnitude by default.
    
    Parameters
    ----------
    im : ndarray or Hologram
        image to save. 
    filename : basestring (optional)
        filename in which to save image. If im is a hologram the
        function should default to the hologram's name field if no
        filename is specified 
    phase : boolean (optional)
        if True, save the phase data rather than the magnitude

    """
    if filename is None:
        filename = im.name

    if filename is None:
        print("No filename specified, aborting")
        return

    # if we don't have an extension, default to tif
    if os.path.splitext(filename)[1] is '':
        filename+='.tif'
    if np.iscomplex(im).any():
        if phase:
            im = np.angle(im)
        else:
            im = np.abs(im)
        
    sp.misc.pilutil.toimage(im).save(filename)
    

def _read(filename, channel=0):
    """
    Handler for opening various types of hologram image files.

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

    # check file extension
    extension = os.path.splitext(filename)[1]
    if extension in ['.npy', '.npz']:
        # numpy format
        return np.load(filename)
    elif extension in ['.tif', '.TIF', '.tiff', '.TIFF']:
        # check for nonstandard TIFF file (e.g. 12-bit, which PIL
        # can't open)  
        arr = _read_tiff(filename)
    else:
        # try PIL
        im = Image.open(filename)
        arr = fromimage(im).astype('d')

    # pick out only one channel of a color image
    if len(arr.shape) > 2:
        if channel >= arr.shape[2]:
            raise LoadError(filename, "The image doesn't have a channel number " + channel)
        else:
            arr = arr[:,:,channel]
    elif channel > 0:
        print "Warning: not a color image (channel number ignored)"

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
        im = Image.open(filename)
        arr = fromimage(im).astype('d')
    elif depth == 12:
        tif.close()
        if width == height:
            arr = _read_tiff_12bit(filename,height)
        else:
            raise NotImplemented("Read non-square 12 bit tiff")
    else:
        # use the tifffile representation
        arr = tif.asarray().astype('d')
        tif.close()

    return arr

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
        im = Image.fromstring("F", (size, size), data[offsetdict[size]:], 
                              "bit", 12, 0, 0, 0, 1)
    else:
        offset = 378 # it's got to be smaller for anything < 1024x1024
        done = False

        while not done:
            try:
                im = Image.fromstring("F", (size, size), 
                                      data[offset:], "bit", 12, 0, 0, 0, 1)
            except ValueError:
                offset = offset - 1
            else:
                done = True

    imarray = fromimage(im).astype('d')
    return imarray


def _make_id(name):
    """ Construct an identifying name for a hologram based on the file
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
