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
Common entry point for holopy io.  Dispatches to the correct load/save
functions.

.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.havard.edu>
"""
import os
import glob
from warnings import warn
import numpy as np

import serialize
from holopy.core.io.image_file_io import load_image, save_image
from holopy.core.marray import Image, arr_like
from holopy.core.metadata import Optics, interpret_args
from holopy.core.helpers import _ensure_array


def load(inf, spacing = None, wavelen=None, index=None, polarization=None, optics = None, channel=None):
    """
    Load data or results

    Parameters
    ----------
    inf : single or list of basestring or files
        File to load.  If the file is a yaml file, all other arguments are
        ignored.  If inf is a list of image files or filenames they are all
        loaded as a a timeseries hologram
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
    obj : The object loaded, :class:`holopy.core.marray.Image`, or as loaded from yaml

    """
    if isinstance(optics, (basestring, file)):
        optics = serialize.load(optics)

    # attempt to load a holopy yaml file
    try:
        loaded = serialize.load(inf)
        #if loading a hologram from a yaml file and trying to overwrite variables, you
        #should have to do so explicitly after the fact. Extra arguments when reloading are ignored.
        return loaded
    except (serialize.ReaderError, AttributeError):
        pass
        # If that fails, we go on and read images

    loaded_files = []
    for inf in _ensure_array(inf):
        loaded = load_image(inf, spacing=spacing, optics=optics, channel=channel)

        loaded_files.append(loaded)

    image = Image(np.dstack(loaded_files).squeeze(), spacing=spacing)
    image = interpret_args(image, index, wavelen, polarization, optics)
    return image

def save(outf, obj):
    """
    Save a holopy object

    Will save objects as yaml text containing all information about the object
    unless outf is a filename with an image extension, in which case it will
    save an image, truncating metadata.

    Parameters
    ----------
    outf : basestring or file
        Location to save the object
    obj : :class:`holopy.core.holopy_object.HoloPyObject`
        The object to save

    Notes
    -----
    Marray objects are actually saved as a custom yaml file consisting of a yaml
    header and a numpy .npy binary array.  This is done because yaml's saving of
    binary array is very slow for large arrays.  HoloPy can read these 'yaml'
    files, but any other yaml implementation will get confused.
    """
    if isinstance(outf, basestring):
        filename, ext = os.path.splitext(outf)
        if ext in ['.tif', '.TIF', '.tiff', '.TIFF']:
            save_image(outf, obj)
            return
    serialize.save(outf, obj)

def get_example_data_path(name):
    path = os.path.abspath(__file__)
    path = os.path.join(os.path.split(os.path.split(path)[0])[0],
                        'tests', 'exampledata')
    return os.path.join(path, name)

def get_example_data(name):
    return load(get_example_data_path(name))

def average_images(images, spacing=None, optics=None, image_glob='*.tif'):
    """
    Average a set of images (usually as a background)

    Parameters
    ----------
    images : string or list(string)
        Directory or list of filenames or filepaths. If images is a directory,
        it will average all images matching image_glob.
    spacing : float
        Spacing between pixels in the images
    optics : :class:`.Optics` object
        Optics for the images
    image_glob : string
        Glob used to select images (if images is a directory)

    Returns
    -------
    averaged_image : :class:`.Image` object
        Image which is an average of images
    """

    try:
        if os.path.isdir(images):
            images = glob.glob(os.path.join(images, image_glob))
    except TypeError:
        pass

    if len(images) < 1:
        raise Error("No images found")

    accumulator = load(images[0], spacing, optics)
    for image in images[1:]:
        accumulator += load(image, spacing, optics)

    return accumulator/len(images)
