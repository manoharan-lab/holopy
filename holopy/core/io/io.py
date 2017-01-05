# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, Anna Wang, Solomon Barkley
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
import yaml
from warnings import warn
from scipy.misc import fromimage
from PIL import Image as pilimage
import xarray as xr
import importlib

from . import serialize
from ..metadata import data_grid, get_spacing, update_metadata
from ..utils import is_none, ensure_array, dict_without
from ..errors import NoMetadata

attr_coords = '_attr_coords'
tiflist = ['.tif', '.TIF', '.tiff', '.TIFF']

def default_extension(inf, defext='.h5'):
    try:
        file, ext = os.path.splitext(inf)
    except AttributeError:
        # this will happen if inf is already a file, which means we don't
        # need to do anything here
        return inf
    if not ext:
        return file + defext
    else:
        return inf

def get_example_data_path(name):
    path = os.path.abspath(__file__)
    path = os.path.join(os.path.split(os.path.split(path)[0])[0],
                        'tests', 'exampledata')
    if isinstance(name, str):
        out = os.path.join(path,name)
    else:
        #name is a list of strings
        out = [os.path.join(path,img) for img in name]
    return out

def get_example_data(name):
    return load(get_example_data_path(name))

def pack_attrs(a, do_spacing=False):
    new_attrs = {'name':a.name, attr_coords:{}}

    if do_spacing:
        new_attrs['spacing']=list(get_spacing(a))
        
    for attr, val in a.attrs.items():
        if isinstance(val, xr.DataArray):
            new_attrs[attr_coords][attr]={}
            for dim in val.dims:
                new_attrs[attr_coords][attr][dim]=val[dim].values
            new_attrs[attr]=list(ensure_array(val))
        else:
            new_attrs[attr_coords][attr]=False
            if not is_none(val):
                new_attrs[attr]=val
    new_attrs[attr_coords]=yaml.dump(new_attrs[attr_coords])
    return new_attrs

def unpack_attrs(a):
    new_attrs={}
    attr_ref=yaml.load(a[attr_coords])
    for attr in dict_without(attr_ref,['spacing','name']):
        if attr_ref[attr]:
            new_attrs[attr] = xr.DataArray(a[attr], coords=attr_ref[attr])
        elif attr in a:
            new_attrs[attr] = a[attr]
        else:
            new_attrs[attr] = None
    return new_attrs

def load(inf, lazy=False):
    """
    Load data or results

    Parameters
    ----------
    inf : string
        String specifying an hdf5 file containing holopy data

    Returns
    -------
    obj : xarray.DataArray
        The array object contained in the file

    """
    try:
        with xr.open_dataset(default_extension(inf), engine='h5netcdf') as ds:
            if '_source_class' in ds.attrs:
                _source_class = ds.attrs.pop('_source_class')
                pathtok = _source_class.split('.')
                cls = getattr(importlib.import_module(".".join(pathtok[:-1])), pathtok[-1])
                ds.close()
                return cls._load(default_extension(inf))

            # Xarray defaults to lazy loading of datasets, but I my reading of
            # things is that we will probably generally prefer eager loading
            # since our data is generally fairly small but we do lots of
            # calculations.
            if not lazy:
                ds = ds.load()



            #loaded dataset potential contains multiple DataArrays. We need
            #to find out their names and loop through them to unpack metadata
            data_vars = list(ds.data_vars.keys())
            for var in data_vars:
                ds[var].attrs = unpack_attrs(ds[var].attrs)

            #return either a single DataArray or a DataSet containing multiple DataArrays.
            if len(data_vars)==1:
                return ds[data_vars[0]]
            else:
                return ds
    except (OSError, ValueError):
        pass

    # attempt to load a yaml file
    try:
        loaded = serialize.load(inf)
        return loaded
    except (serialize.ReaderError, UnicodeDecodeError):
        pass


    if os.path.splitext(inf)[1] in tiflist:
        try:
            meta = yaml.load(pilimage.open(inf).tag[270][0])
            if meta['spacing'] is None:
                raise NoMetadata
            else:
                im = load_image(inf, meta['spacing'], name = meta['name'])
                im.attrs = unpack_attrs(meta)
                return im
        except KeyError:
            raise NoMetadata
    else:
        raise NoMetadata

def load_image(inf, spacing=None, medium_index=None, illum_wavelen=None, illum_polarization=None, normals=None, channel=None, name=None):
    """
    Load data or results

    Parameters
    ----------
    inf : single or list of basestring or files
        File to load.  If the file is a yaml file, all other arguments are
        ignored.  If inf is a list of image files or filenames they are all
        loaded as a a timeseries hologram
    channel : int (optional)
        number of channel to load for a color image (in general 0=red,
        1=green, 2=blue)

    Returns
    -------
    obj : The object loaded, :class:`holopy.core.marray.Image`, or as loaded from yaml

    """
    with pilimage.open(inf) as pi:
        arr=fromimage(pi).astype('d')

    # pick out only one channel of a color image
    if channel is not None and len(arr.shape) > 2:
        if channel >= arr.shape[2]:
            raise LoadError(filename,
                "The image doesn't have a channel number {0}".format(channel))
        else:
            arr = arr[:, :, channel]
    elif channel is not None and channel > 0:
        warn("Warning: not a color image (channel number ignored)")

    if name is None:
        name = os.path.splitext(os.path.split(inf)[-1])[0]
    return data_grid(arr, spacing, medium_index, illum_wavelen, illum_polarization, normals, name)

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
    if isinstance(outf, str):
        filename, ext = os.path.splitext(outf)
        if ext in tiflist:
            save_image(outf, obj)
            return

    if hasattr(obj, '_save'):
        obj._save(outf)
    elif hasattr(obj, 'to_dataset'):
        obj=obj.copy()
        if obj.name is None:
            obj.name=os.path.splitext(os.path.split(outf)[-1])[0]
        obj.attrs = pack_attrs(obj)
        ds = obj.to_dataset()
        ds.to_netcdf(default_extension(outf), engine='h5netcdf')
    else:
        serialize.save(outf, obj)

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
    
    metadat=False
    if os.path.splitext(filename)[1] in tiflist:
        if im.name is None:
            im.name=os.path.splitext(os.path.split(filename)[-1])[0]
        metadat = pack_attrs(im, do_spacing = True)
        from PIL.TiffImagePlugin import ImageFileDirectory_v2 as ifd2 #hiding this import here since it doesn't play nice in some scenarios
        tiffinfo = ifd2()
        tiffinfo[270] = yaml.dump(metadat) #This edits the 'imagedescription' field of the tiff metadata

    im=im.copy().sel(z=0)
    if scaling is not None:
        if scaling is 'auto':
            min = im.min()
            max = im.max()
        elif len(scaling) == 2:
            min, max = scaling
        else:
            raise Error("Invalid image scaling")
        if min is not None:
            im = im - min
        if max is not None:
            im = im / (max-min)

    if depth is not 'float':
        if depth is 8:
            depth = 8
            typestr = 'uint8'
        elif depth is 16 or depth is 32:
            depth = depth-1
            typestr = 'int' + str(depth)
        else:
            raise Error("Unknown image depth")

        if im.max() <= 1:
            im = im * ((2**depth)-1) + .499999
            im = im.astype(typestr)

    if metadat:
        pilimage.fromarray(im.values).save(filename, tiffinfo=tiffinfo)
    else:
        pilimage.fromarray(im.values).save(filename)

def load_average(filepath, refimg=None, spacing=None, medium_index=None, illum_wavelen=None, illum_polarization=None, normals=None, image_glob='*.tif'):
    """
    Average a set of images (usually as a background)

    Parameters
    ----------
    filepath : string or list(string)
        Directory or list of filenames or filepaths. If filename is a directory,
        it will average all images matching image_glob.
    refimg : xarray.DataArray
        reference image to provide spacing and metadata for the new image.
    spacing : float
        Spacing between pixels in the images. Used preferentially over refimg value if both are provided.
    medium_index : float
        Refractive index of the medium in the images. Used preferentially over refimg value if both are provided.
    illum_wavelen : float
        Wavelength of illumination in the images. Used preferentially over refimg value if both are provided.
    illum_polarization : list-like
        Polarization of illumination in the images. Used preferentially over refimg value if both are provided.
    normals : list-like
        Orientation of detector. Used preferentially over refimg value if both are provided.
    image_glob : string
        Glob used to select images (if images is a directory)

    Returns
    -------
    averaged_image : xarray.DataArray
        Image which is an average of images
    """

    try:
        if os.path.isdir(filepath):
            filepath = glob.glob(os.path.join(filepath, image_glob))
    except TypeError:
        pass

    if len(filepath) < 1:
        raise LoadError(filepath, "No images found")

    if is_none(spacing):
        spacing = get_spacing(refimg)
    
    accumulator = load_image(filepath[0],spacing)
    for image in filepath[1:]:
        accumulator += load_image(image, spacing)
    accumulator /= len(filepath)

    if not is_none(refimg):
        accumulator = update_metadata(accumulator, refimg.medium_index, refimg.illum_wavelen, refimg.illum_polarization, refimg.normals)    
    return update_metadata(accumulator, medium_index, illum_wavelen, illum_polarization, normals)
