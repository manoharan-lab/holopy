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
.. moduleauthor:: Ron Alexander <ralexander@g.harvard.edu>
"""
import os
import glob
import yaml
import warnings
from PIL import Image as pilimage
import xarray as xr
import numpy as np
import importlib

from holopy.core.io import serialize
from holopy.core.io.vis import display_image
from holopy.core.metadata import (data_grid, get_spacing, update_metadata,
                    copy_metadata, to_vector, illumination)
from holopy.core.utils import ensure_array, dict_without
from holopy.core.errors import NoMetadata, BadImage, LoadError
from holopy.core.holopy_object import FullLoader# compatibility with pyyaml < 5

attr_coords = '_attr_coords'
tiflist = ['.tif', '.TIF', '.tiff', '.TIFF']


def default_extension(inf, defext='.h5'):
    try:
        file, ext = os.path.splitext(inf)
    except:
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
    new_attrs = {attr_coords:{}}
    if a.name is not None:
        new_attrs['name'] = a.name
    if do_spacing:
        new_attrs['spacing']=list(get_spacing(a))

    for attr, val in a.attrs.items():
        if isinstance(val, xr.DataArray):
            new_attrs[attr_coords][attr] = {}
            for dim in val.dims:
                new_attrs[attr_coords][attr][str(dim)]=val[dim].values
            new_attrs[attr]=list(ensure_array(val.values))
        else:
            new_attrs[attr_coords][attr]=False
            if val is not None:
                new_attrs[attr] = yaml.dump(val)
    new_attrs[attr_coords] = yaml.dump(new_attrs[attr_coords],
                                       default_flow_style=True)
    return new_attrs


def unpack_attrs(a):
    if len(a) == 0:
        return a
    new_attrs={}
    attr_ref = yaml.load(a[attr_coords], Loader=FullLoader)
    attrs_to_ignore = ['spacing', 'name', '_dummy_channel', '_image_scaling']
    for attr in dict_without(attr_ref, attrs_to_ignore):
        if attr_ref[attr]:
            new_attrs[attr] = xr.DataArray(
                a[attr],
                coords=attr_ref[attr],
                dims=list(attr_ref[attr].keys()))
        elif attr in a:
            new_attrs[attr] = yaml.safe_load(a[attr])
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

            # loaded dataset potential contains multiple DataArrays. We
            # need to find out their names and loop through them to unpack
            # metadata
            data_vars = list(ds.data_vars.keys())
            for var in data_vars:
                ds[var].attrs = unpack_attrs(ds[var].attrs)

            # return either a single DataArray or a DataSet containing
            # multiple DataArrays.
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
            with open(inf, 'rb') as imagefile:
                meta = yaml.safe_load(pilimage.open(imagefile).tag[270][0])
            try:
                spacing = meta['spacing']
                assert spacing is not None
            except:
                raise NoMetadata
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    im = load_image(inf, spacing, name = meta['name'],
                                    channel='all')
                if '_dummy_channel' in meta:
                    dummy_channel = yaml.safe_load(meta['_dummy_channel'])
                    dummy_channel = im.illumination[dummy_channel]
                    im = im.drop(dummy_channel.item(), illumination)
                if '_image_scaling' in meta:
                    smin, smax = yaml.safe_load(meta['_image_scaling'])
                    im = (im-im.min())*(smax-smin)/(im.max()-im.min())+smin
                im.attrs = unpack_attrs(meta)
                return im
        except KeyError or TypeError:
            raise NoMetadata
    else:
        raise NoMetadata


def load_image(inf, spacing=None, medium_index=None, illum_wavelen=None,
               illum_polarization=None, noise_sd=None,
               channel=None, name=None):
    """
    Load data or results

    Parameters
    ----------
    inf : string
        File to load.
    spacing : float or (float, float) (optional)
        pixel size of images in each dimension - assumes square pixels if single value.
        set equal to 1 if not passed in and issues warning.
    medium_index : float (optional)
        refractive index of the medium
    illum_wavelen : float (optional)
        wavelength (in vacuum) of illuminating light
    illum_polarization : (float, float) (optional)
        (x, y) polarization vector of the illuminating light
    noise_sd : float (optional)
        noise level in the image, normalized to image intensity
    channel : int or tuple of ints (optional)
        number(s) of channel to load for a color image (in general 0=red,
        1=green, 2=blue)
	name : str (optional)
        name to assign the xr.DataArray object resulting from load_image

    Returns
    -------
    obj : xarray.DataArray representation of the image with associated metadata

    """
    if name is None:
        name = os.path.splitext(os.path.split(inf)[-1])[0]

    with open(inf,'rb') as pi_raw:
        pi = pilimage.open(pi_raw)
        arr = np.asarray(pi).astype('d')
        try:
            if isinstance(yaml.safe_load(pi.tag[270][0]), dict):
                warnings.warn(
                    "Metadata detected but ignored. Use hp.load to read it.")
        except (AttributeError, KeyError):
            pass

    extra_dims = None
    if channel is None:
        if arr.ndim > 2:
            raise BadImage('Not a greyscale image. You must specify which channel(s) to use')
    elif arr.ndim == 2:
            if not channel == 'all':
                warnings.warn("Not a color image (channel number ignored)")
            pass
    else:
        # color image with specified channel(s)
        if channel == 'all':
            channel = range(arr.shape[2])
        channel = ensure_array(channel)
        if channel.max() >= arr.shape[2]:
            raise LoadError(filename,
                "The image doesn't have a channel number {0}".format(channel.max()))
        else:
            arr = arr[:, :, channel].squeeze()

            if len(channel) > 1:
                # multiple channels. increase output dimensionality
                if channel.max() <=2:
                    channel = [['red','green','blue'][c] for c in channel]
                extra_dims = {illumination: channel}
                if illum_wavelen is not None and not isinstance(illum_wavelen,dict) and len(ensure_array(illum_wavelen)) == len(channel):
                    illum_wavelen = xr.DataArray(ensure_array(illum_wavelen), dims=illumination, coords=extra_dims)
                if not isinstance(illum_polarization, dict) and np.array(illum_polarization).ndim == 2:
                    pol_index = xr.DataArray(channel, dims=illumination, name=illumination)
                    illum_polarization=xr.concat([to_vector(pol) for pol in illum_polarization], pol_index)

    image = data_grid(
        arr, spacing=spacing, medium_index=medium_index,
        illum_wavelen=illum_wavelen, illum_polarization=illum_polarization,
        noise_sd=noise_sd, name=name, extra_dims=extra_dims)
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
    filename : basestring
        filename in which to save image. If im is an image the
        function should default to the image's name field if no
        filename is specified
    im : ndarray or :class:`holopy.image.Image`
        image to save.
    scaling : 'auto', None, or (Int, Int)
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
    im = display_image(im, scaling)
    _save_im(filename, im, depth)


def save_images(filenames, ims, scaling='auto', depth=8):
    """
    Saves a volume as separate images (think of reconstruction volumes).

    Parameters
    ----------
    filenames : list
        List of filenames. There have to be the same number of filenames as of
        images to save. Each image will be saved in the corresponding file with
        the same index.
    ims : ndarray or :class:`holopy.image.Image`
        Images to save, with separate z-coordinates from which they each will
        be selected.
    scaling : 'auto', None, or (Int, Int)
        How the images should be scaled for saving. Ignored for float output.
        It defaults to auto, use the full range of the output format. Other
        options are None, meaning no scaling, or a pair of integers specifying
        the values which should be set to the maximum and minimum values of the
        image format.
    depth : 8, 16 or 'float'
        What type of image to save. Options other than 8bit may not be
        supported for many image types. You probably don't want to save 8bit
        images without some kind of scaling.
    """
    if len(ims) != len(filenames):
        raise ValueError("Not enough filenames or images provided.")

    for image_raw, filename in zip(ims, filenames):
        image_displayed = display_image(image_raw, scaling)
        _save_im(filename, image_displayed, depth)


def _save_im(filename, im, depth=8):
    """
    Internal single-image-save-method to be used in save_images. Maybe it can
    be merged with save_image.

    Parameters
    ----------
    filename : basestring
        Filename in which to save image. If im is an image the function should
        default to the image's name field if no filename is specified
    im : ndarray or :class:`holopy.image.Image`
        Image to save.
    depth : 8, 16 or 'float'
        What type of image to save. Options other than 8bit may not be
        supported for many image types. You probably don't want to save 8bit
        images without some kind of scaling.
    """
    # if we don't have an extension, default to tif
    if os.path.splitext(filename)[1] == '': filename += '.tif'

    metadat = False
    if os.path.splitext(filename)[1] in tiflist:
        if im.name == None:
            im.name = os.path.splitext(os.path.split(filename)[-1])[0]
        metadat = pack_attrs(im, do_spacing=True)
        # import ifd2 - hidden here since it doesn't play nice in some cases.
        from PIL.TiffImagePlugin import ImageFileDirectory_v2 as ifd2
        tiffinfo = ifd2()
        # place metadata in the 'imagedescription' field of the tiff metadata
        tiffinfo[270] = yaml.dump(metadat, default_flow_style=True)

    im = im.values
    if im.ndim > 2: im = im[0]

    if depth != 'float':
        if depth == 8:
            depth = 8
            typestr = 'uint8'
        elif depth == 16 or depth == 32:
            depth = depth-1
            typestr = 'int' + str(depth)
        else:
            raise Error("Unknown image depth")

        if im.max() <= 1:
            im = im * ((2**depth)-1) + .499999
            im = im.astype(typestr)

    if metadat:
        pilimage.fromarray(im).save(filename, tiffinfo=tiffinfo)
    else:
        pilimage.fromarray(im).save(filename)


def load_average(
        filepath, refimg=None, spacing=None, medium_index=None,
        illum_wavelen=None, illum_polarization=None,
        noise_sd=None, channel=None, image_glob='*.tif'):
    """
    Average a set of images (usually as a background)

    Parameters
    ----------
    filepath : string or list(string)
        Directory or list of filenames or filepaths. If filename is a
        directory, it will average all images matching image_glob.
    refimg : xarray.DataArray
        reference image to provide spacing and metadata for the new image.
    spacing : float
        Spacing between pixels in the images. Used preferentially over
        refimg value if both are provided.
    medium_index : float
        Refractive index of the medium in the images. Used
        preferentially over refimg value if both are provided.
    illum_wavelen : float
        Wavelength of illumination in the images. Used preferentially
        over refimg value if both are provided.
    illum_polarization : list-like
        Polarization of illumination in the images. Used preferentially
        over refimg value if both are provided.
    image_glob : string
        Glob used to select images (if images is a directory)

    Returns
    -------
    averaged_image : xarray.DataArray
        Image which is an average of images
        noise_sd attribute contains average pixel stdev normalized by
        total image intensity
    """
    if isinstance(filepath, str):
        if os.path.isdir(filepath):
            filepath = glob.glob(os.path.join(filepath, image_glob))
        else:
            #only a single image
            filepath=[filepath]

    if len(filepath) < 1:
        raise LoadError(filepath, "No images found")

    # read spacing from refimg if none provided
    if spacing is None:
        spacing = get_spacing(refimg)

    # read colour channels from refimg
    channel_dict = {'0': 'red', '1': 'green', '2': 'blue'}
    if channel is None and refimg is not None and illumination in refimg.dims:
        channel = [
            i for i, col in enumerate(['red', 'green', 'blue'])
            if col in refimg[illumination].values]

    if np.isscalar(spacing):
        spacing = np.repeat(spacing, 2)

    # calculate the average
    accumulator = Accumulator()
    for path in filepath:
        accumulator.push(load_image(path, spacing, channel=channel))
    mean_image = accumulator.mean()

    # calculate average noise from image
    if noise_sd is None and len(filepath) > 1:
        if channel:
            noise_sd = xr.DataArray(accumulator.cv(),
                                    [[channel_dict[str(ch)] for ch in channel]],
                                    ['illumination'])
        else:
            noise_sd = ensure_array(accumulator.cv())

    # crop according to refimg dimensions
    if refimg is not None:
        def extent(i):
            name = ['x','y'][i]
            return np.around(refimg[name].values/spacing[i]).astype('int')
        mean_image = mean_image.isel(x=extent(0), y=extent(1))
        mean_image['x'] = refimg.x
        mean_image['y'] = refimg.y

    # copy metadata from refimg
    if refimg is not None:
        mean_image = copy_metadata(refimg, mean_image, do_coords=False)

    # overwrite metadata from refimg with provided values
    return update_metadata(mean_image, medium_index, illum_wavelen, illum_polarization, noise_sd)


class Accumulator:
    """Calculates average and coefficient of variance for numerical data in
    one pass using Welford's algorithim.
    """
    def __init__(self):
        self._n = 0
        self._running_mean = None
        self._running_var = None

    def push(self, x):
        self._n += 1

        if self._n == 1:
            self._running_var = x * 0.0
            self._running_mean = self._running_var + x
        else:
            self._running_var += ((x - self._running_mean) *
             ((x - (self._running_mean + (x - self._running_mean) / self._n))))
            self._running_mean += (x - self._running_mean) / self._n

    def mean(self):
        return self._running_mean if self._running_mean is not None else 0.0

    def cv(self):
        """ The coefficient of variation
        """
        if self._n == 0:
            return None
        else:
            try: # If data is a multicolor hologram, average over first 3 dims
                return np.mean(np.array(self._std() / self.mean()),
                               axis=(0, 1, 2))
            except IndexError:
                return np.mean(np.array(self._std() / self.mean()))

    def _std(self):
        if self._n == 0:
            return None
        else:
            return np.sqrt(self._running_var / (self._n))
