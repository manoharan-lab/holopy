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
Prepare HoloPy objects for display on screen or write to file.

.. moduleauthor:: Solomon Barkley
.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.harvard.edu>
"""
from warnings import warn
import sys

import xarray as xr
import numpy as np

from holopy.core.metadata import data_grid, clean_concat
from holopy.core.utils import ensure_array, ensure_scalar
from holopy.core.errors import BadImage, DependencyMissing

try:
    from matplotlib import pyplot as plt
    _NO_MATPLOTLIB = False
except ImportError:
    _NO_MATPLOTLIB = True


class VisualizationNotImplemented(Exception):
    def __init__(self, o):
        self.o = o
    def __str__(self):
        return "Visualization of object of type: {0} not implemented".format(
            self.o.__class__.__name__)


def show(o, scaling='auto', vert_axis='x', horiz_axis='y',
                    depth_axis='z', colour_axis='illumination'):
    """
    Visualize a hologram or reconstruction

    Parameters
    ----------
    o : xarray.DataArray or ndarray
        Object to visualize
    scaling : (float, float), optional
        (min, max) value to display in image, default is full range of o.
    vert_axis : str, optional
        axis to display vertically, default x.
    horiz_axis : str, optional
        axis to display horizontally, default y.
    depth_axis : str, optional
        axis to scroll with arrow keys, default 'z'.
    colour_axis : str, optional
        axis to display as RGB channels in colour image, default illumination.

    Notes
    -----
    Loads plotting library the first time it is required (so that we don't have
    to import all of matplotlib or mayavi just to load holopy)
    """

    if isinstance(o, (xr.DataArray, np.ndarray, list, tuple)):
        Show2D(display_image(o, scaling, vert_axis, horiz_axis, depth_axis,
                                                                colour_axis))
    else:
        raise VisualizationNotImplemented(o)


def save_plot(filenames, data, scaling='auto', vert_axis='x', horiz_axis='y',
              depth_axis='z', colour_axis='illumination'):
    """
    Saves a hologram or reconstruction to (a) file(s).

    Parameters
    ----------
    filenames : list / str
        Name(s) of the file(s). If there is only one image contained (e.g.
        hologram), the name will be used as a file name. If o contains more
        plottable images (e.g. reconstruction volume), it should be a list of
        filenames with the same length as objects.
    data : xarray.DataArray or ndarray
        Object to save.
    scaling : (float, float), optional
        (min, max) value to display in image, default is full range of o.
    vert_axis : str, optional
        axis to display vertically, default x.
    horiz_axis : str, optional
        axis to display horizontally, default y.
    depth_axis : str, optional
        axis to scroll with arrow keys, default 'z'.
    colour_axis : str, optional
        axis to display as RGB channels in colour image, default illumination.

    Notes
    -----
    Loads plotting library the first time it is required (so that we don't have
    to import all of matplotlib or mayavi just to load holopy)
    """
    if isinstance(data, (xr.DataArray, np.ndarray, list, tuple)):
        im = display_image(data, scaling, vert_axis, horiz_axis, depth_axis,
                           colour_axis)
        s = Show2D(im)
        if len(im) > 1:
            s.save_all(filenames)
        else:
            s.save(filenames)
    else:
        raise VisualizationNotImplemented(o)


class Show2D(object):
    def __init__(self, im):
        if _NO_MATPLOTLIB:
            raise DependencyMissing('matplotlib',
                "Install it with \'conda install -c conda-forge matplotlib\'.")

        self.i = 0
        vmin, vmax = im.attrs['_image_scaling']
        if im.ndim == 3:
            #greyscale image
            self.im = im * (vmax-vmin) + vmin
        else:
            self.im = im

        vert_spacing = np.diff(self.im[self.im.dims[1]])[0]
        horiz_spacing = np.diff(self.im[self.im.dims[2]])[0]
        ratio = vert_spacing/horiz_spacing

        self.fig = plt.figure()
        plt.gray()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel(self.im.dims[2])
        self.ax.set_ylabel(self.im.dims[1])
        self._title()
        self.plot = self.ax.imshow(self.im[0], vmin=vmin, vmax=vmax,
                                interpolation="nearest", aspect=ratio)
        self.ax.format_coord = self.format_coord
        self.colorbar = self.fig.colorbar(self.plot)
        self.fig.canvas.mpl_connect('key_press_event', self)
        self.fig.canvas.mpl_connect('button_press_event', self.click)

    def format_coord(self, x, y):
        # our coordinate convention is inverted from
        # matplotlib's default, so we need to swap x and y
        x, y = y, x
        dims = [self.im.dims[i] for i in [1, 2, 0]]
        pixel = [int(x+.5), int(y+.5), self.i]
        location = [np.interp(coord, range(len(self.im[dim])), self.im[dim])
                                for coord, dim in zip([x, y, self.i], dims)]
        pixcoords = ['{} = {},'.format(dim, pix)
                                for dim, pix in zip(dims, pixel)]
        unitcoords = ['{} = {:.1e},'.format(dim, loc)
                                for dim, loc in zip(dims, location)]
        return ' '.join(["pixels:"] + pixcoords + ["units:"] + unitcoords)

    def draw(self):
        self._title()
        self.plot.set_array(self.im[self.i])
        self.fig.canvas.draw()

    def click(self, event):
        if event.ydata is not None and event.xdata is not None:
            y, x = np.array((event.ydata, event.xdata))
            print(self.format_coord(x, y))
            sys.stdout.flush()

    def __call__(self, event):
        if event.key == 'right' and self.i < len(self.im) - 1:
            self.i += 1
            self.draw()
        elif event.key == 'left' and self.i > 0:
            self.i -= 1
            self.draw()

    def _title(self):
        if len(self.im) > 1:
            dimname = self.im.dims[0]
            titlestring = '{} = {}'.format(dimname, self.im[dimname].values[self.i])
            self.ax.set_title(titlestring)

    def save(self, filename):
        """
        Saves the currently displayed Plot into a file.

        Parameters
        ----------
        filename : str
            Name for the file to save to.
        """
        self.draw()
        self.fig.savefig(filename)

    def save_all(self, filenames):
        """
        Saves the complete stack of images into separate files.

        Parameters
        ----------
        filenames : list
            Names of the files to save as a list. Has to have the same length
            as the number of images that are contained in this object.
        """
        if len(filenames) != len(self.im):
            raise ValueError("Number of images and filenames does not match!")

        for i, name in enumerate(filenames):
            self.i = i
            self.save(name)


def display_image(im, scaling='auto', vert_axis='x', horiz_axis='y',
                    depth_axis='z', colour_axis='illumination'):
    im = im.copy()
    if isinstance(im, xr.DataArray):
        if 'z' in im.dims and len(im['z']) == 1 and depth_axis != 'z':
            im = im[{'z': 0}]
        if depth_axis == 'z' and 'z' not in im.dims:
            im = im.expand_dims('z')
        if im.ndim > 3 + (colour_axis in im.dims):
            raise BadImage("Too many dims on DataArray to output properly.")
        attrs = im.attrs
    else:
        attrs = {}
        im = ensure_array(im)
        if im.ndim > 3:
            raise BadImage("Too many dims on ndarray to output properly.")
        elif im.ndim == 2:
            im = np.array([im])
        elif im.ndim < 2:
            raise BadImage("Too few dims on ndarray to output properly.")
        axes = [0, 1, 2]
        for axis in [vert_axis, horiz_axis, depth_axis]:
            if isinstance(axis, int):
                try:
                    axes.remove(axis)
                except KeyError:
                    raise ValueError("Cannot interpret axis specifications.")
        if len(axes) > 0:
            if not isinstance(depth_axis, int):
                depth_axis = axes[np.argmin([im.shape[i] for i in axes])]
                axes.remove(depth_axis)
            if not isinstance(vert_axis, int):
                vert_axis = axes[0]
                axes.pop(0)
            if not isinstance(horiz_axis, int):
                horiz_axis = axes[0]
        im = im.transpose([depth_axis, vert_axis, horiz_axis])
        depth_axis='z'; vert_axis='x'; horiz_axis='y'
        im = data_grid(im, spacing=1, z=range(len(im)))
    if np.iscomplex(im).any():
        warn("Image contains complex values. Taking image magnitude.")
        im = np.abs(im)
    if scaling == 'auto':
        scaling = (ensure_scalar(im.min()), ensure_scalar(im.max()))
    if scaling is not None:
        im = np.maximum(im, scaling[0])
        im = np.minimum(im, scaling[1])
        im = (im-scaling[0])/(scaling[1]-scaling[0])
    im.attrs = attrs
    im.attrs['_image_scaling'] = scaling

    if colour_axis in im.dims:
        cols = [col[0].capitalize() if isinstance(col, str) else ' '
                                        for col in im[colour_axis].values]
        RGB_names = np.all([letter in 'RGB' for letter in cols])
        if len(im[colour_axis]) == 1:
            im = im.squeeze(dim=colour_axis)
        elif len(im[colour_axis]) > 3:
            raise BadImage('Cannot output more than 3 colour channels')
        elif RGB_names:
            channels = {col:im[{colour_axis:i}] for i, col in enumerate(cols)}
            if len(channels) == 2:
                dummy = im[{colour_axis:0}].copy()
                dummy[:] = im.min()
                for i, col in enumerate('RGB'):
                     if col not in cols:
                         dummy[colour_axis] = col
                         channels[col] = dummy
                         channels['R'].attrs['_dummy_channel'] = i
                         break
            channels = [channels[col] for col in 'RGB']
            im = clean_concat(channels, colour_axis)
        elif len(im[colour_axis]) == 2:
            dummy = xr.full_like(im[{colour_axis:0}], fill_value=im.min())
            dummy = dummy.expand_dims({colour_axis: [np.NaN]})
            im.attrs['_dummy_channel'] = -1
            im = clean_concat([im, dummy], colour_axis)
    dim_order = [depth_axis, vert_axis, horiz_axis, colour_axis][:im.ndim]
    return im.transpose(*dim_order)


def show_sphere_cluster(s, color):
    """
    This function to show a 3D rendering of a Spheres obj hasn't worked since
    HoloPy 3.0, due to Mayavi compatibility issues. We keep the code because
    we hope to re-implement this functionality eventually.
    """
    raise NotImplementedError("3D renders of Spheres not currently supported")
    # Delayed imports to avoid hard dependencies on plotting packages and to
    # avoid the cost of importing them in noninteractive code
    from matplotlib import cm

    mlab = import_mayavi()

    # scale factor is 2 because mayavi interprets 4th
    # argument as a diameter, we keep track of radii
    # view is chosen to be looking from the incoming laser's point of view
    if color == 'rainbow':
        for i in arange(0,len(s.x)):
            numberofcolors = max(10,len(s.x))
            mlab.points3d(s.x[i], s.y[i], s.z[i], s.r[i],
                scale_factor=2.0, resolution=32,
                color=cm.gist_rainbow(float(i)/numberofcolors)[0:3])
            mlab.view(-90,0,s.z[:].mean())
    else:
        mlab.points3d(s.x, s.y, s.z, s.r, scale_factor=2.0,
            resolution=32, color=color)
        mlab.view(-90,0,s.z[:].mean())


def show_scatterer_slices(scatterer, spacing):
    """
    Show slices of a scatterer voxelation

    scatterer : .Scatterer
        scatterer to visualize
    spacing : float or (float, float, float)
        voxel spacing for the visualization
    """
    vol = scatterer.voxelate(spacing, 0)
    Show2D(display_image(vol))


def check_display():
    """Diagnostic test to check matplotlib backend.

    You should see a white square inside a black square, with a colorbar.
    Pressing the left or right arrow keys should cycle through z.
    You should see:
        Z = 0 : A white axes-aligned square
        Z = 1 : A white circle
        Z = 2 : A white diamond (square at 45 degrees)
    """
    a = np.zeros([100, 100, 3])
    a[25:75,25:75,0] = 1
    for i in range(25):
        for j in range(25):
            if i + j <= 25:
                a[50+i, 50+j, 1:3] = 1
                a[50-i, 50+j, 1:3] = 1
                a[50-i, 50-j, 1:3] = 1
                a[50+i, 50-j, 1:3] = 1
            elif i**2 + j**2 <= 25**2:
                a[50+i, 50+j, 1] = 1
                a[50-i, 50+j, 1] = 1
                a[50-i, 50-j, 1] = 1
                a[50+i, 50-j, 1] = 1
    show(a)
