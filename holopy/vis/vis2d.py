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
New custom display functions for holograms and reconstructions.

.. moduleauthor:: Tom Dimduk <tdimiduk@physics.harvard.edu>
"""


import numpy as np
import xarray as xr
from ..core.errors import BadImage
from ..core.metadata import get_spacing, get_values
from ..core.utils import ensure_array

class plotter:
    def __init__(self, im, plane_axes, slice_axis, starting_index, color_axis):
        # Delay the pylab import until we actually use it to avoid a hard
        # dependency on matplotlib, and to avoid paying the cost of importing it
        # for non interactive code
        import pylab
        

        self.axis_names = plane_axes
        self.step_name = slice_axis
        self.i = starting_index
        self.vmin = im.min()
        self.vmax = im.max()

        if isinstance(im, xr.DataArray):
            self.dims = im.dims
        else:
            self.dims = range(len(im.shape))

        self.selector={}
        for d in self.dims:
            if d not in self.axis_names and (color_axis is None or d not in color_axis):
                self.selector[d]= 0

        #to show non-square pixels correctly
        try:
            spacing = get_spacing(im)
            self.ratio = spacing[0]/spacing[1]
        except:
            #we are not working with a DataArray containing dimensions labeled 'x' and 'y'
            self.ratio = 1

        if color_axis is not None and len(im[color_axis]) == 2:
            new_ax = im.isel(**{color_axis:0}).copy()
            new_ax[:] = self.vmin
            #missing a dimension
            for col in ['red', 'green', 'blue']:
                if col not in im[color_axis].values:
                    new_ax[color_axis] = col
            self.im = self.vmax - xr.concat([im, new_ax], color_axis) + self.vmin
        else:
            self.im = im
        self.fig = pylab.figure()
        pylab.gray()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel(self.axis_names[1])
        self.ax.set_ylabel(self.axis_names[0])
        self.plot = None
        self.colorbar = None
        self.draw()
        self.fig.canvas.mpl_connect('key_press_event',self)
        self.fig.canvas.mpl_connect('button_press_event', self.click)


    def draw(self):

        if self.step_name is not None:
            self.selector[self.step_name] = self.i

        if isinstance(self.im, xr.DataArray):
            im = self.im.isel(**self.selector)

        else:
            im = ensure_array(self.im)
            counter = 0
            for key, val in self.selector.items():
                im = im.take(val, axis=key-counter)
                counter = counter + 1

        self._title()
        if self.plot is not None:
            self.plot.set_array(im)
        else:
            self.plot = self.ax.imshow(im, vmin=self.vmin, vmax=self.vmax,
                                       interpolation="nearest", aspect=self.ratio)

            #change the numbers displayed at the bottom to be in
            #HoloPy coordinate convention
            if hasattr(im, 'spacing') and im.spacing is not None:
                def user_coords(x, y):
                    s = ", units: {0[0]} = {1[0]:.1e}, {0[1]}={1[1]:.1e}"
                    return s.format(self.axis_names, self.location(x, y))
            else:
                def user_coords(x, y):
                    return ""
            def format_coord(x, y):
                # our coordinate convention is inverted from
                # matplotlib's default, so we need to swap x and y
                x, y = y, x
                s = "pixels: {0[0]} = {1[0]}, {0[1]} = {1[1]}"
                return (s.format(self.axis_names, self.pixel(x, y)) +
                        user_coords(x, y))
            self.ax.format_coord = format_coord


        if not self.colorbar:
            self.colorbar = self.fig.colorbar(self.plot)

    def pixel(self, x, y):
        index = [int(x+.5), int(y+.5)]
        if self.im.ndim == 3:
            index.append(self.i)
        return index

    def location(self, x, y):
        index = [np.interp(pos+.5, range(len(self.im[axis])), self.im[axis]) for axis, pos in zip(self.axis_names, [x, y])]
        if self.im.ndim == 3:
            index.append(self.im[self.step_name].values[self.i])
        return index


    def click(self, event):
        if event.ydata is not None and event.xdata is not None:
            x, y = np.array((event.ydata, event.xdata))
            if isinstance(self.im, xr.DataArray):
                print(("[{0}, {1}],".format(self.pixel(x, y), self.location(x, y))))
            else:
                print((self.pixel(x, y)))
            import sys; sys.stdout.flush()


    def __call__(self, event):
        if self.step_name is not None:
            if event.key=='right':
                if isinstance(self.im, xr.DataArray):
                    dim_len = len(self.im[self.step_name])
                else:
                    dim_len = self.im.shape[self.step_name]
                self.i = min(dim_len - 1, self.i + 1)
            elif event.key == 'left':
                self.i = max(0, self.i-1)
            if self.selector[self.step_name] != self.i:
                self.draw()
                self.fig.canvas.draw()

    def _title(self):
        titlestring = ""
        if hasattr(self.im, 'distances'):
            titlestring += "z={0},i={1}".format(self.im.distances[self.i],
                                                self.i)
        elif hasattr(self.im, 'filenames'):
            titlestring += self.im.filenames[self.i]
        elif self.im.ndim > 2:
            titlestring += "image {0}".format(self.i)
        if titlestring is not "":
            self.ax.set_title(titlestring)

def show2d(im, plane_axes=None, slice_axis=None, starting_index=0, color_axis=None, phase = False):
    """
    Display a hologram or reconstruction

    Allows scrolling through multidimensional holograms or reconstructions.
    Defaults to showing magnitude of complex images

    Parameters
    ----------
    im : ndarray
       Image to be shown
    z0 : int
       slice along z dimension to show first.  
    t : int
       slice along time to show for reconstructions.

    """
 
    shape = list(np.shape(im))
    if len(shape) + (slice_axis is None) <3:
        raise BadImage("Image does not have enough dimensions to display properly.")    

    if isinstance(im, xr.DataArray):
        if plane_axes is None and 'x' in im.dims and 'y' in im.dims:
            plane_axes = ('x','y')
        if slice_axis is None and 'z' in im.dims:
            slice_axis = 'z'
        if color_axis is None and 'illumination' in im.dims:
            color_axis = 'illumination'



    # Default is to display the two longest axes and step through third longest
    axes = []
    for i in range(min(len(shape),3)):
        axes.append(shape.index(max(shape)))
        shape[shape.index(max(shape))] = -1

    if plane_axes is None:
        plane_axes = axes[0:2]
        if slice_axis in plane_axes:
            plane_axes[plane_axes.index(slice_axis)] = axes[2]

    if slice_axis is None and len(shape)>2:
        for i in plane_axes:
            try:
                axes.remove(i)
            except ValueError:
                pass
        slice_axis=axes[0]

    if np.iscomplexobj(im):
        if phase:
            im = np.angle(im)
        else:
            im = np.abs(im)

    plotter(im, plane_axes, slice_axis, starting_index, color_axis)

def show_scatterer_slices(scatterer, spacing):
    """
    Show slices of a scatterer voxelation

    scatterer : .Scatterer
        scatterer to visualize
    spacing : float or (float, float, float)
        voxel spacing for the visualization
    """
    vol = scatterer.voxelate(spacing, 0)
    show2d(vol)
