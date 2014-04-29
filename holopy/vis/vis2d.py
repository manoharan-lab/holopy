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
New custom display functions for holograms and reconstructions.

.. moduleauthor:: Tom Dimduk <tdimiduk@physics.harvard.edu>
"""
from __future__ import division

import numpy as np
from ..core.marray import squeeze

class plotter:
    def __init__(self, im, i=0, j=0, axis_names = ('x', 'y')):
        # Delay the pylab import until we actually use it to avoid a hard
        # dependency on matplotlib, and to avoid paying the cost of importing it
        # for non interactive code
        import pylab

        self.im = im
        self.axis_names = axis_names
        self.i = i
        self.j = j
        self.vmin = im.min()
        self.vmax = im.max()
        self.fig = pylab.figure()
        pylab.gray()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel(axis_names[1])
        self.ax.set_ylabel(axis_names[0])
        self.plot = None
        self.colorbar = None
        self.draw()
        self.fig.canvas.mpl_connect('key_press_event',self)
        self.fig.canvas.mpl_connect('button_press_event', self.click)

    def draw(self):
        if self.im.ndim is 2:
            im = self.im
        if self.im.ndim is 3:
            im = self.im[...,self.i]
        elif self.im.ndim is 4:
            im = self.im[...,self.i,self.j]
        self._title()

        #to show non-square pixels correctly
        if  hasattr(im, 'spacing') and im.spacing is not None:
            ratio = im.spacing[0]/im.spacing[1]
        else:
            ratio = 1.0

        if self.plot is not None:
            self.plot.set_array(im)
        else:
            self.plot = self.ax.imshow(im, vmin=self.vmin, vmax=self.vmax,
                                       interpolation="nearest", aspect=ratio)

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
        index = [x, y]
        if self.im.ndim == 3:
            index.append(self.i)
        return [p * s + c for p, s, c in
                zip(index, self.im.spacing, self.im.origin)]


    def click(self, event):
        if event.ydata is not None and event.xdata is not None:
            x, y = np.array((event.ydata, event.xdata))
            if getattr(self.im, 'spacing', None) is not None:
                print("{0}, {1}".format(self.pixel(x, y), self.location(x, y)))
            else:
                print(self.pixel(x, y))
            import sys; sys.stdout.flush()


    def __call__(self, event):
        if len(self.im.shape) > 2:
            old_i = self.i
            old_j = self.j
            if event.key=='right':
                self.i = min(self.im.shape[2]-1, self.i+1)
            elif event.key == 'left':
                self.i = max(0, self.i-1)
            elif event.key == 'up':
                self.j = min(self.im.shape[3]-1, self.j+1)
            elif event.key == 'down':
                self.j = max(0, self.j-1)
            if old_i != self.i or old_j != self.j:
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

def show2d(im, i=0, t=0, phase = False):
    """
    Display a hologram or reconstruction

    Allows scrolling through multidimensional holograms or reconstructions.
    Defaults to showing magnitude of complex images

    Parameters
    ----------
    im : ndarray
       Image to be shown
    i : int
       slice along third dimension to show first.  For holograms this will be
       time, for reconstructions this will be z
    t : int
       slice along t to show for reconstructions.  Ignored for holograms (or any
       less than 4D array)

    """
    if isinstance(im, (list, tuple)):
        im = np.dstack(im)

    # Switch and show an x-z or y-z plane if the im has unit extent in
    # y or x
    axis_names = ['x', 'y']
    if im.shape[0] == 1:
        axis_names = ['y', 'z']
    if im.shape[1] == 1:
        axis_names = ['x', 'z']

    im = squeeze(im)



    if np.iscomplexobj(im):
        if phase:
            im = np.angle(im)
        else:
            im = np.abs(im)

    plotter(im, i, t, axis_names = axis_names)

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
