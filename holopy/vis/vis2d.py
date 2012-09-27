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
New custom display functions for holograms and reconstructions.

.. moduleauthor:: Tom Dimduk <tdimiduk@physics.harvard.edu>
"""
from __future__ import division

import numpy as np
from ..propagation import propagate
import pylab

class plotter:
    def __init__(self, im, i=0, j=0, optics=None):
        self.im = im
        self.optics = optics
        self.i = i
        self.j = j
        self.vmin = im.min()
        self.vmax = im.max()
        self.fig = pylab.figure()
        pylab.gray()
        self.ax = self.fig.add_subplot(111)
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

#        pylab.show()

        import sys; sys.stdout.flush()
        if self.plot is not None:
            self.plot.set_array(im)
        else:
            self.plot = self.ax.imshow(im, vmin=self.vmin, vmax=self.vmax,
                                       interpolation="nearest")
        if not self.colorbar:
            self.colorbar = self.fig.colorbar(self.plot)

    def click(self, event):
        if event.ydata is not None and event.xdata is not None:
            coord = np.array((event.ydata, event.xdata))
            origin = np.zeros(3)
            if self.im.origin is not None:
                origin = self.im.origin
            coord = tuple(coord.round().astype('int'))

            if self.im.ndim == 3:
                if getattr(self.im, 'spacing', None) is not None:
                    z =  self.im.spacing[2] * self.i + origin[2]
                    
                    print('{0}, {1}'.format(tuple(np.append(coord, self.i)),
                                            tuple(np.append(coord * self.im.spacing[:2] + origin[:2], z))))
                else:
                    print(coord)
            else:
                if getattr(self.im, 'spacing', None) is not None:
                    spacing = self.im.spacing[:2]
                    print('{0}, {1}'.format(coord, tuple(coord * spacing +
                                            origin[:2])))
                else:
                    print(coord)

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
        else:
            titlestring += "image {0}".format(self.i)

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
       less than 4d array)
       
    """

    if hasattr(im, 'optics'):
        optics = im.optics
    else:
        optics = None
    if np.iscomplexobj(im):
        if phase:
            im = np.angle(im)
        else:
            im = np.abs(im)
    
    plotter(im, i, t, optics)


def infocuscheck(hologram, scatterer, offset = 0):
    """
    Display a raw hologram, a calculated hologram, and reconstructions
    of both.

    Parameters
    ----------
    hologram : hologram object
       Hologram to be shown
    scatterer : scatterer object
       Scattering object to calculate hologram from
    offset : float
       Offset from reconstructing at the z-distance given by the
       mean z-distance of the scatterer.
       
    """
    distance = scatterer.centers[:,2].mean()+offset
    #reconstruct the hologram
    r = propagate(hologram,distance)
    #reconstruct the scatterer
    theory = Multisphere(hologram.optics,[256,256])
    tmat = theory.calc_holo(scatterer)
    r2 = propagate(tmat,distance)
    #show the holograms and reconstructions  
    pylab.figure()
    scalemin = min(abs(r[:,:,0,0]).min(), abs(r2[:,:,0,0]).min())
    scalemax = max(abs(r[:,:,0,0]).max(), abs(r2[:,:,0,0]).max())
    pylab.subplot(2,2,1)
    pylab.imshow(hologram)
    pylab.gray()
    pylab.title('Hologram')
    pylab.subplot(2,2,2)
    pylab.imshow(tmat)
    pylab.title('Calculated from Scatterer')
    pylab.subplot(2,2,3)
    pylab.imshow(abs(r[:,:,0,0]),vmin = scalemin,vmax = scalemax)
    pylab.title(str(distance))
    pylab.subplot(2,2,4)
    pylab.imshow(abs(r2[:,:,0,0]),vmin = scalemin,vmax = scalemax)
    pylab.title(str(distance))
    pylab.show()

# was in scattering.geometry
def viewcluster(cluster):
    #this is not elegant, but lets you look at the cluster from three angles
    #to check if it is the cluster you wanted
    #warning: the particles are not shown to scale!!!!!! (markersize is in points)
    dist = distances(cluster)
    pyplot.figure(figsize=[14,4])
    pyplot.subplot(1,3,1)
    l = pyplot.plot(cluster.centers[:,0]-cluster.centers[:,0].mean(),
        cluster.centers[:,1]-cluster.centers[:,1].mean(),'ro')
    pyplot.setp(l, 'markersize', 60)
    pyplot.xlim(-dist.max(),dist.max())
    pyplot.ylim(-dist.max(),dist.max())
    pyplot.subplot(1,3,2)
    l = pyplot.plot(cluster.centers[:,0]-cluster.centers[:,0].mean(),
        cluster.centers[:,2]-cluster.centers[:,2].mean(),'ro')
    pyplot.setp(l, 'markersize', 60)
    pyplot.xlim(min(pyplot.xlim()[0],pyplot.ylim()[0]),max(pyplot.xlim()[1],pyplot.ylim()[1]))
    pyplot.ylim(min(pyplot.xlim()[0],pyplot.ylim()[0]),max(pyplot.xlim()[1],pyplot.ylim()[1]))
    pyplot.xlim(-dist.max(),dist.max())
    pyplot.ylim(-dist.max(),dist.max())
    pyplot.subplot(1,3,3)
    l = pyplot.plot(cluster.centers[:,1]-cluster.centers[:,1].mean(),
        cluster.centers[:,2]-cluster.centers[:,2].mean(),'ro')
    pyplot.setp(l, 'markersize', 60)
    pyplot.xlim(min(pyplot.xlim()[0],pyplot.ylim()[0]),max(pyplot.xlim()[1],pyplot.ylim()[1]))
    pyplot.ylim(min(pyplot.xlim()[0],pyplot.ylim()[0]),max(pyplot.xlim()[1],pyplot.ylim()[1]))
    pyplot.xlim(-dist.max(),dist.max())
    pyplot.ylim(-dist.max(),dist.max())

