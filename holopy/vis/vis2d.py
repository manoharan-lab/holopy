"""
New custom display functions for holograms and reconstructions.

.. moduleauthor:: Tom Dimduk <tdimiduk@physics.harvard.edu>
"""

import numpy as np
import holopy as hp
from scatterpy.theory import Multisphere
import pylab
from holopy.utility.helpers import _ensure_pair

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
            self.plot = self.ax.imshow(im, interpolation="nearest")
#        self.ax.imshow(im, vmin=self.vmin, vmax=self.vmax,
#                       interpolation="nearest")

    def click(self, event):
        if event.ydata is not None and event.xdata is not None:
            coord = np.array((event.ydata, event.xdata))

            pixel = None
            if hasattr(self.im, 'optics'):
                pixel =  self.optics.pixel
            if pixel == None and hasattr(self.im, 'holo'):
                pixel = self.im.holo.optics.pixel

            distance = None
            if hasattr(self.im, 'distances'):
                distance = self.im.distances[self.i]

            print('{0}, {1}'.format(tuple(coord.round().astype('int')),
                                    tuple(np.append(coord * pixel, distance))))


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
        if hasattr(self.im, 'distances'):
            self.ax.set_title("z={0},i={1}".format(self.im.distances[self.i],
                                                   self.i))
        else:
            self.ax.set_title("image {0}".format(self.i))
        
def show(im, i=0, t=0, phase = False):
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
    r = hp.reconstruct(hologram,distance)
    #reconstruct the scatterer
    theory = Multisphere(hologram.optics,[256,256])
    tmat = theory.calc_holo(scatterer)
    r2 = hp.reconstruct(tmat,distance)
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
