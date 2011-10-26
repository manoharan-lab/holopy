"""
New custom display functions for holograms and reconstructions.

.. moduleauthor:: Tom Dimduk <tdimiduk@physics.harvard.edu>
"""

import numpy as np
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
            self.ax.set_title('image {0}'.format(self.i))
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
            pixel =  self.optics.pixel
            if hasattr(self.im, 'distances'):
                print('({0}, {1}, {2}), ({3}, {4}, {5})'.format(
                        event.ydata*pixel[0], event.xdata*pixel[1],
                        self.im.distances[self.i], int(round(event.ydata)),
                        int(round(event.xdata)), self.i))
            else:
                print('({0}, {1}), ({2}, {3})'.format(
                        event.ydata*pixel[0], event.xdata*pixel[1], int(round(event.ydata)),
                        int(round(event.xdata))))


    def __call__(self, event):
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

    optics = im.optics
    if np.iscomplexobj(im):
        if phase:
            im = np.angle(im)
        else:
            im = np.abs(im)
    
    plotter(im, i, t, optics)
