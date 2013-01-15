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
The centerfinder module is a group of functions for locating the
center of a particle or dimer in a hologram. This can be useful for
determining an initial parameter guess for hologram fitting.

.. moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>
.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
"""
from __future__ import division

import scipy
import numpy as np
from scipy import arange, around, array, zeros, sqrt, ndimage, transpose

def center_find(image, centers=1, scale=.5):
    """
    Finds the coordinates of the center of a holographic pattern
    The coordinates returned are in pixels (row number, column number).

    Intended for fiding the center of single particle or dimer
    holograms which basically show concentric circles. The optional
    scale parameter (between 0 and 1) gives a lower threshold of image
    intensity gradient relative to the maximum gradient (1) to take
    into account when finding the center.

    Parameters
    ----------
    image : ndarray
        image to find the center(s) in
    centers : int
        number of centers to find
    scale : float (optional)
        gradient magnitude threshold

    Returns
    -------
    res : ndarray
        row(s) and column(s) of center(s)

    Notes
    -----
    When scale is close to 1, the code will run quickly but may lack
    accuracy. When scale is set to 0, the gradient at all pixels will
    contribute to finding the centers and the code will take a little
    bit longer. The user should pay attention to how the magnitude of
    the gradients correlates with finding accurate centers.
    """
    x_deriv, y_deriv = image_gradient(image)
    res = hough(x_deriv, y_deriv, centers, scale)
    if centers==1:
        res=res[0]
    return res


def image_gradient(image):
    """
    Uses the sobel operator as a numerical approximation of a
    derivative to find the x and y components of the image's intensity
    gradient at each pixel.

    Parameters
    ----------
    image : ndarray
        image to find the gradient of

    Returns
    -------
    gradx : ndarray
        x-components of intensity gradient
    grady : ndarray
        y-components of intensity gradient
    """
    gradx = scipy.ndimage.sobel(image, axis = 0)
    # - sign here from old HoloPy coordinate convention?
    grady = -1*scipy.ndimage.sobel(image, axis=1)
    return gradx, grady


def hough(x_deriv, y_deriv, centers=1, scale=.25):
    """
    Following the approach of a Hough transform, finds the pixel which
    the most gradients point towards or away from. Uses only gradients
    with magnitude greater than scale*maximum gradient. Once the pixel
    is found, uses a brightness-weighted average around that pixel to
    refine the center location to return. After the first center is
    found, the sourrounding area is blocked out and another brightest
    pixel is searched for.

    Parameters
    ----------
    x_deriv : numpy.ndarray
        x-component of image intensity gradient

    y_deriv : numpy.ndarray
        y-component of image intesity gradient
    scale : float (optional)
        gradient magnitude threshold
    centers : int
        number of centers to find

    Returns
    -------
    res : ndarray
        row and column of center
    accumulator : ndarray
        accumulator array.  The maximum of this array should be the
        center of the hologram

    """
    #Finding the center: Using the derivatives we have already found
    #(effectively the gradient), we "draw" lines through pixels
    #parallel to the gradient and add all these lines together in the
    #array called "accumulator."  Because of the
    #concentric-circle-patterned hologram, the maximum of accumulator
    #should be the center of the pattern.  Rebecca W. Perry, Jerome Fung
    #11/20/2009

    #Edited by Rebecca Dec. 1, 2009 to include weighted average
    #Edited by Rebecca Perry June 9, 2011 to change default scale and
    #modify weighted averaging box size for centers close to the edges.

    accumulator = zeros(x_deriv.shape, dtype = int)
    dim_x = x_deriv.shape[0]
    dim_y = x_deriv.shape[1]
    gradient_mag = sqrt(x_deriv**2 + y_deriv**2)
    threshold = scale * gradient_mag.max()

    points_to_vote = scipy.where(gradient_mag > threshold)
    points_to_vote = array([points_to_vote[0], points_to_vote[1]]).transpose()
    #import pdb; pdb.set_trace()
    for coords in points_to_vote:
        # draw a line
        # add it to the accumulator
        if x_deriv[coords[0], coords[1]]==0:
            slope = y_deriv[coords[0], coords[1]]/.00001
        else:
            slope = y_deriv[coords[0], coords[1]]/x_deriv[coords[0], coords[1]]
        if slope > 1. or slope < -1.:
            # minus sign on slope from old convention?
            rows = arange(dim_x, dtype = 'int')
            line = around(coords[1] - slope * (rows - coords[0])).astype('int')
            cols_to_use = (line >= 0) * (line < dim_y)
            acc_cols = line[cols_to_use]
            #import pdb; pdb.set_trace()
            acc_rows = rows[cols_to_use]
        else:
            cols = arange(dim_y, dtype = 'int')
            line = around(coords[0] - 1./slope * (cols - 
                                                  coords[1])).astype('int')
            rows_to_use = (line >= 0) * (line < dim_x)
            acc_cols = cols[rows_to_use]
            acc_rows = line[rows_to_use]
        accumulator[acc_rows, acc_cols] += 1
    weightedRowNum=zeros(centers)
    weightedColNum=zeros(centers)
    for i in arange(0,centers):
        #m is row number, n is column number
        [m, n]=scipy.unravel_index(accumulator.argmax(), accumulator.shape)
        #brightness average around brightest pixel:
        boxsize = min(10, m, n, dim_x-1-m, dim_y-1-n) #boxsize changes with closeness to image edge
        small_sq = accumulator[m-boxsize:m+boxsize+1, n-boxsize:n+boxsize+1]
        #the part of the accumulator to average over
        rowNum, colNum = np.mgrid[m-boxsize:m+boxsize+1, n-boxsize:n+boxsize+1]
        #row and column of the revised center:
        weightedRowNum[i] = scipy.average(rowNum,None,small_sq)
        weightedColNum[i] = scipy.average(colNum,None,small_sq)
        accumulator[m-boxsize:m+boxsize+1, n-boxsize:n+boxsize+1]=accumulator.min()
    return transpose(array([weightedRowNum, weightedColNum]))
