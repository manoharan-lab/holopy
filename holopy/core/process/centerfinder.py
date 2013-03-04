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
centers of holographic ring patterns. The module can find the center
of a single-sphere holographic pattern, a dimer holographic pattern,
or the centers of multiple (well-separated: clearly separate ring 
patterns with separate centers) single spheres or dimers. The intended 
use is for determining an initial parameter guess for hologram fitting.

We thank the Grier Group at NYU for suggesting the use of the Hough 
transform. For their independent implementation of a Hough-based 
holographic feature detection algorithm, see:
http://physics.nyu.edu/grierlab/software/circletransform.pro
For a case study and further reading, see:
F. C. Cheong, B. Sun, R. Dreyfus, J. Amato-Grill, K. Xiao, L. Dixon
& D. G. Grier, Flow visualization and flow cytometry with holographic 
video microscopy, Optics Express 17, 13071-13079 (2009).

.. moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>
.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
"""

from __future__ import division

import numpy as np
from .enhance import normalize
from scipy import ndimage

def center_find(image, centers=1, threshold=.5):
    """
    Finds the coordinates of the center of a holographic pattern.
    The coordinates returned are in pixels (row number, column
    number). Intended for finding the center of single particle or
    dimer holograms which basically show concentric circles. The
    optional threshold parameter (between 0 and 1) gives a bound on
    what magnitude of gradients to include in the calculation. For
    example, threshold=.75 means ignore any gradients that are less
    than 75% of the maximum gradient in the image.

    Parameters
    ----------
    image : ndarray
        image to find the center(s) in
    centers : int
        number of centers to find
    threshold : float (optional)
        fraction of the maximum gradient below which all
        other gradients will be ignored (range 0-.99)

    Returns
    -------
    res : ndarray
        row(s) and column(s) of center(s)

    Notes
    -----
    When scale is close to 1, the code will run quickly but may lack
    accuracy. When scale is set to 0, the gradient at all pixels will
    contribute to finding the centers and the code will take a little
    bit longer.
    """
    col_deriv, row_deriv = image_gradient(image)
    res = hough(col_deriv, row_deriv, centers, threshold)
    if centers==1:
        res = res[0]
    return res


def image_gradient(image):
    """
    Uses the Sobel operator as a numerical approximation of a
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
    image = normalize(image)
    grad_col = ndimage.sobel(image, axis=0)
    grad_row = -ndimage.sobel(image, axis=1)
    return grad_col.astype(float), grad_row.astype(float)


def hough(col_deriv, row_deriv, centers=1, threshold=.25):
    """
    Following the approach of a Hough transform, finds the pixel which
    the most gradients point towards or away from. Uses only gradients
    with magnitudes greater than threshold*maximum gradient. Once the
    pixel is found, uses a brightness-weighted average around that
    pixel to refine the center location to return. After the first
    center is found, the sourrounding area is blocked out and another
    brightest pixel is searched for if more centers are required.

    Parameters
    ----------
    col_deriv : numpy.ndarray
        y-component of image intensity gradient
    row_deriv : numpy.ndarray
        x-component of image intensity gradient
    centers : int
        number of centers to find
    threshold : float (optional)
        fraction of the maximum gradient below which all
        other gradients will be ignored (range 0-.99)

    Returns
    -------
    res : ndarray
        row and column of center or centers
    """
    #Finding the center: Using the derivatives we have already found
    #(effectively the gradient), we "draw" lines through pixels
    #parallel to the gradient and add all these lines together in the
    #array called "accumulator."  Because of the
    #concentric-circle-patterned hologram, the maximum of accumulator
    #should be the center of the pattern.  
    #Rebecca W. Perry, Jerome Fung 11/20/2009
    #Edited by Rebecca Dec. 1, 2009 to include weighted average
    #Edited by Rebecca Perry June 9, 2011 to change default scale and
    #modify weighted averaging box size for centers 
    #close to the edges.
    
    accumulator = np.zeros(col_deriv.shape, dtype = int)
    dim_x = col_deriv.shape[0]
    dim_y = col_deriv.shape[1]
    gradient_mag = np.sqrt(col_deriv**2 + row_deriv**2)
    abs_threshold = threshold * gradient_mag.max()

    points_to_vote = np.where(gradient_mag > abs_threshold)
    points_to_vote = np.array([points_to_vote[0], 
            points_to_vote[1]]).transpose()

    for coords in points_to_vote:
        # draw a line, and add it to the accumulator
        if col_deriv[coords[0], coords[1]]==0:
            slope = row_deriv[coords[0], coords[1]]/.00001
        else:
            slope = row_deriv[coords[0], 
                coords[1]]/col_deriv[coords[0], coords[1]]
        
        if slope > 1. or slope < -1.:
            rows = np.arange(dim_x, dtype = 'int')
            line = np.around(coords[1] - slope * 
                (rows - coords[0])).astype('int')
            cols_to_use = (line >= 0) * (line < dim_y)
            acc_cols = line[cols_to_use]
            acc_rows = rows[cols_to_use]
        else:
            cols = np.arange(dim_y, dtype = 'int')
            if slope==0:
                slope = 0.00001
            line = np.around(coords[0] - 1./slope * 
                (cols - coords[1])).astype('int')
            rows_to_use = (line >= 0) * (line < dim_x)
            acc_cols = cols[rows_to_use]
            acc_rows = line[rows_to_use]
        
        accumulator[acc_rows, acc_cols] += 1

    weightedRowNum = np.zeros(centers)
    weightedColNum = np.zeros(centers)
    
    for i in np.arange(0,centers):
        #m is row number, n is column number
        [m, n] = np.unravel_index(accumulator.argmax(),
                accumulator.shape)
                
        #brightness average around brightest pixel:
        boxsize = min(10, m, n, dim_x-1-m, dim_y-1-n)
        
        #boxsize changes with closeness to image edge
        small_sq = accumulator[m-boxsize:m+boxsize+1, 
                n-boxsize:n+boxsize+1]
                
        #the part of the accumulator to average over
        rowNum, colNum = np.mgrid[m-boxsize:m+boxsize+1, 
                n-boxsize:n+boxsize+1]
                
        #row and column of the revised center:
        weightedRowNum[i] = np.average(rowNum,None,small_sq)
        weightedColNum[i] = np.average(colNum,None,small_sq)
        accumulator[m-boxsize:m+boxsize+1, 
                n-boxsize:n+boxsize+1]=accumulator.min()
        
    return np.array([weightedRowNum, weightedColNum]).T
