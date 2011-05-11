# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca W. Perry,
# Jerome Fung, and Ryan McGorty
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
Code to propagate objects/waves using scattering models.  

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Ryan McGorty <mcgorty@fas.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

import numpy as np
import scipy as sp
from holopy.process.math import fft, ifft
from holopy.hologram import Hologram
from holopy.utility.helpers import _ensure_pair

# May eventually want to have this function take a propagation model
# so that we can do things other than convolution

def propagate(holo, d, ft=None, fourier_filter=None, squeeze=True,
              gradient_filter=False, rec_zx_plane=False, project_along_z=False):
    """
    Propagates a hologram a distance d along the optical axis.

    Uses scalar diffraction theory calculate a hologram at a given distance
    away.  If you are doing a series of propagations of the same image (like in
    a reconstruction stack), you can save time by doing the fourier transform
    once outside this function.  If you do not provide the fourier transform, it
    will be computed here.  Propagate can apply an arbitrary array to
    the data as a fourier filter, this will be multiplied by the hologram in the
    fourier domain.  If you want to apply several, just multiply them together.
    This functionallity is used for efficiently applying contrast enhancing
    masks to data.

    Parameters
    ----------
    holo : Hologram
       Hologram to propagate
    d : float
       Distance to propagate, in meters
    ft : ndarray<complex> (optional)
       Fourier transform of image to propagate
    fourier_filter : ndarray<complex> (optional)
       Fourier filter to apply to reconstructed data
    squeeze : Boolean (optional)
       If True, remove size 1 dimensions (so if a single z is provided return
       holo will be a 2d array
    gradient_filter : float
       For each distance, compute a second propagation a distance
       gradient_filter away and subtract.  This enhances contrast of rapidly
       varying features
    rec_zx_plane : Boolean (optional)
       Set to True to reconstruct a plane along the z-dimension, this will
       average along the shorter dimension in x and y and return a 2d array of
       (x or y),  z
    project_along_z : Boolean (optional)
       Set to True to reconstruct a projection of z-slices, d should be an
       array of distances, return will be a sum of reconstructions at those
       distances 

    Returns
    -------
    holo(d) : Hologram
       The hologram progagated to a distance d from its current location.  
        
    Notes
    -----
    propagate is used primarily to support reconstructions, but is separated out
    because occasionally you want to propagate a hologram without invoking the
    rest of the reconstruction machinery.  
    """
    
    m,n = holo.shape[:2]
    # To quickly reconstruct along the z-direction, we'll take a 1D slide
    # of the hologram and reconstruct over a range of z.
    # To get the 1D hologram, we'll sum over the shortest axis. 
    if rec_zx_plane:
        if m>n:
            dim_im = m
            holo = holo.sum(axis=1)
        else:
            dim_im = n
            holo = holo.sum(axis=0)
            
    G = trans_func([m,n], holo.optics.pixel_scale, holo.optics.med_wavelen, d,
                   squeeze=False, gradient_filter=gradient_filter,
                   recon_in_zx_plane=rec_zx_plane, zprojection=project_along_z)
    # Half dimensions of the psf, they will be used to define the area
    # on which it is applied
    mm, nn = [dim/2 for dim in G.shape[:2]]
    if ft is None:
        ft = fft(holo)
    else:
        # make a local copy so that we don't modify the ft we are passed
        ft = ft.copy()

    if fourier_filter is not None:
        ft *= fourier_filter

    if rec_zx_plane:
        ft = np.repeat(ft[:,np.newaxis,...], G.shape[1], axis=1)

        ft[(dim_im/2-mm):(dim_im/2+mm),:] *= G[:(mm*2),:]
        # Transfer function may not cover the whole image, any values outside it
        # need to be set to zero to make the reconstruction correct
        ft[0:dim_im/2-mm,...]=0
        ft[dim_im/2+mm:m,...]=0

        ift = sp.zeros_like(ft)

        for i in range(0,ft.shape[1]):
            ift[:,i] = ifft(ft[:,i],overwrite=True)
            
        return ift
        
    if ft.ndim is 2:
        ft = ft.reshape(ft.shape[0], ft.shape[1], 1)
    ft = np.repeat(ft[:,:,np.newaxis,...], G.shape[2], axis=2)
    ft[(m/2-mm):(m/2+mm),(n/2-nn):(n/2+nn),...] *= G[:(mm*2),:(nn*2),:,np.newaxis]
    # Transfer function may not cover the whole image, any values outside it
    # need to be set to zero to make the reconstruction correct
    ft[0:n/2-nn,...]=0
    ft[n/2+nn:n,...]=0
    ft[:,0:m/2-mm,...]=0
    ft[:,m/2+mm:m,...]=0

    
    if squeeze:
        return np.squeeze(ifft(ft, overwrite=True))
    else:
        return ifft(ft, overwrite=True)

def trans_func(shape, pixel, wavelen, d, cfsp=0, squeeze=True,
               gradient_filter=0, recon_in_zx_plane=False, zprojection=False):
    """
    Calculates the optical transfer function to use in reconstruction

    This routine uses the analytical form of the transfer function
    found in in Kreis [1].  It can optionally do cascaded free-space
    propagation for greater accuracy [2] , although the code will run
    slightly slower.

    Parameters
    ----------
    shape : (int, int)
       maximum dimensions of the transfer function
    pixel: (float, float)
       effective pixel dimensions
    wavelen: float
       recording wavelength
    d: float or list of floats
       reconstruction distance.  If list or array, this function will
       return an array of transfer functions, one for each distance 
    cfsp: integer (optional)
       cascaded free-space propagation factor.  If this is an integer
       > 0, the transfer function G will be calculated at d/csf and
       the value returned will be G**csf.
    squeeze: Bool (optional)
       Remove length 1 dimensions (so that if only one distance is specified
       trans_func will be a 2d array)
    gradient_filter: float (optional)
       Subtract a second transfer function a distance gradint_filter from each z
    recon_in_zx_plane: Boolean (optional)
       Calculate a 2d transfer function along z and (x or y) [pick the larger
       dimension].  
    zprojection: Boolean (optional)
       Set to True to return a sum of the transfer functions at distances z,
       used to calculate a projected hologram

    Returns
    -------
    trans_func : np.ndarray
       The calculated transfer function.  This will be at most as large as
       shape, but may be smaller if the frequencies outside that are zero

    References
    ----------
    [1] Kreis, Handbook of Holographic Interferometry (Wiley,
    2005), equation 3.79 (page 116)

    [2] Kreis, Optical Engineering 41(8):1829, section 5
    
    """
    d = np.array([d])

    pixel = _ensure_pair(pixel)
    shape = _ensure_pair(shape)
       
    dx, dy = pixel
    xdim, ydim = shape

    if recon_in_zx_plane:
        #first determine which axis to use along x-y (that is, x or y)
        if xdim>ydim:
            xy_length = xdim
            d_xy = dx
        else:
            xy_length = ydim
            d_xy = dy

        d = d.reshape([1,d.size])
        
        #max_xy = int(np.min(xy_length**2*d_xy**2/d/wavelen/2))+1

        #max_xy = min(xy_length,max_xy*2)/2
        
        max_xy = xy_length/2.

        m_xy = sp.arange(-max_xy, max_xy)
        m_xy = m_xy.reshape(m_xy.size, 1)
        
        root = 1.+0j - (wavelen*m_xy/xy_length/d_xy)**2

        g = np.exp(1j*2*np.pi*d/wavelen*np.sqrt(root))

        g = g*(root>=0)

        return g

    d = d.reshape([1,1,d.size])
    
    if(cfsp > 0):
       cfsp = int(abs(cfsp)) # should be nonnegative integer
       d = d/cfsp

    # The transfer function is only defined on a finite domain of
    # spatial frequencies; outside this domain the transfer function
    # is zero (see Kreis, Optical Engineering 41(8):1829, page 1836).
    # It is important to set it to zero outside the domain, otherwise
    # the reconstruction is not correct.  Here I save memory by making
    # the size of the array only as large as the domain corresponding
    # to the transfer function at the smallest z-distance

    # for this we need to use the magnitude of d, size of the image should be a
    # positive number
    max_m = int(np.max(xdim**2*dx**2/np.abs(d)/wavelen/2))+1
    max_n = int(np.max(ydim**2*dy**2/np.abs(d)/wavelen/2))+1

    
    # make sure that the array is not larger than the hologram if we
    # are using cascaded free space propagation
    max_m = min(xdim,max_m*2)/2
    max_n = min(ydim,max_n*2)/2
   
    m, n= np.ogrid[-max_m:max_m,-max_n:max_n]
    
    
    root = 1.+0j-(wavelen*n/(xdim*dx))**2 - (wavelen*m/(ydim*dy))**2

    root *= (root >= 0)
    
    # add the z axis to this array so it broadcasts correctly
    root = root[...,np.newaxis]

    g = np.exp(-1j*2*np.pi*d/wavelen*np.sqrt(root))

    if gradient_filter:
        g -= np.exp(-1j*2*np.pi*(d+gradient_filter)/wavelen*np.sqrt(root))

    # set the transfer function to zero where the sqrt is imaginary
    # (this is equivalent to making sure that the largest spatial
    # frequency is 1/wavelength).  (root>=0) returns a boolean matrix
    # that is equal to 1 where the condition is true and 0 where it is
    # false.  Multiplying by this boolean matrix masks the array.
    g = g*(root>=0)

    if cfsp > 0:
        g = g**cfsp

    if zprojection and len(g.shape)>2:
        g = g.sum(axis=-1)
        g = g[...,np.newaxis]

    if squeeze:
        return np.squeeze(g)
    else:
        return g

