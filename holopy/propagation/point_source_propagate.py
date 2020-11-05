# Copyright 2011-2017, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, Anna Wang, Solomon Barkley, Aaron Goldfain
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

import numpy as np
from holopy.core.metadata import get_spacing, data_grid, copy_metadata
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import fsolve
from xarray import concat


def ps_propagate(data, d, L, beam_c, out_schema = None):
    '''
    Propagates light back through a hologram that was taken using a
    diverging reference beam.

    Parameters
    ----------
    data is a holopy Xarray. It is the hologram to reconstruct. Must be
    square. The pixel spacing must also be square.
    d = distance from pinhole to reconstructed image, in meters (this is
    z in Jericho and Kreuzer). Can be a scalar or a 1D list or array.
    L = distance from screen to pinhole, in meters
    beam_c = [x,y] coodinates of beam center, in pixels
    out_schema = size of output image and pixel spacing, default is the schema
    of data.

    Returns
    -------
    an image(volume) corresponding to the reconstruction at plane(s) d.

    Notes
    -----
    Only propagation through media with refractive index 1 is supported.
    This is a wrapper function for ps_propagate_plane()
    This function can handle a single reconstruction plane or a volume.

    Based on the algorithm described in Manfred H. Jericho and H. Jurgen
    Kreuzer, "Point Source Digital In-Line Holographic Microscopy,"
    Chapter 1 of Coherent Light Microscopy, Springer, 2010.
    http://link.springer.com/chapter/10.1007%2F978-3-642-15813-1_1
    '''

    #handle a list of reconstruction planes
    if isinstance(d, list) or isinstance(d, np.ndarray):

        # save time by getting portion of recontruction that doesn't change
        # when z changes
        old_Ip, npix_plane = ps_propagate_plane(
            data, d[0], L, beam_c, out_schema, old_Ip=True)

        # Loop through each value of d.
        # This saves memory because only the cropped output image is stored.
        result = [
            ps_propagate_plane(data, z, L ,beam_c, out_schema, old_Ip = old_Ip)
            for z in d]
        result = concat(result, dim='z')

    else:  # if only reconstructing at one plane
        result = ps_propagate_plane(
            data, d, L ,beam_c, out_schema, old_Ip=False)

    return result


def ps_propagate_plane(data, d, L, beam_c, out_schema = None, old_Ip = False):
    '''
    Propagates light back through a hologram that was taken using a diverging
    reference beam.

    Parameters
    ----------
    data is a holopy Xarray. It is the hologram to reconstruct. Must be square.
    The pixel spacing must also be square.
    d = distance from pinhole to reconstructed image, in meters (this is z in
    Jericho and Kreuzer). Must be a scalar.
    L = distance from screen to pinhole, in meters
    beam_c = [x,y] coodinates of beam center, in pixels
    out_schema = size of output image and pixel spacing, default is the schema
    of data.
    if Ip == True, returns Ip to be used on calculations in the stack
    if Ip == False compute reconstructed image as normal
    if Ip is an image, use this to speed up calculations

    Returns
    -------
    returns an image(volume) corresponding to the reconstruction at plane(s) d.

    Notes
    -----
    Propataion can be to one plane only.
    Only propagation through media with refractive index 1 is supported.

    Based on the algorithm described in Manfred H. Jericho and H. Jurgen
    Kreuzer, "Point Source Digital In-Line Holographic Microscopy," Chapter 1
    of Coherent Light Microscopy, Springer, 2010.
    http://link.springer.com/chapter/10.1007%2F978-3-642-15813-1_1
    '''

    npix0 = float(len(data.x))  # size of original image in pixels
    wavelen = float(data.illum_wavelen)  # laser wavelength in meters
    n_medium = float(data.medium_index)  # not used (assumes n_medium = 1)
    datavals = data.values.squeeze()

    Dx, Dy = get_spacing(data)  # size of pixels on camera

    if out_schema is None:
        # mag = 1
        out_spacing = Dx
    else:
        # mag = Dx/get_spacing(out_schema)[0]
        out_spacing = get_spacing(out_schema)[0]

    #get number of pixels to reconstruct given the desired output spacing
    def X0_f(npix):
        result = -Dx * (beam_c[0] + (npix - npix0)*0.5)
        return result

    def to_solve(npix):
        result = (X0_f(npix)+(npix-1)*Dx) / np.sqrt(L**2 + (X0_f(npix)+(npix-1)*Dx)**2) - X0_f(npix)/np.sqrt(L**2-X0_f(npix)**2) - wavelen/out_spacing
        return result

    npix = int(fsolve(to_solve, npix0)[0])

    #npix = npix0*mag #number of pixels to reconstruct (this is an older way of doing the magnification)

    #center coordinates
    i_c = beam_c[0] + (npix - npix0)/2
    j_c = beam_c[1] + (npix - npix0)/2

    #set (X0,Y0) so beam center is at index (i_c,j_c)
    X0=-i_c*Dx
    Y0=-j_c*Dy

    #Scaling constants (eqn 1.32)
    X0p = X0*L/np.sqrt(L*L+X0*X0)
    Y0p = Y0*L/np.sqrt(L*L+Y0*Y0)
    con = X0+(npix-1)*Dx #useful constant
    Dxp = L*con/npix/np.sqrt(L*L+con*con) - L*X0/npix/np.sqrt(L*L+X0*X0) #Delta_x^prime
    con = Y0+(npix-1)*Dy #useful constant
    Dyp = L*con/npix/np.sqrt(L*L+con*con) - L*Y0/npix/np.sqrt(L*L+Y0*Y0) #Delta_y^prime

    #scale actually used in reconstructed image
    spacing = wavelen*L/npix/np.array([Dxp,Dyp]) #calculate 'magic' spacing (eqn 1.34).

    #useful constant
    ikz = 2j*np.pi*d/wavelen # this is (ikz)

    #Calculate I'(X,Y) (eqn 1.27)
    print('Calculating Ip')

    def Ip_calc(i,j):

        # (X',Y') coordinates corresponding to indecies (i,j)
        Xp=X0p+i*Dxp
        Yp=Y0p+j*Dyp

        #Useful constant (this is L/R')
        L_over_Rp = L**2-Xp**2-Yp**2
        L_over_Rp = np.where(L_over_Rp >= 0, L_over_Rp, 0.0)
        L_over_Rp = L/np.sqrt(L_over_Rp)
        L_over_Rp = np.where(L_over_Rp == np.inf, 0.000001, L_over_Rp)



        if isinstance(old_Ip,bool):

            # (X,Y) coordinate in original image
            X = Xp*L_over_Rp
            Y = Yp*L_over_Rp

            # (X,Y) indecies of original image, but (npix,npix) in size
            i_X = np.array( (X-X0)/Dx )
            i_Y = np.array( (Y-Y0)/Dy )

            i_X = i_X - (npix - npix0)/2
            i_Y = i_Y - (npix - npix0)/2

            i_X = i_X.astype(int)
            i_Y = i_Y.astype(int)

            if old_Ip:  # returns partially computed I'
                result = interpolate2D(datavals,i_X,i_Y,0) * L_over_Rp**4
            else: #returns full I'
                result = interpolate2D(datavals,i_X,i_Y,0) * L_over_Rp**4 * np.exp(ikz/L_over_Rp)

        else:
            result = old_Ip * np.exp(ikz/L_over_Rp)

        return result

    #get I'
    result = np.fromfunction(lambda i,j: Ip_calc(i,j), (npix, npix), dtype=int) #result is I'

    if isinstance(old_Ip,bool) and old_Ip: # returns partially computed I' and uncropped size of reconstruction
        return result, npix

    #compute final result, K_nm (eqn 1.33)
    i2Pi_over_N = 2j*np.pi/npix # this is i*2pi/N
    phase_factor = np.fromfunction(lambda i,j: np.exp( -i2Pi_over_N * (i*i_c + j*j_c) ), (npix, npix), dtype=int)
    print('Taking FFT')
    result = np.fft.ifft2(
        np.fft.fftshift(result * phase_factor, axes=[0, 1]),
        axes=[0, 1])

    print('Multiplying prefactor')
    phase_factor = np.fromfunction(lambda i,j: np.exp( i2Pi_over_N * ((i-i_c)*X0p/Dxp + (j-j_c)*Y0p/Dyp) ), (npix, npix), dtype=int)

    result = Dxp*Dyp*phase_factor*result


    #crop to correct size
    if npix > npix0:
        x_cen = int(npix/2)
        y_cen = int(npix/2)

        if out_schema is None:
            offset = int(npix0/2)
        else:
            offset = int(len(out_schema.x)/2)
        result = result [x_cen - offset : x_cen + offset, y_cen - offset : y_cen + offset]


    #return Image result
    return copy_metadata(data, data_grid(result, spacing=spacing, z=d))


def interpolate2D(data,i,j,fill=None):

    ''' Interpolates values from a 2D array (data) given non-integer indecies i and j.
    If [i,j] is outside of the shape of data, fill is returned.
    If fill=None, the value of the closest edge pixel to (i,j) is used.
    '''


    #only access pixels in the range of the image by replacing out of bounds indecies with an edge index

    #create arrays of pixels to index
    i_range = i
    j_range = j

    #replace out of range pixels with edge pixels
    i_range = np.where(i <= (data.shape[0]-1), i_range, data.shape[0]-1.0)
    i_range = np.where(i >= 0, i_range, 0.0)
    j_range = np.where(j <= (data.shape[1]-1), j_range, data.shape[1]-1.0)
    j_range = np.where(j >= 0, j_range, 0.0)

    #do interpolation
    result = RectBivariateSpline(np.arange(data.shape[0]), np.arange(data.shape[1]), data).ev(i_range, j_range)

    #replace the values that were out of bounds with fill
    #if (not isinstance(fill, str) ):
    if fill != None:
        result = np.where(i <= (data.shape[0]-1), result, fill)
        result = np.where(i >= 0, result, fill)
        result = np.where(j <= (data.shape[1]-1), result, fill)
        result = np.where(j >= 0, result, fill)

    return result

