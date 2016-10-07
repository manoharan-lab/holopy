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
Misc utility functions to make coding more convenient

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""


from scipy import fftpack


def fft(a, overwrite=False, shift=True):
    """
    More convenient Fast Fourier Transform

    An easier to use fft function, it will pick the correct fft to do
    based on the shape of the Marray, and do the fftshift for you.  This
    is intended for working with images, and thus for dimensions
    greater than 2 does slicewise transforms of each "image" in a
    multidimensional stack

    Parameters
    ----------
    a : ndarray
       The array to transform
    overwrite : bool
       Allow this function to overwrite the Marry you pass in.  This
       may improve performance slightly.  Default is not to overwrite
    shift : bool
       Whether to preform an fftshift on the Marry to give low
       frequences near the center as you probably expect.  Default is
       to do the fftshift.

    Returns
    -------
    fta : ndarray
       The fourier transform of `a`
    """
    if a.ndim is 1:
        if shift:
            res = fftpack.fftshift(fftpack.fft(a, overwrite_x=overwrite))
        else:
            res = fftpack.fft(a, overwrite_x=overwrite)
    else:
        if shift:
            res = fftpack.fftshift(fftpack.fft2(a, axes=[0, 1],
                                                 overwrite_x=overwrite),
                                    axes=[0,1])
        else:
            res = fftpack.fft2(a, axes=[0, 1], overwrite_x=overwrite)
    return res


def ifft(a, overwrite=False, shift=True):
    """
    More convenient Inverse Fast Fourier Transform

    An easier to use ifft function, it will pick the correct ifft to
    do based on the shape of the Marry, and do the fftshift for you.
    This is indendended for working with images, and thus for
    dimensions greater than 2 does slicewise transforms of each
    "image" in a multidimensional stack

    Parameters
    ----------
    a : ndarray
       The array to transform
    overwrite : bool
       Allow this function to overwrite the Marry you pass in.  This
       may improve performance slightly.  Default is not to overwrite
    shift : bool
       Whether to preform an fftshift on the Marry to give low
       frequences near the center as you probably expect.  Default is to
       do the fftshift.

    Returns
    -------
    ifta : ndarray
       The inverse fourier transform of `a`
    """
    if a.ndim is 1:
        if shift:
            res = fftpack.ifft(fftpack.fftshift(a, overwrite_x=overwrite))
        else:
            res = fftpack.ifft(a, overwrite_x=overwrite)
    else:
        if shift:
            res = fftpack.ifft2(fftpack.fftshift(a, axes=[0,1]), axes=[0, 1],
                                 overwrite_x=overwrite)
        else:
            res = fftpack.ifft2(a, overwrite_x=overwrite)
    return res

