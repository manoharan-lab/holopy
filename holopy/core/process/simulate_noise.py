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
"""Add simulated noise to images. Intended for use with exact
calculated images to make them look more like noisy 'real'
measurements.

.. moduleauthor:: Tom G. Dimiduk <tdimiduk@physics.harvard.edu>
"""
import numpy as np
from scipy.ndimage import gaussian_filter

def add_noise(image, noise_mean=.1, smoothing=.01, poisson_lambda=1000):
    """Adds simulated noise to an image

    Real image noise usually has correlation, so we smooth the raw
    random variable. The noise_mean can be controlled independently of
    the poisson_lambda that controls the shape of the distribution. In
    general, you can stick with our default of a large poisson_lambda
    (ie for imaging conditions not near the shot noise limit).

    Defaults are set to give noise vaguely similar to what we tend to
    see in our holographic imaging.

    Parameters
    ----------
    image : ndarray or Image
        The image to add noise to.
    intensity : float
        How large the noise mean should be relative to the image mean
    smoothing : float
        Fraction of the image size to smooth by. Should in general be << 1
    poisson_lambda : float
        Used to compute the shape of the noise distribution. You can generally
        leave this at its default value unless you are simulating shot noise
        limited imaging.

    Returns
    -------
    noisy_image : ndarray
       A copy of the input image with noise added.

    """
    return image + simulate_noise(image.shape, noise_mean, smoothing,
                                  poisson_lambda) * image.mean()


def simulate_noise(shape, mean=.1, smoothing=.01, poisson_lambda=1000):
    raw_poisson = np.random.poisson(poisson_lambda, shape)
    smoothed = gaussian_filter(raw_poisson, np.array(shape)*smoothing)
    return smoothed/smoothed.mean() * mean
