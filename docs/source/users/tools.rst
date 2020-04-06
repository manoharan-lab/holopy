.. _tools:

HoloPy Tools
============

Holopy contains a number of tools to help you with common tasks when analyzing holograms.
This page provides a summary of the tools available, while full descriptions can be found
in the relevant code reference.

General Image Processing Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The tools described here are frequently used when analyzing holgrams. They are available from the ``holopy.core.process`` namespace.

The :func:`.normalize` function divides an image by its average, returning an
image with a mean pixel value of 1. Note that this is the same normalization
convention used by HoloPy when calculating holograms with :func:`.calc_holo`.

Cropping an image introduces difficulties in keeping track of the relative
coordinates of features within an image and maintaining metadata. By using the
:func:`.subimage` function, the image origin is maintained in the cropped image,
so coordinate locations of features (such as a scatterer) remain unchanged.

Since holograms of particles usually take the form of concentric rings, the
location of a scatterer can usually be found by locating the apparent center(s)
of the image. Use :func:`.center_find` to locate one or more centers in an
image.

You can remove isolated dead pixels with zero intensity (e.g. for a background
division) by using :func:`.zero_filter`. This function replaces the dead pixel
with the average of its neighbours, and fails if adjacent pixels have zero
intensity.

The :func:`.add_noise` function allows you to add Gaussian-correlated random
noise to a calculated image so that it more closely resembles experimental data.

To find gradient values at all points in an image, use :func:`.image_gradient`.
To simply remove a planar intensity gradient from an image, use
:func:`.detrend`. Note that this gives a mean pixel value of zero.

Frequency space analysis provides a powerful tool for working with images. Use
:func:`.fft` and :func:`.ifft` to perform fourier transforms and inverse fourier
transforms, respectively. These make use of ``numpy.fft`` functions, but are
wrapped to correctly interpret HoloPy objects. HoloPy also includes a Hough
transform (:func:`.hough`) to help identify lines and other features in your
images.


Math Tools
~~~~~~~~~~
HoloPy contains implementations of a few mathematical functions related to
scattering calculations. These functions are available from the
``holopy.core.math`` namespace.

To find the distance between two points, use :func:`.cartesian_distance`.

To rotate a set of points by arbitrary angles about the three coordinate axes,
use :func:`.rotate_points`. You can also calculate a rotation matrix with
:func:`.rotation_matrix` to save and use later.

To convert spherical coordinates into Cartesian coordinates, use
:func:`.to_cartesian`. To convert Cartesian coordinates into spherical
coordinates, use :func:`.to_spherical`.

When comparing data to a model, the chi-squared and r-squared values provide
measures of goodness-of-fit. You can access these through :func:`.chisq` and
:func:`.rsq`.

