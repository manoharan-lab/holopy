First steps
===========

HoloPy works with many types of data, but most of our examples will
use images obtained from digital holography. Before executing the
example scripts below, it may be helpful to change to the root
directory of your HoloPy installation (e.g. ``/home/me/holopy``), or
append this directory to your Python path.  Then you can copy and
paste the examples directly into a Python shell such as IPython, which
will allow you to explore and visualize the results of the script.

.. _loading:

Loading and viewing a hologram
------------------------------

HoloPy can import many different image formats, including TIFF files,
numpy data format files, and any image format that can be handled by
the `Python Imaging Library
<http://www.pythonware.com/products/pil/>`_.  It's often helpful to
load your images in HoloPy and view them to see that they have
imported correctly.  Let's say you have a digital hologram stored in
the file ``image.tif``: ::

   import holopy as hp
   holo = hp.load('image.tif')
   hp.show(holo)

The function :func:`hp.load <holopy.core.io.io.load>` returns an
:class:`.Image`, a 2D array of the pixel values of the image.

Subtracting a background image taken with the same optical setup as
that for ``image.tif`` but without the object of interest can frequently
improve the hologram.
Suppose the background image is saved as ``bg.tif``. Then you can
subtract it: ::

  bg = hp.load('bg.tif')
  holo = holo - bg

where the original image has been replaced with one where the background
is subtracted.

.. note ::
   
  You can also do math or image processing operations on ``holo`` just like
  you would on a normal `numpy
  <http://docs.scipy.org/doc/numpy/reference/arrays.html>`_ array.  For
  example::

    import scipy.ndimage
    import scipy.fftpack
    filtered_image = scipy.ndimage.uniform_filter(holo, [10,10])
    ffted_image = scipy.fftpack.fft2(holo)

.. _metadata:

Telling holopy about your optical train
---------------------------------------

Simply loading a TIFF won't help you analyze your data, since the
image file generally won't tell you where the camera is with respect
to the light source or how the images were recorded. This additional
information is referred to as :dfn:`metadata`, which must be
included when you do actual calculations on your data.

All of the objects HoloPy uses for storing data also support the
addition of such metadata.  The most common metadata are pixel size and
optical information, described by an :class:`.Optics` object.
Metadata can be loaded in :func:`hp.load
<holopy.core.io.io.load>` along with the image data: ::

   import holopy as hp
   optics = hp.core.Optics(wavelen=.660, index=1.33, \
                           polarization=[1.0, 0.0])
   holo = hp.load('image.tif', pixel_size = .1,  optics = optics)

Above, we have created an instance of the :class:`.Optics` metadata
class for incident light at 660 nm (in vacuum) propagating in a medium
with refractive index 1.33, and with a polarization in the
x-direction. The pixel size of the image is 100 nm.  You can simulate
the effect of adding an objective lens in the optical path simply by
reducing the physical pixel size of your detector by the magnification
of the objective.

.. note::

    HoloPy uses the given wavelength and medium refractive
    index to calculate the wavelength in the medium, which
    is available as: ::

        optics.med_wavelen
        0.49624060150375937

More advanced methods for saving and loading objects in HoloPy (e.g.
as :ref:`yaml_ref`) can be found in the :ref:`Details` section of the
User Guide.
