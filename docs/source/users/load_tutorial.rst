.. _loading:

Loading Data
============

HoloPy can work with any kind of image data, but we use it for digital
holograms, so our tutorials will focus mostly on hologram data.

Loading and viewing a hologram
------------------------------

We include a couple of example holograms with HoloPy. Lets start by
loading and viewing one of them
  
.. plot:: pyplots/show_example_holo.py
   :include-source:

But you probably want to look at your data not ours. If you have an
``image.tif`` in your current directory, you can do::

  import holopy as hp
  holo = hp.load('image.tif')
  hp.show(holo)

HoloPy can import many different image formats, including TIFF files,
numpy data format files, and any image format that can be handled by
the `Python Imaging Library
<http://www.pythonware.com/products/pil/>`_. 

If you are able to take an image with the same optical setup but
without the object of interest, removing that background can usually
improve the image a lot.  Suppose the background image is saved as
``bg.tif``. Then you can divide it out by::

  bg = hp.load('bg.tif')
  holo = holo / bg

where the original image has been replaced with one with the
background removed. This usually does a fairly good job of correcting
for nonuniform illumination, dust elsewhere in the optics, and things
of that sort.

.. note ::
   
  If you know numpy, our :class:`.Image` is a `numpy
  <http://docs.scipy.org/doc/numpy/reference/arrays.html>`_ array
  subclass, so you can use all the math numpy provides.  For
  example::

    import scipy.ndimage
    import scipy.fftpack
    filtered_image = scipy.ndimage.uniform_filter(holo, [10,10])
    ffted_image = scipy.fftpack.fft2(holo)

.. _metadata:

Telling HoloPy about your Experimental Setup
--------------------------------------------

Simply loading a TIFF won't help you analyze your data, since the
image file generally won't tell you where the camera is with respect
to the light source or how the images were recorded. This additional
information is referred to as :dfn:`metadata`, which must be
included when you do actual calculations on your data.

In order to be able to do calculations with your data, you will need
to specify this metadata when you load your image::

   import holopy as hp
   optics = hp.core.Optics(wavelen=.660, index=1.33, 
                           polarization=[1.0, 0.0])
   holo = hp.load('image.tif', spacing = .1,  optics = optics)

Above, we have created an instance of the :class:`.Optics` metadata
class for incident light at 660 nm (.66 micron) (in vacuum)
propagating in a medium with refractive index 1.33, and with a
polarization in the x-direction. The pixel spacing of the image is 100
nm (.1 micron).  You can simulate the effect of adding an objective
lens in the optical path simply by reducing the specified spacing by
the magnification factor of the objective::
  
  magnification = 40
  holo = hp.load('image.tif', spacing = 4.0/magnification,  
                 optics = optics)
  

.. note::

    HoloPy uses the given wavelength and medium refractive
    index to calculate the wavelength in the medium, which
    is available as: ::

        optics.med_wavelen
        0.49624060150375937

You may have noticed that the very first example loaded a ``.yaml``
file instead of a tiff image. HoloPy's native data format is ``.yaml``
files which can store all of our metadata. So in that example, we
provide the metadata in the file. For more information about loading
and saving HoloPy ``.yaml`` files see :ref:`yaml_ref`.
