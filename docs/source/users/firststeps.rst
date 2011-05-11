First steps
===========

Initially you'll probably find it helpful to work from a python shell
such as ipython.  In all the examples that follow we'll assume you've
done an ``import pylab`` which makes it easy to plot data or show
images. 

.. _loading:

Loading and viewing a hologram
------------------------------

Let's say you have a digital hologram stored in the file
``image0001.tif``.  Holopy can import many different image formats,
including TIFF files, numpy data format files, and any image format
that can be handled by the Python Imaging Library.  But it's always
good to load your images in Holopy and view them to see if they are
imported correctly.  You can do this as follows

.. sourcecode:: ipython

   In [1]: import holopy

   In [2]: holo = holopy.load('image0001.tif')

The function :func:`holopy.load` (which is an alias to :func:`holopy.io.load`
returns an instance of the :class:`holopy.hologram.Hologram` class. But it can
be treated just like an array of numbers (for experts: it is a subclass of
numpy.ndarray). So, for example, to view it you can just use

.. sourcecode:: ipython

   In [3]: pylab.imshow(holo)

If it's a grayscale image, it will probably look better with a
grayscale colormap:

.. sourcecode:: ipython

   In [4]: pylab.gray()

You can do math or image processing operations on ``holo`` just like
you would for a normal
`numpy <http://numpy.scipy.org/>`_ array.  For example (note you need
to ``import scipy`` for these to work)::

    filtered_image = scipy.ndimage.uniform_filter(holo, [10,10])
    ffted_image = scipy.fftpack.fft2(holo)

.. _metadata:

Telling holopy about your optical train
---------------------------------------

Analyzing a hologram isn't easy without some additional data that
tells us about how the hologram was created.  If we're going to
reconstruct a 3D volume from the hologram, for instance, we won't
be able to discern the length scales of anything in the reconstruction
unless we know the wavelength.  The wavelength is an example of
:dfn:`optical metadata`.  

In Holopy, this metadata is stored in a :class:`holopy.optics.Optics`
object.  This object can then be stored along with the image data in the
same :class:`holopy.hologram.Hologram` object, as follows:

.. sourcecode:: ipython

   In [1]: import holopy
   In [2]: opts = holopy.optics.Optics(wavelen=658e-9, \
                                        index=1.33, \
                                        pixel_scale=[0.1e-6,0.1e-6])
   In [3]: holo = holopy.load('image0001.tif', optics=opts)

In the first line above we create an instance of the
:class:`holopy.optics.Optics` from three pieces of metadata: the
imaging laser wavelength, the medium refractive index, and the pixel
size of the camera in the imaging plane.  You can specify more
metadata, but this is all we need for now.  The second line creates an
instance of the :class:`holopy.hologram.Hologram` class with the
hologram data (in image0001.tif) and the optics information.

You can also specify a :dfn:`background image`, which will
automatically be divided from the hologram before fitting and
reconstruction.  You might do this to correct for dust or other flaws
in the optical train:

.. sourcecode:: ipython

    In [3]: holo = holopy.load('image0001.tif', \
	   			 bg='../bg01.tif', optics=opts)

.. note::

    The wavelength specified in the :class:`holopy.optics.Optics` object
    is that in vacuum. When the hologram is reconstructed, the wavelength
    in the medium is used. The wavelength in the medium is stored in the
    optics object.

    .. sourcecode:: ipython

        In [4]: opts.med_wavelen
        Out[4]: 4.653e-7


Using YAML files
````````````````

You might alternatively store the optical metadata in a
`YAML <http://www.yaml.org/>`_ file. Information is stored in
the yaml file as `parameter_name` followed by a colon and
then the value. Comments in the yaml file may be added following 
the "#" character. 

The following text shows what one might want stored in their optics
yaml file. Such data can then be read into an instance of the
:class:`holopy.optics.Optics` object.::


    wavelen: 658e-9                      # Wavelength of light (in vacuum) used in creating holograms
    polarization: [0., 1.0]
    divergence: 0
    pixel_scale: [.1151e-6, .1151e-6]    # Size of camera pixel in the image plane
    index: 1.33                          # Index of medium


.. sourcecode:: ipython

    In [5]: optics = holopy.optics.Optics(**holopy.load_yaml('optics_file.yaml'))
	

:func:`holopy.load()` will also accept the filename of an optics yaml file as the argument for the optics parameter and automatically load the yaml file.  

