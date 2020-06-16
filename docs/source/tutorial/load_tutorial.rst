.. _load_tutorial:

Loading Data
============

HoloPy can work with any image data, but our tutorials will focus on holograms.

Loading and viewing a hologram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We include a couple of example holograms with HoloPy. Lets start by
loading and viewing one of them
  
.. plot:: pyplots/show_example_holo.py
   :include-source:

.. testcode::
    :hide:

    import holopy as hp
    from holopy.core.io import get_example_data_path
    imagepath = get_example_data_path('image01.jpg')

The first few lines just specify where to look for an image. 
The most important line actually loads the image so that you can work with it: 

..  testcode::
    
    raw_holo = hp.load_image(imagepath, spacing = 0.0851)

HoloPy can import any image format that can be handled by `Pillow
<http://pillow.readthedocs.io/en/3.3.x/handbook/image-file-formats.html>`_.
 
The spacing argument tells holopy about the scale of your image. Here, we had 
previously measured that each pixel is a square with side length 0.0851
micrometers. In general, you should specify ``spacing`` as the distance between
adjacent pixel centres. You can load an image without a spacing value if you
just want to look at it, but holopy calculations will give incorrect results. 

The final line displays the loaded image on your screen with the built-in 
HoloPy :func:`.show` function. If you don't see anything, you may need to set
your matplotlib backend. Refer to :ref:`usage` for instructions. 

Correcting Noisy Images
~~~~~~~~~~~~~~~~~~~~~~~

The raw hologram has some non-uniform illumination and an artifact in the 
upper-right corner. These can be removed if you take a background image with
the same optical setup but without the object of interest. Dividing the raw
hologram by the background using :func:`.bg_correct` improves the image a lot.

..  testcode::

    from holopy.core.process import bg_correct
    bgpath = get_example_data_path('bg01.jpg')
    bg = hp.load_image(bgpath, spacing = 0.0851)
    holo = bg_correct(raw_holo, bg)
    hp.show(holo)

..  plot:: pyplots/show_bg_holo.py

Often, it is beneficial to record multiple background images. In this case,
we want an average image to pass into :func:`.bg_correct` as our background. 

..  testcode::
     
    bgpath = get_example_data_path(['bg01.jpg', 'bg02.jpg', 'bg03.jpg'])
    bg = hp.core.io.load_average(bgpath, refimg = raw_holo)
    holo = bg_correct(raw_holo, bg)
    hp.show(holo)

Here, we have used :func:`.load_average` to construct an average of the three background
images specified in ``bgpath``. The ``refimg`` argument allows us to specify a reference
image that is used to provide spacing and other metadata to the new averaged image. 

If you are worried about stray light in your optical train, you should 
also capture a dark-field image of your sample, recorded with no laser illumination.
A dark-field image is specified as an optional third argument to :func:`.bg_correct`.

..  testcode::

    dfpath = get_example_data_path('df01.jpg')
    df = hp.load_image(dfpath, spacing = 0.0851)
    holo = bg_correct(raw_holo, bg, df)
    hp.show(holo)

..  testcode::
    :hide:
    
    print(holo.values[0,0,0])

..  testoutput::
    :hide:
    
    0.91964285...

Holopy includes some other convenient tools for manipulating image data.
See the :ref:`tools` page for details.

.. _metadata:

Telling HoloPy about your Experimental Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recorded holograms are a product of the specific experimental setup that
produced them. The image only makes sense when considered with the experimental 
conditions in mind. When you load an image, you have the option to specify some
of this information in the form of :dfn:`metadata` that is associated with the
image. In fact, we already saw an example of this when we specified image
spacing earlier. The sample in our image was immersed in water (refractive
index 1.33). Illumination was by a red laser with wavelength 660 nm and
polarization in the x-direction. We can tell HoloPy all of this information
when loading the image:

..  testcode::

    raw_holo = hp.load_image(imagepath, spacing=0.0851, medium_index=1.33, illum_wavelen=0.660, illum_polarization=(1,0))


You can then view these metadata values as attributes of ``raw_holo``, as in ``raw_holo.medium_index``.
However, you must use a special function :func:`.update_metadata` to edit them. If we forgot to 
specify metadata when loading the image, we can use :func:`.update_metadata` to add it later:

..  testcode::

    holo = hp.core.update_metadata(holo, medium_index=1.33, illum_wavelen=0.660, illum_polarization=(1,0))

.. note::
    Spacing and wavelength must be given in the same units - micrometers in the
    example above. Holopy has no built-in length scale and requires only that
    you be consistent. For example, we could have specified both parameters in
    terms of nanometers or meters instead.

..  testcode::
    :hide:
    
    print(holo.medium_index-holo.illum_wavelen)
    print(raw_holo.medium_index-raw_holo.illum_wavelen)

..  testoutput::
    :hide:
    
    0.67
    0.67

HoloPy images are stored as `xarray DataArray <http://xarray.pydata.org/en/stable/data-structures.html>`_ objects.
xarray is a powerful tool that makes it easy to keep track of metadata and extra image dimensions, distinguishing between
a time slice and a volume slice, for example. While you do not need any knowledge of xarray to use HoloPy, some
familiarity will make certain tasks easier. This is especially true if you want to directly manipulate data
before or after applying HoloPy's built-in functions.

Saving and Reloading Holograms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have background-divided a hologram and associated it with metadata, you might
want to save it so that you can skip those steps next time you are working with the 
same image::
    
    hp.save('outfilename', holo)

saves your processed image to a compact HDF5 file. In fact, you can use :func:`.save` 
on any holopy object. To reload your same hologram with metadata you would write::

    holo = hp.load('outfilename')

If you would like to save your hologram to an image format for easy visualization, use::

    hp.save_image('outfilename', holo)

Additional options of :func:`.save_image` allow you to control how image intensity is scaled. 
Although HoloPy stores metadata when writing to .tif image files, you should save
holograms in HDF5 format using :func:`.save` to avoid rounding.
