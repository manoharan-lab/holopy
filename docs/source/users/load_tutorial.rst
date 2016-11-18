.. _load_tutorial:

Loading Data
============

HoloPy can work with any kind of image data, but we use it for digital
holograms, so our tutorials will focus mostly on hologram data.

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
previously measured that each pixel is a square with side length 0.0851 microns.
In general, you should specify ``spacing`` as the distance between adjacent pixel centres.
You can also load an image without specifying a spacing value if you just want
to look at it, but most holopy calculations will give erroneous results on such an image. 

The final line simply displays the loaded image on your screen 
with the built-in HoloPy :func:`.show` function. 

Correcting Noisy Images
~~~~~~~~~~~~~~~~~~~~~~~

The raw hologram has some non-uniform illumination and an artifact in the 
upper right hand corner from dust somewhere in the optics. These types of  
things can be removed if you are able to take a background image with the same optical setup but
without the object of interest. Dividing the raw hologram by the background using :func:`.bg_correct` 
can usually improve the image a lot.

..  testcode::

    bgpath = get_example_data_path('bg01.jpg')
    bg = hp.load_image(bgpath, spacing = 0.0851)
    holo = hp.core.process.bg_correct(raw_holo, bg)
    hp.show(holo)

..  plot:: pyplots/show_bg_holo.py

Often, it is beneficial to record multiple background images. In this case,
we want an average image to pass into :func:`.bg_correct` as our background. 

..  testcode::
     
    bgpath = get_example_data_path(['bg01.jpg', 'bg02.jpg', 'bg03.jpg'])
    bg = hp.core.io.load_average(bgpath, refimg = raw_holo)
    holo = hp.core.process.bg_correct(raw_holo, bg)
    hp.show(holo)

Here, we have used :func:`.load_average` to construct an average of the three background
images specified in ``bgpath``. The ``refimg`` argument allows us to specify a reference
image that is used to provide spacing and other metadata to the new, averaged image. 

If you are worried about stray light in your optical train, you should 
also capture a dark-field image of your sample, recorded with no laser illumination.

..  testcode::

    dfpath = get_example_data_path('df01.jpg')
    df = hp.load_image(dfpath, spacing = 0.0851)
    holo = hp.core.process.bg_correct(raw_holo, bg, df)
    hp.show(holo)

..  testcode::
    :hide:
    
    print(holo.values[0,0,0])

..  testoutput::
    :hide:
    
    0.919642857143

.. _metadata:

Telling HoloPy about your Experimental Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recorded holograms are a product of the specific experimental setup that produced them.
The image only makes sense when considered with information about the experimental 
conditions in mind. When you load an image, you have the option to specify some of this
information in the form of :dfn:`metadata` that is associated with the image. In fact, we 
already saw an example of this when we specified image spacing earlier. The sample in our
image was immersed in water, which has a refractive index of 1.33. It was illuminated by
a red laser with wavelength of 660 nm and polarization in the x-direction. We can write:

..  testcode::

    holo = hp.core.update_metadata(holo, medium_index = 1.33, illum_wavelen = 0.660, illum_polarization = (1,0))

You can then view these metadata values as attributes of holo, as in ``holo.medium_index``.
However, you must use :func:`.update_metadata` to edit them. Alternatively, we can specify
some or all of these parameters immediately when loading the image:

..  testcode::

    raw_holo = hp.load_image(imagepath, medium_index = 1.33, illum_wavelen = 0.660, spacing = 0.0851)

.. note::
    Spacing and wavelength must both be written in the same units - microns in the example
    above. Holopy has no built-in length scale and requires only that you be consistent. 
    For example, we could have specified both parameters in terms of nanometers instead.

..  testcode::
    :hide:
    
    print(holo.medium_index-holo.illum_wavelen)
    print(raw_holo.medium_index-raw_holo.illum_wavelen)

..  testoutput::
    :hide:
    
    0.67
    0.67

Saving and Reloading Holograms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have a background-divided hologram and associated it with metadata, you might
want to save it so that you can skip those steps next time you are working with the 
same image::
    
    hp.save('outfilename', holo)

saves your processed image to a compact HDF5 file. In fact, you can use :func:`.save` 
on any holopy object. To reload your same hologram with metadata you would write::

    holo = hp.load('outfilename')

If you would like to save your hologram to an image format for easy visualization, use::

    hp.save_image('outfilename', holo)

Additional options of :func:`.save_image` allow you to control how image intensity is scaled. 
Images saved as .tif (the default) will still contain metadata, which will be retrieved if
you reload with :func:`.load`, but not :func:`.load_image`

..  note::

    Although holopy stores metadata even when writing to .tif image files, it is still recommended that 
    holograms be saved in HDF5 format using :func:`.save`. Floating point intensity values are rounded 
    to 8-bit integers when using :func:`.save_image`, resulting in information loss.


Non-Square Pixels
~~~~~~~~~~~~~~~~~

The holograms above make use of several default assumptions.  When you load an image like ::

  raw_holo = hp.load_image(imagepath, spacing = 0.0851)

you are making HoloPy assume a square array of evenly spaced grid
points. If your pixels are not square, you can provide pixel spacing values in each direction: 

..  testcode::

  raw_holo = hp.load_image(imagepath, spacing = (0.0851, 0.0426))

Most displays will default to displaying square pixels but if you 
use HoloPy's built-in :func:`.show` function to display the image, your hologram will display
with pixels of the correct aspect ratio.
