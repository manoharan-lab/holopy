.. _load_tutorial:

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

.. testcode::
    :hide:

    import holopy as hp
    from holopy.core.io import get_example_data_path
    imagepath = get_example_data_path('image01.jpg')

The first few lines just specify where to look for an image. 
The most important line actually loads the image so that you can work with it: 

..  testcode::
    
    raw_holo = hp.load_image(imagepath, spacing = 0.0851)

HoloPy can import many different image formats, including TIFF files,
numpy data format files, and any image format that can be handled by `Pillow
<http://pillow.readthedocs.io/en/3.3.x/handbook/image-file-formats.html>`_.
 
The spacing argument tells holopy about the scale of your image. Here, we had 
previously measured that each pixel is a square with side length 0.0851 microns.
You can also load an image without specifying a spacing value if you just want
to look at it, but most holopy calculations will fail on an image without spacing. 

The last line simply displays the loaded image on your screen. 

.. note ::
   
  If you know numpy, our :class:`.Image` is a `numpy
  <http://docs.scipy.org/doc/numpy/reference/arrays.html>`_ array
  subclass, so you can use all the math numpy provides.  For
  example:
    
  ..    testcode::
    
        import scipy.ndimage
        import scipy.fftpack
        filtered_image = scipy.ndimage.uniform_filter(raw_holo, [10,10])
        ffted_image = scipy.fftpack.fft2(raw_holo)

  ..    testcode::
        :hide:

        print(filtered_image[0,0])
        print(ffted_image[0,0])

  
..  testoutput::
    :hide:
        
    99.04
    (36600582+0j)


Correcting Noisy Images
-----------------------

The raw hologram has some non-uniform illumination and an artifact in the 
upper right hand corner from dust somewhere in the optics. These types of  
things can be removed if you are able to take a background image with the same optical setup but
without the object of interest. Dividing the raw hologram by the background can usually
improve the image a lot.

..  testcode::

    bgpath = get_example_data_path('bg01.jpg')
    bg = hp.load_image(bgpath, spacing = 0.0851)
    holo = raw_holo / bg
    hp.show(holo)

..  plot:: pyplots/show_bg_holo.py

If you are worried about stray light in your optical train, you should 
also subtract a dark-field image of your sample, recorded with no laser illumination.

..  testcode::

    dfpath = get_example_data_path('df01.jpg')
    df = hp.load_image(dfpath, spacing = 0.0851)
    holo = (raw_holo - df) / (bg - df)
    hp.show(holo)

..  testcode::
    :hide:
    
    print(holo[0,0])

..  testoutput::
    :hide:
    
    0.81746031746


.. _metadata:

Telling HoloPy about your Experimental Setup
--------------------------------------------

Recorded holograms are a product of the specific experimental setup that produced them.
The image only makes sense when considered with information about the experimental 
conditions in mind. When you load an image, you have the option to specify some of this
information in the form of :dfn:`metadata` that is associated with the image. In fact, we 
already saw an example of this when we specified image spacing above. The sample in our
image was immersed in water, which has a refractive index of 1.33. It was illuminated by
a red laser with wavelength of 660 nm and polarization in the x-direction. We can write:

..  testcode::

    holo.index = 1.33
    holo.wavelen = 0.660
    holo.polarization = (1.0, 0.0)

Alternatively, we can specify some or all of these parameters immediately when loading the image:

..  testcode::

    raw_holo = hp.load_image(imagepath, index = 1.33, wavelen = 0.660, spacing = 0.0851)

.. note::
    Spacing and wavelength must both be written in the same units - microns in the example
    above. Holopy has no built-in length scale and requires only that you be consistent. 
    For example, we could have specified both parameters in terms of nanometers instead.

..  testcode::
    :hide:
    
    print(raw_holo.index-holo.wavelen)

..  testoutput::
    :hide:
    
    0.67

Saving and Reloading Holograms
------------------------------

Once you have background-divided a hologram and associated it with metadata, you might
want to save it so that you can skip those steps next time you are working with the 
same image::
    
    hp.save('outfilename', holo)

This will save your processed image to a compact HDF5 file. In fact, you can use :func:`.save` 
on any holopy object. To reload a hologram with metadata you would write::

    holo = hp.load('outfilename')

If you would like to save your hologram to an image format for easy visualization, use::

    hp.save_image('outfilename', holo)

Additional options allow you to control how image intensity is scaled. Images saved as .tif (and other?)
formats will still contain metadata, which will be retrieved if you reload with :func:`.load`, but not :func:`.image_load`

..  note::

    Although holopy stores metadata even when writing to image files, it is still recommended that 
    holograms be saved to HDF5 using :func:`.save`. Floating point intensity values are rounded to
    8-bit integers when using :func:`.save_image`, resulting in information loss.


Non-Square Pixels
-----------------

The holograms above make use of several default assumptions.  When you
load an image like ::

  raw_holo = hp.load_image(imagepath, spacing = 0.0851)

you are making HoloPy assume a square array of evenly spaced grid
points. If your pixels are not square, you can provide pixel spacing values in each direction: 

..  testcode::

  raw_holo = hp.load_image(imagepath, spacing = (0.0851, 0.0426))

Most displays will default to displaying square pixels, but if your
hologram has an associated spacing (holo.spacing), and you use
holopy.show(holo) to display the image, your hologram will display
with pixels of the correct aspect ratio.
