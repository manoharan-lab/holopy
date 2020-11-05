.. _ps_recon_tutorial:

Reconstructing Point Source Holograms
=====================================

Holograms are typically reconstructed optically by shining light back through
them. This corresponds mathematically to propagating the field stored in the
hologram to some different plane. The propagation performed here assumes that
the hologram was recorded using a point source (diverging spherical wave) as the
light source. This is also known as lens-free holography. Note that this is
different than propagation calculations where a collimated light source (plane
wave) is used. For recontructions using a plane wave see :ref:`recon_tutorial`.

This point-source propagation calculation is an implementation of the algorithm
that appears in `Jericho and Kreuzer 2010
<http://link.springer.com/chapter/10.1007%2F978-3-642-15813-1_1>`_. Curently,
only square input images and propagation through media with a refractive index
of 1 are supported.

Example Reconstruction
~~~~~~~~~~~~~~~~~~~~~~

.. plot:: pyplots/basic_ps_recon.py
   :include-source:

We'll examine each section of code in turn. The first block:

..  testcode::

    import holopy as hp
    import numpy as np
    from holopy.core.io import get_example_data_path
    from holopy.propagation import ps_propagate
    from scipy.ndimage.measurements import center_of_mass

loads the relevant modules. The second block:

..  testcode::
    
    imagepath = get_example_data_path('ps_image01.jpg')
    bgpath = get_example_data_path('ps_bg01.jpg')
    L = 0.0407 # distance from light source to screen/camera
    cam_spacing = 12e-6 # linear size of camera pixels
    mag = 9.0 # magnification
    npix_out = 1020 # linear size of output image (pixels)
    zstack = np.arange(1.08e-3, 1.18e-3, 0.01e-3) # distances from camera to reconstruct

defines all parameters used for the reconstruction. Numpy's linspace 
was used to define a set of distances at 10-micron intervals to 
propagate our image to. You can also propagate to a single distance
or to a set of distances obtained in some other fashion. The third
block: 

..  testcode::
   
    holo = hp.load_image(imagepath, spacing=cam_spacing, illum_wavelen=406e-9, medium_index=1) # load hologram
    bg = hp.load_image(bgpath, spacing=cam_spacing) # load background image
    holo = hp.core.process.bg_correct(holo, bg+1, bg) # subtract background (not divide)
    beam_c = center_of_mass(bg.values.squeeze()) # get beam center
    out_schema = hp.core.detector_grid(shape=npix_out, spacing=cam_spacing/mag) # set output shape

reads in a hologram and subtracts the corresponding background 
image. If this is unfamiliar to you, please review the 
:ref:`load_tutorial` tutorial. The third block also finds the center 
of the reference beam and sets the size and pixel spacing of the 
output images.

Finally, the actual propagation is accomplished with 
:func:`.ps_propagate` and a cropped region of the result is 
displayed. See the :ref:`recon_tutorial` page for details on
visualizing the reconstruction results.

..  testcode::

    recons = ps_propagate(holo, zstack, L, beam_c, out_schema) # do propagation
    hp.show(abs(recons[:,350:550,450:650])) # display result

..  testoutput::
    :hide:

    Calculating Ip
    Calculating Ip
    Taking FFT
    Multiplying prefactor
    Calculating Ip
    Taking FFT
    Multiplying prefactor
    Calculating Ip
    Taking FFT
    Multiplying prefactor
    Calculating Ip
    Taking FFT
    Multiplying prefactor
    Calculating Ip
    Taking FFT
    Multiplying prefactor
    Calculating Ip
    Taking FFT
    Multiplying prefactor
    Calculating Ip
    Taking FFT
    Multiplying prefactor
    Calculating Ip
    Taking FFT
    Multiplying prefactor
    Calculating Ip
    Taking FFT
    Multiplying prefactor
    Calculating Ip
    Taking FFT
    Multiplying prefactor
    Calculating Ip
    Taking FFT
    Multiplying prefactor

..  testcode::
    :hide:

    print(round(abs(recons[0,450,550].values), 20))

..  testoutput::
    :hide:

    1.370980655e-11

Magnification and Output Image Size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unlike the case where a collimated beam is used as the illumination
and the pixel spacing in the reconstruction is the same as in the 
original hologram, for lens-free reconstructions the pixel spacing
in the reconstruction can be chosen arbitrarily. In order to magnify 
the reconstruction the spacing in the reconstruction plane should be 
smaller than spacing in the original hologram. In the code above, the
magnification of the reconstruction can be set using the variable 
``mag``, or when calling :func:`.ps_propagate` directly the desired
pixel spacing in the reconstruction is specified through the 
spacing of ``out_schema``. Note that the output spacing will not be
the spacing of ``out_schema`` exactly, but should be within a few
percent of it. We recommend calling :func:`~holopy.core.metadata.get_spacing` on ``recons`` 
to get the actual spacing used.

Note that the total physical size of the plane that is reconstructed 
remains the same when different output pixel spacings are used. This 
means that reconstructions with large output spacings will only have
a small number of pixels, and reconstructions with small output
spacings will have a large number of pixels. If the linear size (in 
pixels) of the total reconstruction plane is smaller than 
``npix_out``, the entire reconstruction plane will be returned. 
However, if the linear size of total reconstruction plane is
larger than ``npix_out``, only the center region of the 
reconstruction plane with linear size ``npix_out`` is returned.

In the current version of the code, the amount of memory needed to 
perform a reconstruction scales with ``mag``:sup:`2`. Presumably this
limitation can be overcome by implementing the steps described in the
*Convolution* section of the *Appendix* of 
`Jericho and Kreuzer 2010 <http://link.springer.com/chapter/10.1007%2F978-3-642-15813-1_1>`_. 

