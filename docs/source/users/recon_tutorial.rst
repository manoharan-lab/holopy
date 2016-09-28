.. _recon_tutorial:

*******************************************
Reconstructing Data (Numerical Propagation)
*******************************************

Holograms are typically reconstructed optically by shining light back
through them.  This corresponds mathematically to propagating the
field stored in the hologram to some different plane.  HoloPy
generalizes this concept and allows you to numerically propagate any
hologram (or electric field) to another position in space.

Here is an example:

.. plot:: pyplots/basic_recon.py
   :include-source:

We'll examine each section of code in turn. The first block:

..  testcode::

    import numpy as np
    import holopy as hp
    from holopy import propagate
    from holopy.core.io import get_example_data_path

loads the relevant modules from HoloPy and NumPy. The second block:

..  testcode::
    
    imagepath = get_example_data_path('image01.jpg')
    raw_holo = hp.load_image(imagepath, spacing = 0.0851, wavelen = 0.66, index = 1.33)
    bgpath = get_example_data_path('bg01.jpg')
    bg = hp.load_image(bgpath)
    holo = raw_holo / bg

reads in a hologram and divides it by a corresponding background image.
If this is unfamiliar to you, please review the :ref:`load_tutorial` tutorial.

Next, we use numpy's linspace to define a set of distances to 
propagate to at 2-micron intervals. You can also propagate to a single distance,
or to a set of distances obtained in some other fashion. 
The actual propagation is accomplished with :func:`.propagate`:

..  testcode::

    zstack = np.linspace(1, 15, 8)
    rec_vol = propagate(holo, zstack)

..  testcode::
    :hide:
    
    print(rec_vol[0,0,0])

..  testoutput::
    :hide:

    (0.834984178898-0.0856125790499j)

Here, HoloPy has projected the image through space using the metadata that we 
specified when loading the image. If we forgot to load optical metadata with the image,
we can explicitly indicate the parameters for propagation to obtain an identical object:

..  testcode::

    rec_vol = propagate(holo, zstack, wavelen = 0.660, index = 1.33)


You can then visualize the reconstruction with :func:`.show`::
  
  hp.show(rec_vol)

You can step through volume slices with the left and right arrow keys
(Don't use the down arrow key; it will mess up the stepping due to a
peculiarity of Matplotlib. If this happens, close your plot window and
show it again. Sorry.). 

Reconstructions are actually comprised of complex numbers. :func:`.show`
defaults to showing you the amplitude of the image. You can get
different, and sometimes better, contrast by viewing the phase angle or
imaginary part of the reconstruction::

  hp.show(rec_vol.imag)
  hp.show(np.angle(rec_vol))

These phase sensitive visualizations will change contrast as you step
through because you hit different places in the phase period. Such a
reconstruction will work better if you use steps that are an integer
number of wavelengths in medium:

..  testcode::
    
  med_wavelen = holo.wavelen / holo.index
  rec_vol = propagate(holo, zstack*med_wavelen)
  hp.show(rec_vol.imag)

..  testcode::
    :hide:

    print(rec_vol[0,0,0].imag)

..  testoutput::
    :hide:
    
    -0.00284432855731
