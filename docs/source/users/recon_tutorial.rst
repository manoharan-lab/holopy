.. _recon_tutorial:

Reconstructing Data (Numerical Propagation)
===========================================

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
    from holopy.core.io import get_example_data_path, load_average

loads the relevant modules from HoloPy and NumPy. The second block:

..  testcode::
    
    imagepath = get_example_data_path('image01.jpg')
    raw_holo = hp.load_image(imagepath, spacing = 0.0851, illum_wavelen = 0.66, medium_index = 1.33)
    bgpath = get_example_data_path(['bg01.jpg','bg02.jpg','bg03.jpg'])
    bg = load_average(bgpath, refimg = raw_holo)
    holo = hp.core.process.bg_correct(raw_holo, bg)

reads in a hologram and divides it by a corresponding background image.
If this is unfamiliar to you, please review the :ref:`load_tutorial` tutorial.

Next, we use numpy's linspace to define a set of distances at 2-micron intervals to 
propagate our image to. You can also propagate to a single distance,
or to a set of distances obtained in some other fashion. 
The actual propagation is accomplished with :func:`.propagate`:

..  testcode::

    zstack = np.linspace(1, 15, 8)
    rec_vol = hp.propagate(holo, zstack)

..  testcode::
    :hide:
    
    print(rec_vol.values[0,0,0])

..  testoutput::
    :hide:

    (0.911671338697-0.0816366824816j)


Here, HoloPy has projected the image through space using the metadata that we 
specified when loading the image. If we forgot to load optical metadata with the image,
we can explicitly indicate the parameters for propagation to obtain an identical object:

..  testcode::

    rec_vol = hp.propagate(holo, zstack, illum_wavelen = 0.660, medium_index = 1.33)


Visualizing Reconstructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can display the reconstruction with :func:`.show`::
  
  hp.show(rec_vol)

Pressing the left and right arrow keys steps through volumes slices - 
propagation to different z-planes. 
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
    
  med_wavelen = holo.illum_wavelen / holo.medium_index
  rec_vol = hp.propagate(holo, zstack*med_wavelen)
  hp.show(rec_vol.imag)

..  testcode::
    :hide:

    print(rec_vol[0,0,0].imag.values)

..  testoutput::
    :hide:
    
    0.005048845807476341


Cascaded Free Space Propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HoloPy calculates reconstructions by performing a convolution of the hologram with
the reference illuminating electromagnetic wave over the distance to be propagated.
By default, HoloPy calculates a single transfer function to perform the convolution
over the specified distances. However, a better reconstruction can sometimes be
obtained by iteratively propagating the hologram over short distances. This 
cascaded free space propagation is particularly useful when the reconstructions have
fine features or when propagating over large distances. For further details, refer to 
`Kreis 2002 <http://opensample.info/frequency-analysis-of-digital-holography-with-reconstruction-by-convolution>.

To impolement cascaded free space propagation in HoloPy, simply pass a ``cfsp`` variable
into :func:`.propagate` indicating how many times the hologram should be iteratively
propagated. For example, to propagate in three steps over each distance, we write:

..  testcode::
    
    rec_vol = hp.propagate(holo, zstack, cfsp = 3)

..  testcode::
    :hide:

    print(rec_vol.values[0,0,0])

..  testoutput::
    :hide:

    (0.911671338697-0.0816366824816j)
