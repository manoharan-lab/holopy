.. _recon_tutorial:

Reconstructing Data (Numerical Propagation)
===========================================

A hologram contains information about the electric field amplitude and phase at the detector plane.
Shining light back through a hologram allows reconstruction of the electric field at points upstream of the detector plane.
HoloPy performs this function mathematically by numerically propagating a hologram (or electric field) to another position in space.

This allows you to reconstruct 3D sample volumes from 2D images. The light source is assumed to be collimated here, but HoloPy is also capable of :ref:`ps_recon_tutorial`.

Example Reconstruction
~~~~~~~~~~~~~~~~~~~~~~

.. plot:: pyplots/basic_recon.py
   :include-source:

We'll examine each section of code in turn. The first block:

..  testcode::

    import numpy as np
    import holopy as hp
    from holopy.core.io import get_example_data_path, load_average
    from holopy.core.process import bg_correct

loads the relevant modules from HoloPy and NumPy. The second block:

..  testcode::

    imagepath = get_example_data_path('image01.jpg')
    raw_holo = hp.load_image(imagepath, spacing = 0.0851, medium_index = 1.33, illum_wavelen = 0.66)
    bgpath = get_example_data_path(['bg01.jpg','bg02.jpg','bg03.jpg'])
    bg = load_average(bgpath, refimg = raw_holo)
    holo = bg_correct(raw_holo, bg)

reads in a hologram and divides it by a corresponding background image.
If this is unfamiliar to you, please review the :ref:`load_tutorial` tutorial.

Next, we use numpy's linspace to define a set of distances between the image plane and the reconstruction plane at 2-micron intervals to
propagate our image to. You can also propagate to a single distance
or to a set of distances obtained in some other fashion.
The actual propagation is accomplished with :func:`.propagate`:

..  testcode::

    zstack = np.linspace(0, 20, 11)
    rec_vol = hp.propagate(holo, zstack)

..  testcode::
    :hide:

    print(rec_vol.values[0,0,0].real)
    print(rec_vol.values[0,0,0].imag)

..  testoutput::
    :hide:

    0.9196428571...
    0.0


Here, HoloPy has projected the hologram image through space to each of the distances contained in ``zstack`` by using the metadata that we
specified when loading the image. If we forgot to load optical metadata with the image,
we can explicitly indicate the parameters for propagation to obtain an identical object:

..  testcode::

    rec_vol = hp.propagate(holo, zstack, illum_wavelen = 0.660, medium_index = 1.33)


Visualizing Reconstructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can display the reconstruction with :func:`.show`::

  hp.show(rec_vol)

Pressing the left and right arrow keys steps through volumes slices -
propagation to different z-planes. If the left and right arrow keys don't
do anything, you might need to set your matplotlib backend. Refer to
:ref:`usage` for instructions.

Reconstructions are actually comprised of complex numbers. :func:`.show`
defaults to showing you the amplitude of the image. You can get different, and
sometimes better, contrast by viewing the phase angle or imaginary part of the
reconstruction::

  hp.show(rec_vol.imag)
  hp.show(np.angle(rec_vol))

These phase sensitive visualizations will change contrast as you step through
because you hit different places in the phase period. Such a reconstruction will
work better if you use steps that are an integer number of wavelengths in
medium:

..  testcode::

  med_wavelen = holo.illum_wavelen / holo.medium_index
  rec_vol = hp.propagate(holo, zstack*med_wavelen)
  hp.show(rec_vol.imag)

..  testcode::
    :hide:

    print(rec_vol[0,0,0].imag.values)

..  testoutput::
    :hide:

    0.0


Cascaded Free Space Propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HoloPy calculates reconstructions by performing a convolution of the hologram with
the reference wave over the distance to be propagated.
By default, HoloPy calculates a single transfer function to perform the convolution
over the specified distance. However, a better reconstruction can sometimes be
obtained by iteratively propagating the hologram over short distances. This
cascaded free space propagation is particularly useful when the reconstructions have
fine features or when propagating over large distances. For further details, refer to
`Kreis 2002 <http://dx.doi.org/10.1117/1.1489678>`_.

To implement cascaded free space propagation in HoloPy, pass a ``cfsp`` argument
into :func:`.propagate` indicating how many times the hologram should be iteratively
propagated. For example, to propagate in three steps over each distance, we write:

..  testcode::

    rec_vol = hp.propagate(holo, zstack, cfsp = 3)

..  testcode::
    :hide:

    print(rec_vol.values[0,0,0].real)
    print(rec_vol.values[0,0,0].imag)

..  testoutput::
    :hide:

    0.91964285714...
    0.0
