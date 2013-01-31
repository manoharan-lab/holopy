.. _recon_tutorial:

*******************************************
Reconstructing Data (Numerical Propagation)
*******************************************

Holograms are typically reconstructed optically by shining light back
through them.  This corresponds mathematically to propagating the
field stored in the hologram to some different plane.  HoloPy
generalizes this concept and allows you to numerically propagate any
hologram (or electric field) to another position in space.

Reconstructions generally consist of the following steps

1. :ref:`Load <loading>` or :ref:`calculate <calc_tutorial>` a
   Hologram :class:`.Image`.

2. :ref:`Propagate <propagating>` the image to the desired distance or
   set of distances.

3. :ref:`Visualize <visualize_recon>` the reconstruction. 

Here is an example:

.. plot:: pyplots/basic_recon.py
   :include-source:

We'll examine each section of code in turn. The first block::

  import numpy as np
  import holopy as hp
  from holopy.propagation import propagate
  from holopy.core.tests.common import get_example_data
  from holopy.core import load

loads the relevant modules from HoloPy and NumPy. 

Loading Data
------------

We use::

  holo = get_example_data('image0001.yaml')

to load one of the example images shipped with HoloPy. It already
contains all needed metadata. For working with your data you will
instead want to use::
  
  holo = load('your_image.tif', spacing = 1e-7, 
              optics = Optics(wavelen = 660e-9, index = 1.33, 
                              polarization = (1,0)))

The spacing and wavelength can be specified in whatever units you
prefer, as long as you are consistent (see :ref:`units`). Holopy load
any image formats `PIL <http://www.pythonware.com/products/pil/>`_ can
load (which is most image formats).

.. _propagating:

Propagating
-----------

The actual propagation is accomplished with :func:`.propagate`::

  rec_vol = propagate(holo, linspace(4e-6, 10e-6, 7))

Here we have used numpy's linspace to get a set of distances to
propagate to. You can also propagate to a single distance, or to set
of distances obtained in some other fashion.

.. _visualize_recon:

Visualizing Reconstructions
---------------------------

You can then visualize the reconstruction with :func:`.show`::
  
  hp.show(rec_vol)

You can step through volume slices with the left and right arrow keys
(Don't use the down arrow key; it will mess up the stepping due to a
peculiarity of Matplotlib. If this happens, close your plot window and
show it again. Sorry.). 

Reconstructions are actually comprised of complex numbers. hp.show
defaults to showing you the amplitude of the image. You can get
different, and sometimes better, contrast by viewing the phase angle or
imaginary part of the reconstruction::

  hp.show(rec_vol.imag)
  hp.show(np.angle(rec_vol))

These phase sensitive visualizations will change contrast as you step
through because you hit different places in the phase period. Such a
reconstruction will work better if you use steps that are an integer
number of wavelengths in medium::

  from numpy import arange
  rec_vol = propagate(holo, linspace(4e-6, 10e-6, holo.optics.med_wavelen))
  hp.show(rec_vol.imag)
