.. _dda_tutorial:

Scattering from Arbitrary Structures with DDA
=============================================

The discrete dipole approximation (DDA) lets us calculate scattering
from any arbitrary object by representing it as a closely packed array
of point dipoles. In HoloPy you can make use of the DDA by specifying
a general :class:`.Scatterer` with an indicator function (or set of
functions for a composite scatterer containing multiple media).

HoloPy uses `ADDA <http://code.google.com/p/a-dda/>`_ to do the actual
DDA calculations, so you will need to install ADDA and be able to run::

  adda

at a terminal for HoloPy DDA calculations to succeed. To install ADDA,
first download or clone the `code <https://github.com/adda-team/adda>`_
from GitHub. In a terminal window, go to the directory ’adda/src’ 
and compile using one of three options::

 make seq

or::

 make 

or::

 make OpenCL 

``make seq`` will not take advantage of any parallel processing of the cores
on your computer. ``make`` uses mpi for parallel processing. ``make OpenCL`` uses 
OpenCL for parallel processing. If the make does not work due to missing packages,
you will have to download the specified packages and install them.

Next, you must modify your path in your .bashrc or /bash_profile (for mac). Add the 
line::

  export PATH=$PATH:userpath/adda/src/seq

or::

  export PATH=$PATH:userpath/adda/src/mpi

or::

  export PATH=$PATH:userpath/adda/src/OpenCL

where you should use the path that matches the make you chose above.

A lot of the code associated with DDA is fairly new so be careful;
there are probably bugs. If you find any, please `report
<https://github.com/manoharan-lab/holopy/issues>`_ them.

Defining the geometry of the scatterer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To calculate the scattering pattern for an arbitrary object, you first
need an indicator function which outputs 'True' if a test coordinate
lies within your scatterer, and 'False' if it doesn't. The indicator function
is an argument of the constructor of your scatterer. 

For example, if you wanted to define a dumbbell consisting of the union
of two overlapping spheres you could do so like this::

  import holopy as hp    
  from holopy.scattering import Scatterer, Sphere, calc_holo
  import numpy as np
  s1 = Sphere(r = .5, center = (0, -.4, 0))
  s2 = Sphere(r = .5, center = (0, .4, 0))
  detector = hp.detector_grid(100, .1)
  dumbbell = Scatterer(lambda point: np.logical_or(s1.contains(point), s2.contains(point)),
                       1.59, (5, 5, 5))
  holo = calc_holo(detector, dumbbell, medium_index=1.33, illum_wavelen=.66, illum_polarization=(1, 0))

Here we take advantage of the fact that Spheres can tell us if a point
lies inside them. We use ``s1`` and ``s2`` as purely geometrical
constructs, so we do not give them indices of refraction, instead
specifying n when defining ``dumbbell``.

HoloPy contains convenient wrappers for many built-in ADDA constructions. 
The dumbbell defined explicitly above could also have been defined with the HoloPy :class:`.Bisphere` class instead. 
Similar classes exist to define an :class:`.Ellipsoid`, :class:`.Cylinder`, or :class:`.Capsule`.

Mutiple Materials: A Janus Sphere
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also provide a set of indicators and indices to define a scatterer
containing multiple materials. As an example, lets look at a `janus
sphere <http://en.wikipedia.org/wiki/Janus_particles>`_ consisting of
a plastic sphere with a high index coating on the top half::

  from holopy.scattering.scatterer import Indicators
  import numpy as np
  s1 = Sphere(r = .5, center = (0, 0, 0))
  s2 = Sphere(r = .51, center = (0, 0, 0))
  def cap(point):
      return(np.logical_and(np.logical_and(point[...,2] > 0, s2.contains(point)),
             np.logical_not(s1.contains(point))))
  indicators = Indicators([s1.contains, cap],
                          [[-.51, .51], [-.51, .51], [-.51, .51]])
  janus = Scatterer(indicators, (1.34, 2.0), (5, 5, 5))
  holo = calc_holo(detector, janus, medium_index=1.33, illum_wavelen=.66, illum_polarization=(1, 0))

We had to manually set up the bounds of the indicator functions here
because the automatic bounds determination routine gets confused by
the cap that does not contain the origin.

We also provide a :class:`.JanusSphere` scatterer which is very
similar to the scatterer defined above, but can also take a rotation
angle to specify other orientations::

  from holopy.scattering import JanusSphere
  janus = JanusSphere(n = [1.34, 2.0], r = [.5, .51], rotation = (-np.pi/2, 0),
                    center = (5, 5, 5))
