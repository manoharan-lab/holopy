.. _dda_tutorial:

*********************************************
Scattering from Arbitrary Structures with DDA
*********************************************

The discrete dipole approximation (DDA) lets us calculate scattering
from any arbitrary object by representing it as a closely packed array
of point dipoles. In HoloPy you can make use of the DDA by specifying
a general :class:`.Scatterer` with an indicator function (or set of
functions for a composite scatterer containing multiple media).

HoloPy uses `ADDA <http://code.google.com/p/a-dda/>`_ to do the actual
DDA calculations, so you will need to install ADDA and be able to run::

  adda

at a terminal for HoloPy DDA calculations to succeed.

A lot of the code associated with DDA is fairly new so be careful;
there are probably bugs. If you find any, please `report
<https://bugs.launchpad.net/holopy/+filebug>`_ them.

Defining the geometry of the scatterer
======================================

To calculate the scattering pattern for an arbitrary object, you first
need an indicator function which outputs 'True' if a test coordinate
lies within your scatterer, and 'False' if it doesn't.

For example, if you wanted to define a dumbbell consisting of the union
of two overlapping spheres you could do so like this::

  from holopy.core import Optics, ImageSchema
  from holopy.scattering.scatterer import Scatterer, Sphere
  from holopy.scattering.theory import DDA
  import numpy as np
  s1 = Sphere(r = .5, center = (0, -.4, 0))
  s2 = Sphere(r = .5, center = (0, .4, 0))
  schema = ImageSchema(100, .1, Optics(.66, 1.33, (1, 0)))
  dumbbell = Scatterer(lambda point: np.logical_or(s1.contains(point), s2.contains(point)),
                       1.59, (5, 5, 5))
  holo = DDA.calc_holo(dumbbell, schema)

Here we take advantage of the fact that Spheres can tell us if a point
lies inside them. We use ``s1`` and ``s2`` as purely geometrical
constructs, so we do not give them indicies of refraction, instead
specifying n when defining ``dumbell``.

Mutiple Materials: A Janus Sphere
=================================

You can also provide a set of indicators and indices to define a scatterer
containing multiple materials. As an example, lets look at a `janus
sphere <http://en.wikipedia.org/wiki/Janus_particles>`_ consisting of
a plastic sphere with a high index coating on the top half::

  from holopy.core import Optics, ImageSchema
  from holopy.scattering.scatterer import Scatterer, Sphere
  from holopy.scattering.scatterer import Indicators
  from holopy.scattering.theory import DDA
  import numpy as np
  s1 = Sphere(r = .5, center = (0, 0, 0))
  s2 = Sphere(r = .51, center = (0, 0, 0))
  schema = ImageSchema(100, .1, Optics(.66, 1.33, (1, 0)))
  def cap(point):
      return(np.logical_and(np.logical_and(point[...,2] > 0, s2.contains(point)),
             np.logical_not(s1.contains(point))))
  indicators = Indicators([s1.contains, cap],
                          [[-.51, .51], [-.51, .51], [-.51, .51]])
  janus = Scatterer(indicators, (1.34, 2.0), (5, 5, 5))
  holo = DDA.calc_holo(janus, schema)

We had to manually set up the bounds of the indicator functions here
because the automatic bounds determination routine gets confused by
the cap that does not contain the origin.

We also provide a :class:`.JanusSphere` scatterer which is very
similar to the scatterer defined above, but can also take a rotation
angle to specify other orientations::

  from holopy.scattering.scatterer import JanusSphere
  janus = JanusSphere(n = [1.34, 2.0], r = [.5, .51], rotation = (-np.pi/2, 0),
                    center = (5, 5, 5))
