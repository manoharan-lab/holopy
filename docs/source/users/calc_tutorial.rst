.. _calc_tutorial:

************************
Scattering Calculations
************************

Optical physicists and astronomers have worked out how to compute the
light scattering off of many kinds of objects.  HoloPy provides an
easy interface for computing many types of scattering from microscopic
objects.  These include scattered electric fields, scattered intensity
(what you would record on a normal camera), holograms, and scattering
matrices and cross sections 


Scattering calculations generally consist of three steps:

1. Define a scatterer using a :mod:`~holopy.scattering.scatterer` object

2. Specify a wavelength and where to compute the values using a :mod:`~holopy.core.marray.Schema` object

3. Use a calc function from a :mod:`~holopy.scattering.theory` object
   to compute scattered quantities

Example
==================
 ::

  from holopy.scattering.scatterer import Sphere
  from holopy.core import ImageSchema, Optics
  from holopy.scattering.theory import Mie

  sphere1 = Sphere(center=(3, 2, 5), n = 1.59, r = 0.5) #choice of units: microns
  schema = ImageSchema(spacing = .1, shape = 100, 
    optics = Optics(wavelen = .660, index=1.33)) #detector: 100x100 pixel camera, red laser
  holo = Mie.calc_holo(sphere1, schema)


Defining a Scatterer
==================

Scatterer objects are used to describe the geometry and optical properties of objects that
scatter light.  They contain information about the position, size, shape, and
index of refraction of the scatterers.  You define a scatterer like
this: ::

  from holopy.scattering.scatterer import Sphere
  sphere = Sphere(n = 1.59+.0001j, r = .5, center = (3, 2, 5))

This scatterer is one we commonly use in our lab: a 1 micron diameter
polystyrene sphere.  Its index of refraction is 1.59 with a very small
complex part, and here we have placed it 5 microns away from our
imaging plane.

.. note::

   We include a small imaginary component in the index of refraction
   because we believe it may avoid some potential problems in the
   scattering calculations.  In any event an imaginary index of
   :math:`10^{-4}` should have negligible impact on the result.
  
You can describe other objects or collections of objects with other
scatterers in :mod:`holopy.scattering.scatterer`. Go on take a look
at the code or see the More Examples section below,
we will still be here when you get back.

Defining a Schema
===============

Schema objects tell HoloPy what its calculated results should look
like.  They are intentionally very similar to the objects you get back
when you load data.  To compute a simulated hologram you can define a
Schema like this::
  
  from holopy.core import ImageSchema, Optics
  schema = ImageSchema(shape = 100, spacing = .1, optics = Optics(wavelen = .660, index = 1.33))

Here we have chosen to use a :class:`.ImageSchema` Schema because we
want a square hologram like we would record on a camera.  If you want
to calculate some other kind of data, you could use a raw
:class:`.Schema` object or a different subclass.

This :class:`.ImageSchema` is just like a normal :class:`.Image`
object in that it specifies coordinates of the camera pixels and the
relevant optical information.  The distinction between a
:class:`.ImageSchema` an :class:`.Image` is that
:class:`.ImageSchema`'s do not contain all of the data values, only
where they would be measured.  The astute reader might notice that a
:class:`.Image` contains all the information that an :class:`.Image`
does.  Because of this, HoloPy lets you provide an :class:`.Image` or
object anywhere an :class:`.ImageSchema` is needed, essentially
telling HoloPy "calculate something like this data."



Scattering Theory
=================

HoloPy contains a number of scattering theories that are useful for
different kinds of scatterers.  Take a look at them in
:mod:`holopy.scattering.theory`.

Lets focus on the simplest of them, the
:class:`~holopy.scattering.theory.mie.Mie` theory.  If you have the
scatterer and schema from above, you can compute a hologram with ::

  from holopy.scattering.theory import Mie
  holo = Mie.calc_holo(sphere, schema)

Similar functions exist to calculate all kinds of interesting
scattered quantities and they are called the same way (except for
calc_cross_sections which only an Optics object and not a full
:class:`.Schema`).

.. note::
   All units in the above code sample are in microns. You are free to work in any self consistent set of units, for example you could work in pixels by doing: ::
	
     sphere = Sphere(center = (50, 50, 50), n = 1.59, r = 5)
     schema = ImageSchema(spacing = 1, shape = 100, optics = Optics(wavelen = 6.58, index=1.33))

   In a similar vein you could work in meters, inches, furlongs, smoots, or cubits. 
	 
More Examples
========

Now let's take this a step further and see how you can compute scattering from 
objects more complex than a single sphere.  



Cluster of Spheres
------------------

Calculating a hologram from a cluster of spheres is done in a very
similar manner ::

  from holopy.scattering.theory import Mie
  from holopy.scattering.scatterer import Sphere, Spheres
  schema = ImageSchema(spacing = 1, shape = 100, optics = Optics(wavelen = 6.58, index=1.33))
  s1 = Sphere(center=(5, 5, 5), n = 1.59, r = 0.5)
  s2 = Sphere(center=(4, 4, 5), n = 1.59, r = 0.5)
  cluster = Spheres([s1, s2])
  holo = Mie.calc_holo(cluster, schema)

This will do the calculation with superposition of Mie solutions, if
you want to solve the actual multisphere problem for higher accuracy
you would instead use ::

    from holopy.scattering.theory import Multisphere
    holo = Multisphere.calc_holo(cluster, schema)

Adding more spheres to the cluster is as simple as defining more
sphere objects and passing a longer list of spheres to the
:class:`.Spheres` constructor.

Coated Spheres
--------------

Coated (or layered) spheres can use the same Mie theory as normal
spheres. Coated spheres differ from normal spheres only in taking a
list of indexes and radii corresponding to the layers. The indices are
given in order starting from the core. ::

  from holopy.scattering.scatterer import CoatedSphere
  from holopy.scattering.theory import Mie
  schema = ImageSchema(spacing = 1, shape = 100, optics = Optics(wavelen = 6.58, index=1.33))
  cs = CoatedSphere(center=(5, 5, 5), n = (1.59, 1.42), r = (0.3, 0.6))
  holo = Mie.calc_holo(cs, schema)

.. note::
	The multisphere theory does not as yet work with coated spheres.


Advanced Calculations
=====================

Scattering Matrices
-------------------
In a static light scattering measurement you record scattered intensity at a number of angles.  In this kind of experiment you are usually not interested in the exact distance from the particles, and so instead work with scattering matrices ::

  from holopy.core import Schema, Angles, Optics
  from holopy.scattering.scatterer import Sphere
  from holopy.scattering.theory import Mie
  schema = Schema(positions = Angles(theta = np.linspace(0, np.pi, 100)),
                  optics = Optics(wavelen=.660, index = 1.33))
  sphere = Sphere(r = .5, n = 1.59)

  matr = Mie.calc_scat_matrix(sphere, schema)
  # It is typical to look at scattering matrices on a semilog plot,
  # you can make one with this code
  figure()
  semilogy(np.linspace(0, np.pi, 100), abs(matr[:,0,0])**2)
  semilogy(np.linspace(0, np.pi, 100), abs(matr[:,1,1])**2)
  
Here we omit specifying center specification on a scatterer.  This is
only valid when you want a farfield quantity like we do here.


Hologram With Beam Tilt or Nonstandard Polarization
---------------------------------------------------

.. note::

   This description is provided as a preview, HoloPy cannot actually
   do beam tilts yet, and we have not tested varying polarization

Tilted incident illumination can be specified in the Optics metadata ::
  
   optics = Optics(wavelen= .66, index=1.33, illum_vector = (0, .2, 1), polarization = [.3, .4])

The default illum_vector is (0, 0, 1) indicating light incident along the z axis (propagating in the -z direction).  Polarization and illumination vectors are automatically normalized, so provide them however is convenient.

Non-Square Detectors and/or Pixels
----------------------------------

The holograms above make use of several default assumptions.  When you make an ImageSchema like ::

  ImageSchema(shape = 100, spacing = .1...)

This is equivalent to ::

  ImageSchema(shape=(100, 100), spacing = (.1, .1)...)
  

So if you wanted a rectangular detector with rectangular pixels, you could specify it as ::

   ImageSchema(shape = (100, 200), spacing = (.2, .1))

The most general way to specify detectors would be ::

  schema = Schema(positions = Pixels([Pixel(coordinates, normal = (n_x, n_y, n_z),
                                            area = Rectangle(.1, .1)), ...]), ...)

.. note::
											
   This kind of detector is not implemented yet, but we leave it here
   to show how general this specification format is.  If you need
   these kinds of features, let us know.

Non Default Theory Parameters
-----------------------------

Some theories like :class:`~holopy.scattering.theory.multisphere.Multisphere` have some adjustable parameters.  In general our defaults will work fine, but you can adjust them if you want.  You do this by instantiating the theory and calling calc functions on that specific object.  ::

  from holopy.scattering.theory import Multisphere
  s1 = Sphere(center=(5, 5, 5), n = 1.59, r = 0.5)
  s2 = Sphere(center=(4, 4, 5), n = 1.59, r = 0.5)
  cluster = Spheres([s1, s2])
  schema = ImageSchema(shape = 100, spacing = .1, optics = Optics(wavelen = .660, index = 1.33))
  multi = Multisphere(niter = 100)
  holo = multi.calc_holo(cluster, schema)
