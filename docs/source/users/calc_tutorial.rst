.. _calc_tutorial:

************************
Scattering Calculations.
************************

In rough pseudo code, computing a hologram looks like this  ::
  holo = Theory.calc_holo(scatterer, target)

You can compute other scattering quantities by calling appropriate
calc functions (see
:class:`holopy.scattering.theory.scatteringtheory`)

To actually compute scattering, you need to do three things:

1. Define a scatterer using a :mod:`holopy.scattering.scatterer` object

2. Specify the output Data format using a :mod:`holopy.core.data.DataTarget` object

3. Use a calc function from a :mod:`holopy.scattering.theory` object
   to compute scattered quantities



Define a Scatterer
==================

Scatterer objects are used to describe the geometry of objects that
scatter light.  They contain information about the position, size, and
index of refraction of the scatterers.  You define a scatterer like
this ::

  from holopy.scattering.scatterer import Sphere
  sphere = Sphere(n = 1.59, r = .5, center = (5, 5, 5))

This scatterer is one we commonly use in our lab, a 1 micron
polystyrene sphere.  Its index of refraction is 1.59, and here we have
placed it 5 microns away from our camera.

You can describe other objects or collections of objects with other
scatterers in :mod:`holopy.scattering.scatterer`, go on take a look,
we will still be here when you get back.

Desired data
============

Holopy needs to know where in space you want to compute scattered quantities.  This is done be specifying a DataTarget object ::
  
  from holopy.core import ImageTarget, Optics
  target = ImageTarget(shape = 100, spacing = .1, optics = Optics(wavelen = .66, index = 1.33))

Here we have chosen to use an :class:`holopy.core.data.ImageTarget` object because we want a square hologram like we would record on a camera.  If you want to calculate some other kind of data, you would use a raw :class:`holopy.core.data.DataTarget` object or a different subclass.  

This :class:`ImageTarget` is just like a normal :class:`holopy.core.data.Data` object in that it specifies coordinates of a measurement and any other relevant experimental conditions.  The distinction between a DataTarget and Data is that DataTarget's do not contain all of the data values, only where they would be measured.  The astute reader will notice that a Data object is a valid DataTarget, you can provide a Data object anywhere a DataTarget is needed, essentially telling holopy "calculate something like this Data."



Scattering Theory
=================

Holopy contains a number of scattering theories that are useful for different kinds of scatterers.  Go take a look at :mod:`holopy.scattering.theory`.

Lets focus on the simplest of them, the :class:`holopy.scattering.theory.mie.Mie` theory.  If you have the scatterer and target from above, you can compute a hologram with ::

  from holopy.scattering.theory import Mie
  holo = Mie.calc_holo(sphere, target)

Similar functions exist to calculate all kinds of interesting scattered quantities and they are called the same way (except for calc_cross_sections which only needs an Optics object not a full fledged target.  

.. note::
   All units in the above code sample are in microns. You are free to work in any self consistent set of units, for example you could work in pixels by doing: ::
	
     sphere = Sphere(center = (50, 50, 50), n = 1.59, r = 5)
	 target = ImageTarget(spacing = 1, shape = 100, optics = Optics(wavelen = 6.6, index=1.33))

   In a similar vein you could work in meters, inches, furlongs, or cubits. 
	 
Examples
========

Now lets put this all together and see how you would compute scattering from some other objects.  



Cluster of Spheres
------------------

Calculating a hologram from a cluster of spheres is done in a very
similar manner ::

    from holopy.scattering.scatterer import SphereCluster
    s1 = Sphere(center=(5, 5, 5), n = 1.59, r = 0.5)
    s2 = Sphere(center=(4, 4, 5), n = 1.59, r = 0.5)
    cluster = SphereCluster([s1, s2])
    holo = Mie.calc_holo(cluster, target)

This will do the calculation with superposition of Mie solutions, if
you want to solve the actual multisphere problem for higher accuracy
you would instead use ::

    from holopy.scattering.theory import Multisphere
    holo = Multisphere.calc_holo(cluster, target)

Adding more spheres to the cluster is as simple as defining more
sphere objects and passing a longer list of spheres to the
:class:`holopy.scattering.scatterer.SphereCluster` constructor.

Coated Spheres
--------------

Coated (or layered) spheres can use the same Mie theory as normal
spheres. Coated spheres differ from normal spheres only in taking a
list of indexes and radii corresponding to the layers ::

    from holopy.scattering.scatterer import CoatedSphere
    cs = CoatedSphere(center=(5, 5, 5), n = (1.59, 1.42), r = (0.3, 0.6))
    holo = Mie.calc_holo(cs, target)

.. note::
	The multisphere theory does not as yet work with coated spheres.


Advanced Calculations
=====================

Static Light Scattering
-----------------------
In a static light scattering measurement you record scattered intensity at a number of angles.  In this kind of experiment you are usually not interested in the exact distance from the particles, and so instead work with scattering matricies ::

  from holopy.core import DataTarget, Angles, Optics
  from holopy.scattering.scatterer import Sphere
  from holopy.scattering.theory import Mie
  target = DataTarget(positions = Angles(theta = np.linspace(0, np.pi)),
                      optics = Optics(wavelen=.66, index = 1.33))
  sphere = Sphere(r = .5, n = 1.59)

  matr = Mie.calc_scat_matr(sphere, target)
  
If you ommit the center specification on a scatterer, holopy will assume you want farfield values.  


Hologram With Beam Tilt or Nonstandard Polarization
---------------------------------------------------

Tilted incident illumination can be specified in the Optics metadata ::
  
   optics = Optics(wavelen= .66, index=1.33, illum_vector = (0, .2, 1), pol = [.3, .4])

The default illum_vector is (0, 0, 1) indicating light incident along the z axis (propagating in the -z direction).  Polarization and illumination vectors are automatically normalized, so provide them however is convenient.

Non-Square Detectors and/or Pixels
----------------------------------

The holograms above make use of several default assumptions.  When you make an ImageTarget like ::

  ImageTarget(shape = 100, spacing = .1...)

This is equivalent to ::

  DataTarget(positions=Grid(shape=(100, 100), spacing = (.1, .1)...)
  

So if you wanted a rectangular detector with rectangular pixels, you could specify it as ::

   DataTarget((100, 200), spacing = (.2, .1))

The most general way to specify detectors would be ::

  target = DataTarget(positions = Pixels([Pixel(coordinates, normal = (n_x, n_y, n_z),
                                                area = Rectangle(.1, .1)), ...]), ...)

This kind of detector is not implemented yet, but we leave it here to show how general this specification format is.  If you need steps towards this general detector, let us know.  

Non Default Theory Parameters
-----------------------------

Some theories like :class:`holopy.scattering.theory.multisphere.Multisphere` have some adjustable parameters.  In general our defaults will work fine, but you can adjust them if you want.  You do this by instantiating the theory and calling calc functions on that specific object.  ::

  from holopy.scattering.theory import Multisphere
  multi = Multisphere(niter = 100)
  holo = multi.calc_holo(....)