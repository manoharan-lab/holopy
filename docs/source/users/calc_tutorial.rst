.. _calc_tutorial:

************************
Scattering Calculations.
************************

In order to compute light scattering, you need to do three things:

1. Define a scatterer

2. Specify the output Data format

3. Choose a scattering theory

Scatterer Objects: :mod:`scatterpy.scatterer`
=============================================

Scatterer objects are used to describe the geometry of objects that
scatter light.  They contain information about the position,
size, and index of refraction of the scatterers.  The most commonly
useful scatterers are

:Sphere:

    Describes a single sphere

:SphereCluster:

    A collection of multiple spheres

:CoatedSphere:

    A sphere with multiple layers, each of a different index of refraction.

:VoxelatedScatterer:

   Represent any arbitrary object discretizing space and specifying the index of refraction of the scatterer at every voxel.

Desired data: DataTarget
========================

Holopy needs to know where in space you want to compute scattered quantities.  This is done be specifying a DataTarget object.  This object is just like a normal Data object in that it specifies coordinates of a measurement and any other relevant experimental conditions.

The distinction between a DataTarget and Data is that DataTarget's do not contain all of the data values, only where they would be measured.  The astute reader will notice that a Data object is a valid DataTarget, you can provide a Data object anywhere a DataTarget is needed, essentially telling holopy "calculate something like this Data."



Scattering Theories: :mod:`scatterpy.theory`
============================================

Holopy contains several different scattering theories for calculating scattered quantities:

:Mie:

    This model calculates the scattered field of a single spherical
    particle using the Lorenz-Mie solution. The calculated field is
    the exact solution for the scattering of a single spherical
    particle. The model can also be used to calculate the hologram of
    multiple particles by adding their scattered fields. Since this
    method does not account for multiple scattering, it yields only an
    approximation to the scattering for multiple particles.  The
    approximation is a good one if the particles are sufficiently
    separated.

    This model can also calculate the exact scattered field from a 
    spherically symmetric particle with an arbitrary number of layers
    with differing refractive indices, using Yang's recursive
    algorithm ([Yang2003]_).
    
:Multisphere: 

    This model calculates the scattered field of a collection of
    particles through a numerical method that accounts for multiple
    scattering and near-field effects (see [Fung2011]_, [Mackowski1996]_).  This
    approach is much more accurate than Mie superposition, but it is
    also more computationally intensive.  The Multisphere code can
    handle any number of spheres.

:DDA:

    This model (which requires `a-dda <http://code.google.com/p/a-dda/>`_ 
    to be installed, can calculate the field scattered from (in principle)
    an arbitrary scatterer. It uses the discrete dipole approximation, 
    a numerical method that represents arbitrary scatterers as an array
    of point dipoles and then self-consistently solves Maxwell's equations
    to determine the scattered field. In practice, this model can be 
    extremely computationally intensive, particularly if the size of the 
    scatterer is larger than the wavelength of light.

Each model provides a ``calc_holo`` function that will calculate a
hologram, a ``calc_field`` function that will calculate electric fields, and a ``calc_intensity`` function that will compute field intensity. Below we demonstrate how to calculate a hologram with each
of the available models.

The scattering theories are objects you can instantiate if you want to adjust details of how they do their calculations (ie iteration limits), but if you are happy with the defaults (which you mostly will be), you can call the calc methods without specifically instantiating a theory.  


Examples
========

In the following code snippets we calculate a hologram from a 1 micron
diameter spherical polystyrene particle. We assume the refractive
index of the particle is 1.58 (close to that of polystyrene) and the
particle is in water. The particle is 10 microns from the focal plane
and centered in the camera's field of view.  We assume a camera size
of 256-by-256 pixels where the pixels are squares with sides of 10
microns. Using a 100x magnifications optical train the pixel size in
the imaging plane will be 0.1 microns.  This information is specified as a DataTarget 

   .. sourcecode:: python
  
      import holopy
      target = holopy.DataTarget(positions = Grid(shape = (256, 256), spacing = (1e-7,1e-7)),
                                 optics = Optics(wavelen = 658e-9, index = 1.33))

This make use of Holopy's default assumption that you are working with a square grid of measurements, see later examples for how to specify other geometries.

In the following examples we will use units of meters and calculate
holograms created with 658 nm light.

Single sphere
-------------

Here we use the single sphere Mie calculations for computing the
hologram.  We create a :class:`scatterpy.scatterer.sphere.Sphere`
object to describe the sphere, use the Mie theory to do the calculation.  The arguments to the ``calc_holo`` function are specified in :meth:`holopy.model.mie.calc_holo`.  They include the size of the hologram we want to calculate (in pixels) and the properties and position of the particle ::

    from scattering.theory import Mie
    from scattering.scatterer import Sphere
    sphere = Sphere(center=(12.8e-6, 12.8e-6, 10e-6), n = 1.58, r = 0.5e-6)
    holo = Mie.calc_holo(sphere, target, scaling = 0.8)
	
.. note::
    All units in the above code sample are in meters. This will work
    out fine if the wavelength is also specified in meters. If you
    wanted to do everything in pixels you would instead define the
    sphere as ::

        sphere = Sphere(center(128, 128, 100), n = 1.58, r = 5)

    Provided that the wavelength of light was specified in units of
    pixels, this will calculate the same hologram as the previous
    example.


Cluster of Spheres
------------------

Calculating a hologram from a cluster of spheres is done in a very
similar manner ::

    from scatterpy.scatterer import SphereCluster
    s1 = Sphere(center=(12.8e-6, 12.8e-6, 10e-6), n = 1.58, r = 0.5e-6)
    s2 = Sphere(center=(12e-6, 11e-6, 10e-6), n = 1.58, r = 0.5e-6)
    cluster = SphereCluster([s1, s2])
    holo = Mie.calc_holo(cluster, target, alpha = 0.8)

This will do the calculation with superposition of Mie solutions, if
you want to solve the actual multisphere problem for higher accuracy
you would instead use ::

    from scatterpy.theory import Multisphere
    holo = Multisphere.calc_holo(cluster, target, alpha = 0.8)

Adding more spheres to the cluster is as simple as defining more
sphere objects and passing a longer list of spheres to the
:class:`scatterpy.scatterer.SphereCluster` constructor.

Coated Spheres
--------------

Coated (or layered) spheres can use the same Mie theory as normal
spheres. Coated spheres differ from normal spheres only in taking a
list of indexes and radii corresponding to the layers ::

    from scatterpy.scatterer import CoatedSphere
    cs = CoatedSphere(center=(12.8e-6, 12.8e-6, 10e-6), n = (1.58, 1.42), r = (0.3e-6, 0.6e-6))
    holo = Mie.calc_holo(cs, target, alpha = .8)

.. note::
	The multisphere theory does not as yet work with coated spheres.


Advanced Calculations
=====================

Farfield Scattering Matricies
-----------------------------

If you only want farfield scattering matricies, you don't need to give
holopy nearly as much information ::

  from holopy.core import DataTarget, Angles, Optics
  from holopy.scattering.scatterer import Sphere
  from holopy.scattering.theory import Mie
  target = DataTarget(positions = Angles(theta = np.linspace(0, np.pi)),
                      optics = Optics(wavelen=.66, index = 1.33))
  sphere = Sphere(r = .5, n = 1.59)

  matr = Mie.calc_scat_matr(sphere, target)

Static Light Scattering
-----------------------

In a static light scattering measurement you record scattered intensity at a number of angles, holopy can simulate such a measurement as ::

  from holopy.data import DataTarget, SpecifiedAngles
  target = DataTarget(SpecifiedAngles(linspace(-90, 90, 30), optics = Optics(wavelen = 659e-9, index = 1.33))
  s = Sphere(center=None, n = 1.58, r = .5e-6)
  scat = Mie.calc_intensity(s, target)

Specifying center as None (or simply omitting the argument) indicates that you want the computation done in the farfield.

Hologram With Beam Tilt or Nonstandard Polarization
---------------------------------------------------

Tilted incident illumination can be specified in the metadata ::
  
   target = DataTarget(256, wavelen= 659e-9, index=1.33, illum_vector = (0, .2, 1), pol = [.3, .4])

The default illum_vector is (0, 0, 1) indicating light incident along the z axis (propagating in the -z direction).  Polarization and illumination vectors are automatically normalized, so provide them however is convenient.

Non Detectors and/or Pixels
---------------------------

The holograms above make use of several default assumptions.  When the points argument of DataTarget is given as 256, it is assumed to mean ::
  RectangularGrid(256)

Which in turn interprets a single value as meaning a square detector ::
   RectangularGrid((256, 256))

In a similar manner, the single 0.1e-6 is interpreted to mean square pixels.  So if you wanted a rectangular detector with rectangular pixels, you could specify it as ::

   target = DataTarget((128, 256), pixel_scale = (.2, .1))

The most general way to specify detectors is as ::

  target = DataTarget([Pixel(x, y, z, normal = (n_x, n_y, n_z), area = Rectangle(.1, .1)), ...], ...)

