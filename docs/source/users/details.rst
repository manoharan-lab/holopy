Details
=======

.. _yaml_ref:

YAML files
----------------

HoloPy can save and load all of its objects from `YAML
<http://www.yaml.org/>`_ files.  These are designed to be both human and
computer readable and provide both our serialization format, and an
alternative method for specifying things like optical metadata.

You can save an optics object for future use::

  holopy.save('optics.yaml', optics)

optics.yaml will look something like this (yours will not contain the
explanatory comments, you can add any comments you want by proceeding
them with a '#" character)

.. sourcecode:: yaml
  
  !Optics
  wavelen: 0.660    # Wavelength of light (in vacuum) used in creating holograms
  index: 1.33       # Index of medium
  polarization: [1.0, 0.0]
  divergence: 0.0

You can also write this file by hand.  In either case you can make an
:class:`.Optics` object from the file ::

         optics = holopy.load('optics.yaml')
         holo = holopy.load('image.tif', pixel_size = .1,  optics = optics)

:func:`hp.load <holopy.core.io.io.load>` will also accept the filename
of an metadata yaml file as the argument for the optics parameter and
automatically load the yaml file. ::

  holo = holopy.load('image.tif', pixel_size = .1, optics='optics.yaml')

.. Note::
   
   Data objects are a special case for yaml output because they 
   will likely contain a large array of data.  They can still be 
   saved, but will generate  very large files that may not be 
   easily opened in a text editorlike other holopy yamls.

   For the curious advanced user, what we actually do is put a yaml
   header with optics and other information, and then encode the array
   of data as a .npy binary (as from np.save) all in the same file.  This
   keeps the whole object in a single file, but generates a file
   that is not quite as easy to work with as other yamls.





.. _more_scattering_ex:

More Scattering Examples
========================

Now let's take these calculations a step further and compute scattering from
objects more complex than a single sphere.

Ellipsoid
---------
You can calculate a hologram of an ellipsoid by using the discrete dipole
approximation (DDA). This requires first installing ADDA which you can find `here <http://code.google.com/p/a-dda/>`_.
. ::

  from holopy.scattering.theory import DDA
  from holopy.scattering.scatterer import Ellipsoid

  e = Ellipsoid(1.5, r = (.5, .1, .1), center = (1, -1, 10))
  schema = ImageSchema(100, .1, optics = Optics(wavelen=.66, index=1.33))
  h = DDA.calc_holo(e, schema)

Collection of Spheres
---------------------

Calculating a hologram from a collection of spheres is done in a very
similar manner ::

  from holopy.scattering.theory import Mie
  from holopy.scattering.scatterer import Sphere, Spheres
  schema = ImageSchema(spacing = 1, shape = 100, 
    optics = Optics(wavelen = 6.58, polarization = [1,0], index=1.33))
  s1 = Sphere(center=(5, 5, 5), n = 1.59, r = 0.5)
  s2 = Sphere(center=(4, 4, 5), n = 1.59, r = 0.5)
  collection = Spheres([s1, s2])
  holo = Mie.calc_holo(collection, schema)

This will do the calculation with superposition of Mie solutions, if
you want to solve the actual multisphere problem for higher accuracy
you would instead use ::


    from holopy.scattering.theory import Multisphere
    holo = Multisphere.calc_holo(cluster, schema)

Adding more spheres to the cluster is as simple as defining more
sphere objects and passing a longer list of spheres to the
:class:`.Spheres` constructor.


Non Default Theory Parameters
-----------------------------

Some theories like :class:`~holopy.scattering.theory.multisphere.Multisphere` have some adjustable parameters.  In general our defaults will work fine, but you can adjust them if you want.  You do this by instantiating the theory and calling calc functions on that specific object.  ::

  from holopy.core import ImageSchema, Optics
  from holopy.scattering.scatterer import Sphere, Spheres
  from holopy.scattering.theory import Multisphere
  s1 = Sphere(center=(5, 5, 5), n = 1.59, r = 0.5)
  s2 = Sphere(center=(4, 4, 5), n = 1.59, r = 0.5)
  cluster = Spheres([s1, s2])
  schema = ImageSchema(shape = 100, spacing = .1, 
    optics = Optics(wavelen = .660, polarization = [0,1], index = 1.33))
  multi = Multisphere(niter = 100)
  holo = multi.calc_holo(cluster, schema)

Coated Spheres
--------------

Coated (or multilayered) spheres can use the same Mie theory as simple
spheres. Constructing a coated spheres differs only in specifying a
list of indexes and radii corresponding to the layers. The indices are
given in order starting from the core. ::

  from holopy.scattering.scatterer import Sphere
  from holopy.scattering.theory import Mie
  schema = ImageSchema(spacing = 1, shape = 100, 
    optics = Optics(wavelen = 6.58, polarization = [1,0], index=1.33))
  cs = Sphere(center=(25, 50, 50), n = (1.59, 1.42), r = (0.3, 0.6))
  holo = Mie.calc_holo(cs, schema)
  

Additionally, you can use Mie superposition for multiple spheres each with multiple layers, ::

  from holopy.scattering.scatterer import Sphere, Spheres
  from holopy.scattering.theory import Mie
  schema = ImageSchema(spacing = 1, shape = 100, 
    optics = Optics(wavelen = 6.58, polarization = [1,0], index=1.33))
  cs1 = Sphere(center=(80, 80, 50), n = (1.59, 1.42), r = (0.3, 0.6))
  cs2 = Sphere(center=(25, 20, 45), n = (1.59, 1.33, 1.59), r = (0.3, 0.6, .9))
  cs3 = Sphere(center=(20, 70, 40), n = (1.33, 1.59, 1.34), r = (0.3, 0.6, .9))
  coatedspheres = Spheres([cs1,cs2,cs3])
  holo = Mie.calc_holo(coatedspheres, schema)

.. note::
        The multisphere theory does not as yet work with coated spheres.
re making HoloPy assume a square array of evenly spaced grid points. You could have written
the same instructions explicitly as: ::

  schema = ImageSchema(shape = (100, 100), spacing = (.1, .1)...)
  

If you wanted a rectangular detector with rectangular pixels, you could specify that as: ::

  schema = ImageSchema(spacing = (.1,.2), shape = (400,300), 
    optics = Optics(wavelen = .660, polarization = [1, 0], index=1.33))

Most displays will default to displaying square pixels, but if your hologram has
an associated spacing (holo.spacing), and you use holopy.show(holo) to display the image, your hologram
will display with pixels of the correct aspect ratio.

Advanced Calculations
=====================

.. _scattering_matrices:

Scattering Matrices
-------------------
In a static light scattering measurement you record scattered intensity at a number of angles.  In this kind of experiment you are usually not interested in the exact distance from the particles, and so instead work with scattering matrices ::

  from holopy.core import Schema, Angles, Optics
  from holopy.scattering.scatterer import Sphere
  from holopy.scattering.theory import Mie
  schema = Schema(positions = Angles(theta = np.linspace(0, np.pi, 100)),
                  optics = Optics(wavelen=.660, polarization = [0,1], index = 1.33))
  sphere = Sphere(r = .5, n = 1.59)

  matr = Mie.calc_scat_matrix(sphere, schema)
  # It is typical to look at scattering matrices on a semilog plot,
  # you can make one with this code
  figure()
  semilogy(np.linspace(0, np.pi, 100), abs(matr[:,0,0])**2)
  semilogy(np.linspace(0, np.pi, 100), abs(matr[:,1,1])**2)
  
Here we omit specifying the location (center) of the scatterer.  This is
only valid when you want a farfield quantity like we do here.

Non-Square Detectors and/or Pixels
----------------------------------

The holograms above make use of several default assumptions.  When you make an ImageSchema like ::

  schema = ImageSchema(shape = 100, spacing = .1...)

you are making HoloPy assume a square array of evenly spaced grid points. You could have written
the same instructions explicitly as: ::

  schema = ImageSchema(shape = (100, 100), spacing = (.1, .1)...)
  

If you wanted a rectangular detector with rectangular pixels, you could specify that as: ::

  schema = ImageSchema(spacing = (.1,.2), shape = (400,300), 
    optics = Optics(wavelen = .660, polarization = [1, 0], index=1.33))

Most displays will default to displaying square pixels, but if your hologram has
an associated spacing (holo.spacing), and you use holopy.show(holo) to display the image, your hologram
will display with pixels of the correct aspect ratio.

