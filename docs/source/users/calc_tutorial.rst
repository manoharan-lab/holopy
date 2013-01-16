.. _calc_tutorial:

************************
Scattering Calculations
************************

Optical physicists and astronomers have worked out how to compute the
light scattering off of many kinds of objects.  HoloPy provides an
easy interface for computing many types of scattering from microscopic
objects.  These include scattered electric fields, scattered intensity
(what a camera records), holograms, and scattering
matrices and cross sections.


Scattering calculations using HoloPy generally consists of the following steps:

1. Define a scatterer using a :mod:`~holopy.scattering.scatterer` object

2. Specify laser properties and where to compute the values using a :mod:`~holopy.core.marray.Schema` object

3. Use a calc function from a :mod:`~holopy.scattering.theory` object
   to compute scattered quantities

Example
==================

For a sphere of radius 0.5 microns centered at (3, 2, 5) microns and a 100x100 pixel camera:
 ::

  from holopy.scattering.scatterer import Sphere
  from holopy.core import ImageSchema, Optics
  from holopy.scattering.theory import Mie

  sphere = Sphere(n = 1.59+.0001j, r = .5, center = (4, 3, 5)) #Scatterer
  schema = ImageSchema(shape = 100, spacing = .1, 
    optics = Optics(wavelen = .660, polarization = [1,0], index = 1.33)) #Schema
  holo = Mie.calc_holo(sphere, schema) #computed hologram using Mie theory

.. note::
   All units of length in the code above are the same (microns). The laser of wavelength 0.660 microns is polarized along x-direction. See :ref:`units` and :ref:`coordinate_system` for additional details. 

Defining a Scatterer
==================

Scatterer objects describe the geometry and optical properties of the objects that scatter light.
They contain information about the position, size, shape, and refractive index of the scatterers.

.. note::

   We include a small imaginary component for the refractive index because it helps avoid potential problems in the scattering calculations. 
   However, it is small enough to have a negligible impact on the computed results.

For other types or collections of scatterers, see :mod:`holopy.scattering.scatterer` or
the :ref:`more_scattering_ex` section.

Defining a Schema
===============

Schema objects tell HoloPy what its calculated results should look
like.
Schemas are intentionally very similar to the objects created by loading data.
We chose to use an :class:`.ImageSchema` Schema here for a square hologram.
For calculations using other kinds of data, you could use a raw :class:`.Schema` object or a different subclass.

.. note::
  This :class:`.ImageSchema` is similar to the :class:`.Image`
  object in that it specifies the coordinates of the camera pixels and contains the
  same optical information. The two object classes differ in that an :class:`.Image`
  contains data for all of the coordinates an :class:`.ImageSchema` may contain only
  the coordinates. So, Holopy allows you to provide an :class:`Image` anywhere an 
  :class:`ImageSchema` is needed.

Scattering Theory
=================

HoloPy contains a number of scattering theories that are useful for different kinds of scatterers which can be found in :mod:`holopy.scattering.theory`.
The simplest one, Mie theory (:class:`holopy.scattering.theory.mie`), can be used to compute a hologram after the Scatterer and ImageSchema has been defined.

.. note::
Functions similar to ``Mie.calc.holo`` may be called in a similar way to compute interesting quantities such as scattering matrices (except for  ``calc_cross_sections``, which is only an Optics object and not a full :class:`.Schema`).
Examples of such a calculation may be found in :ref:`scattering_matrices`.
