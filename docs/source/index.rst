.. holopy documentation master file, created by
   sphinx-quickstart on Wed Dec 30 20:43:15 2009.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Holography and Light Scattering in Python
=========================================

:Release: |version|
:Date: |today|

The :mod:`holopy` provides tools for working with digital holograms
and light scattering.  It provides a user friendly package to a lot of
optical physics and a number of tools for working with experimental
data.  It allows you to reconstruct a 3D volume from a hologram (the
digital equivalent of shining light through a holographic film),
compute light scattering or holograms of objects, or construct models
of light scattering in your data and fit them to obtain precise
information about the positions and optical properties of the objects
used to make the hologram.  A single 2D holographic image of a sample
encodes information about its 3D structure.  Holopy helps you retrieve
that information.


The following features work and are well-tested:

* Calculating holograms from scattering models of:

  * Single spheres (based on the Lorenz-Mie solution) 
  * Arbitrary clusters of spheres (using multisphere superposition or Mie superposition)
  * Elipsoids and other complicated structures (based on the Discrete Dipole Approximation (DDA))

* Fitting any of the above scattering models to holograms
* Reconstructing 3D volumes from holograms

Holopy started as a project in the `Manoharan Lab at Harvard
University <http://manoharan.seas.harvard.edu/>`_ If you use Holopy,
you may wish to cite one or more of the sources listed in :ref:`credits`. 

To get started, please have a look at the :ref:`users`.

Contents:

.. toctree::
   :maxdepth: 3

   credits
   users/index
   reference/modules


Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


