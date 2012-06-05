.. holopy documentation master file, created by
   sphinx-quickstart on Wed Dec 30 20:43:15 2009.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Holography in Python
==================================

:Release: |version|
:Date: |today|

The :mod:`holopy` package contains routines for loading digital
holograms, processing them, and extracting useful data.  It allows you
to reconstruct a 3D volume from a hologram -- the digital equivalent
of shining light through a holographic film -- or fit holograms to
scattering models, allowing you to obtain precise information about
the positions and optical properties of the objects used to make the
hologram.  In both cases, the idea is that a single 2D holographic
image of a sample encodes information about its 3D structure.  Holopy
helps you retrieve that information.  It can also be used in reverse,
to calculate holograms given a 3D structure.

The following features work and are well-tested:

* Calculating holograms from scattering models of:

  * Single spheres (based on the Lorenz-Mie solution) 
  * Sphere doublets (based on a T-Matrix approach)
  * Sphere trimers (based on a T-Matrix approach)
  * Elipsoids and other complicated structures (based on the Discrete Dipole Approximation (DDA))

* Fitting holograms to any of the above scattering models
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
   reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


