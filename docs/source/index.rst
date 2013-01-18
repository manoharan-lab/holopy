.. holopy documentation master file, created by
   sphinx-quickstart on Wed Dec 30 20:43:15 2009.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Holography and Light Scattering in Python
=========================================

:Release: |version|
:Date: |today|

The :mod:`HoloPy` package provides tools for working with digital
holograms and light scattering data.  It provides user-friendly access
to light scattering models and a number of tools for working with
experimental data.  It allows you to reconstruct a 3D volume from a
hologram (the digital equivalent of shining light through a
holographic film), compute scattering patterns or holograms of
objects, or fit scattering models to your data to obtain precise
information about the positions and optical properties of the objects
that were used to generate the optical data.

The following features work and are well-tested:

* Calculating holograms using scattering models of:

  * Single spheres (based on the Lorenz-Mie solution) 
  * Arbitrary clusters of spheres (using multisphere superposition or Mie superposition)
  * Elipsoids and other complicated structures (based on the Discrete Dipole Approximation or DDA)

* Fitting any of the above scattering models to holograms
* Reconstructing 3D volumes from holograms

HoloPy started as a project in the `Manoharan Lab at Harvard
University <http://manoharan.seas.harvard.edu/>`_. If you use HoloPy,
you may wish to cite one or more of the sources listed in :ref:`credits`. 

To get started, please have a look at the :ref:`users`.

Contents:

.. toctree::
   :maxdepth: 2

   users/index
   reference/holopy
   credits


Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

