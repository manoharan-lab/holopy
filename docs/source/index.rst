.. holopy documentation master file, created by
   sphinx-quickstart on Wed Dec 30 20:43:15 2009.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Holography and Light Scattering in Python
=========================================

:Release: |version|
:Date: |today|

:mod:`HoloPy` is a python based tool for working with digital
holograms and light scattering. HoloPy's main purposes are to:

* :ref:`Load <loading>` images and associate them with experimental
  :ref:`metadata <metadata>`

* :ref:`Reconstruct <recon_tutorial>` 3d volumes from digital holograms

* Do :ref:`Scattering Calculations <calc_tutorial>`

  * **Compute** Holograms, Electric fields, scattered intensity, 
    cross sections, ...

  * **From** spheres, clusters of spheres, and arbitrary structures 
    (using :ref:`DDA <dda_tutorial>`)

* Make precise measurements by :ref:`fitting scattering models
  <fit_tutorial>` (based on the above structures) to experimental
  data.
 
HoloPy provides a powerful and (relatively) user friendly interface to
scattering theories, nonlinear minimizers, and optical propagation. It
provides a set of flexible objects that make it easy to describe
complex experiments or theoretical situations.

The easiest see what HoloPy is all about is to jump in to the examples
in our :ref:`user_guide`.

HoloPy started as a project in the `Manoharan Lab at Harvard
University <http://manoharan.seas.harvard.edu/>`_. If you use HoloPy,
you may wish to cite one or more of the sources listed in :ref:`credits`. 

.. toctree::
   :maxdepth: 2

   downloads
   users/index
   reference/holopy
   credits


Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

