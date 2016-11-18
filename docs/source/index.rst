Holography and Light Scattering in Python
=========================================


:Release: |release|
:Date: |today|

:mod:`HoloPy` is a python based tool for working with digital
holograms and light scattering. HoloPy can:

* :ref:`Load <load_tutorial>` images, associate them with experimental
  :ref:`metadata <metadata>`, and visualize loaded or calculated images.

* :ref:`Reconstruct <recon_tutorial>` 3D volumes from digital holograms

* Perform :ref:`Scattering Calculations <calc_tutorial>` with fast Fortran/C code

  * **Compute** Holograms, electric fields, scattered intensity,
    cross sections, ...

  * **From** spheres, clusters of spheres, and arbitrary structures
    (using :ref:`DDA <dda_tutorial>`)

* Use Bayesian analysis methods to :ref:`infer <infer_tutorial>` a
  scattering model that is most consistent with experimental data.

* Make fast, precise measurements to :ref:`fit scattering models
  <fit_tutorial>` (based on the above structures) to experimental
  data, based on an inital guess.

HoloPy provides a powerful and user-friendly interface to scattering
and optical propagation theories. It also provides a set of flexible
objects that make it easy to describe and analyze data from complex
experiments or simulations.

The easiest way to see what HoloPy is all about is to jump to the
examples in our :ref:`user_guide`.

HoloPy started as a project in the `Manoharan Lab at Harvard
University <http://manoharan.seas.harvard.edu/>`_. If you use HoloPy,
you may wish to cite one or more of the sources listed in
:ref:`credits`. We also encourage you to sign up for our `User Mailing
List <https://groups.google.com/d/forum/holopy-users>`_ to keep up to
date on releases, answer questions, and benefit from other users'
questions.

.. toctree::
   :maxdepth: 2

   users/index
   reference/holopy
   credits

HoloPy is based upon work supported by the National Science Foundation
under Grant No. CBET-0747625 and performed in the `Manoharan Lab at
Harvard University <http://manoharan.seas.harvard.edu>`_
