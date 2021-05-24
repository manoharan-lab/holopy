Holography and Light Scattering in Python
=========================================

.. image:: https://travis-ci.com/manoharan-lab/holopy.svg?branch=develop
    :target: https://travis-ci.com/github/manoharan-lab/holopy
    :alt: Development Branch Build Status

.. image:: https://readthedocs.org/projects/holopy/badge/?version=latest
    :target: http://holopy.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

`HoloPy <http://holopy.readthedocs.io>`_ is a python based
tool for working with digital holograms and light scattering. HoloPy
can:

* `Load <http://holopy.readthedocs.io/en/latest/tutorial/load_tutorial.html#loading-and-viewing-a-hologram>`_ images, associate them with experimental
  `metadata <http://holopy.readthedocs.io/en/latest/tutorial/load_tutorial.html#telling-holopy-about-your-experimental-setup>`_, and visualize loaded or calculated images.

* `Reconstruct <http://holopy.readthedocs.io/en/latest/tutorial/recon_tutorial.html>`_ 3D volumes from digital holograms

* Do `Scattering Calculations <http://holopy.readthedocs.io/en/latest/tutorial/calc_tutorial.html>`_

  * **Compute** Holograms, electric fields, scattered intensity,
    cross sections, ...

  * **From** spheres, clusters of spheres, and arbitrary structures
    (using `DDA <http://holopy.readthedocs.io/en/latest/tutorial/dda_tutorial.html>`_)

* Make precise measurements by `fitting scattering models
  <http://holopy.readthedocs.io/en/latest/tutorial/fit_tutorial.html>`_ (based on the above structures) to experimental
  data.

HoloPy provides a powerful and user-friendly interface to scattering
and optical propagation theories. It also provides a set of flexible
objects that make it easy to describe and analyze data from complex
experiments or simulations. HoloPy's optical propagation theories work
for holograms of arbitrary objects; HoloPy's current scattering
calculations accurately describe scatterers in sizes from tens of
micrometers and smaller.

The easiest way to see what HoloPy is all about is to jump to the
examples in our `user guide <http://holopy.readthedocs.io/en/latest/tutorial/index.html>`_.

HoloPy started as a project in the `Manoharan Lab at Harvard
University <http://manoharan.seas.harvard.edu/>`_. If you use HoloPy,
you may wish to cite one or more of our `papers
<http://manoharan.seas.harvard.edu/holographic-microscopy>`_. We also
encourage you to sign up for our `User Mailing List
<https://groups.google.com/d/forum/holopy-users>`_ to keep up to date
on releases, answer questions, and benefit from other users'
questions.


HoloPy is based upon work supported by the National Science Foundation
under grant numbers CBET-0747625, DMR-0820484, DMR-1306410, and
DMR-1420570.
