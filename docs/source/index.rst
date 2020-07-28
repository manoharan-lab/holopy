Holography and Light Scattering in Python
=========================================


:Release: |release|

:mod:`HoloPy` is a python based tool for working with digital
holograms and light scattering. HoloPy can be used to analyze holograms in two complementary ways:

* Backward propagation of light from a digital hologram to :ref:`reconstruct <recon_tutorial>` 3D volumes.
    * This approach requires no prior knowledge about the scatterer

* Forward propagation of light from a :ref:`scattering calculation <calc_tutorial>` of a predetermined scatterer.
    * Comparison to a measured hologram with :ref:`Bayesian inference <fit_tutorial>` allows precise measurement of scatterer properties and position.

HoloPy provides a powerful and user-friendly python interface to fast scattering
and optical propagation theories implemented in Fortran and C code. It also provides a set of flexible
objects that make it easy to describe and analyze data from complex
experiments or simulations.

HoloPy started as a project in the `Manoharan Lab at Harvard
University <http://manoharan.seas.harvard.edu/holopy>`_. If you use HoloPy,
you may wish to cite one or more of the sources listed in
:ref:`credits`. We also encourage you to sign up for our `User Mailing
List <https://groups.google.com/d/forum/holopy-users>`_ or join us on `GitHub
<https://github.com/manoharan-lab/holopy>`_ to keep up to date on releases, 
answer questions, and benefit from other users' questions.

.. toctree::
   :maxdepth: 2

   releasenotes
   tutorial/index
   users/index
   reference/holopy
   credits

HoloPy is based upon work supported by the National Science Foundation
under grant numbers CBET-0747625, DMR-0820484, DMR-1306410, and
DMR-1420570. 
