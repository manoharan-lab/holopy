.. _theories_user:


The HoloPy Scattering Theories
==============================

The HoloPy :class:`.ScatteringTheory` classes know how to calculate scattered fields from detector and scatterer information. Each scattering theory is only able to work with certain specific scatterers.

There are two broad classes of scattering theories in HoloPy:
:ref:`lens-free<lens_free>` theories which treat the recorded fields as
the magnified image of the fields at the focal plane, and
:ref:`lens-based<with_lens>` theories which use a more detailed
description of the effects of the objective lens. The lens-free theories
usually do not need any additional parameters specified, whereas the
lens theories need the lens's acceptance angle, which can be specified
as either a fixed number or a :class:`.Prior` object, representing an
unknown value to be determined in an inference calculation.

All scattering theories in HoloPy inherit from the :class:`.ScatteringTheory` class.

Not sure how to choose a scattering theory? See the
:ref:`how_to_choose_theory` section.


ScatteringTheory Methods
------------------------

HoloPy Scattering theories calculate the scattered fields through one of the following methods.

    - :meth:`~.ScatteringTheory._raw_fields`
      Calculates the scattering fields.
    - :meth:`~.ScatteringTheory._raw_scat_matrs`
      Calculates the scattering matrices.
    - :meth:`~.ScatteringTheory._raw_cross_sections`
      Calculates the cross sections.
    - :meth:`~.ScatteringTheory._can_handle`
      Checks if the theory is compatible with a given scatterer.

If a theory is asked for the raw fields, but does not have a ``_raw_fields`` method, the scattering theory attempts to calculate them via the scattering matrices, as called by ``_raw_scat_matrs``. More than one of these methods may be implemented for performance reasons.

Be advised that the :class:`.ScatteringTheory` class is under active development, and these method names may change.


.. _lens_free:


Lens-Free Scattering Theories
-----------------------------

- :class:`.DDA`
    * Can handle every :class:`.Scatterer` object in HoloPy
    * Computes scattered fields using the discrete dipole approximation, as
      implemented by ADDA.
    * Requires the ADDA package to be installed separately, as detailed in
      the :ref:`DDA section<dda_tutorial>`
    * Functions in two different ways, as controlled by the
      ``use_indicators`` flag. If the ``use_indicators`` flag is
      ``True``, the scatterer is voxelated within HoloPy before passing
      to ADDA. If the flag is ``False``, ADDA's built-in scatterer
      geometries are used for things like spheres, cylinders,
      ellipsoids, etc.
- :class:`.Mie`
    * Can handle :class:`.Sphere` objects, :class:`.LayeredSphere` objects, or
      :class:`.Spheres` through superposition.
    * Computes scattered fields using Mie theory.
- :class:`.Multisphere`
    * Can handle :class:`.Spheres` objects.
    * Cannot handle :class:`.Spheres` objects composed of layered
      spheres.
    * Computes scattered fields through a T-matrix-based solution of
      scattering, accounting for multiple scattering between spheres to
      find a (numerically) exact solution.
- :class:`.Tmatrix`
    * Can handle :class:`.Sphere`, :class:`.Cylinder`, or :class:`.Spheroid`
      objects.
    * Computes scattered fields by calculating the T-matrix for axisymmetric
      scatterers, to find a (numerically) exact solution.
    * Occasionally has problems due to Fortran compilations.


.. _with_lens:


Lens-Based Scattering Theories
------------------------------
- :class:`.Lens`
    * Create by including one of the :ref:`Lens-Free<lens_free>` theories.
    * Can handle whatever the additional included theory can handle.
    * Considerably slower than the normal scattering theory.
    * Performance can be improved if the `numexpr
      <https://pypi.org/project/numexpr/>`_ package is installed.
- :class:`.MieLens`
    * Can handle :class:`.Sphere` objects, or :class:`.Spheres` through
      superposision.
    * Computes scattered fields using Mie theory, but incorporates diffractive
      effects of a perfect objective lens.
    * Used for performance; ``MieLens(lens_angle)`` is much faster than calling
      ``Lens(lens_angle, Mie())`` and slightly faster than ``Mie()``.


.. _how_to_choose_theory:

Which Scattering Theory should I use?
-------------------------------------

HoloPy chooses a default scattering theory based off the scatterer type,
currently determined by the function
:func:`.determine_default_theory_for`. If you're not satisfied with
HoloPy's default scattering theory selection, you should choose the
scattering theory based off of (1) the scatterer that you are modeling,
and (2) whether you want to describe the effect of the lens on the
recorded hologram in detail.


An Individual Sphere
~~~~~~~~~~~~~~~~~~~~

For single spheres, the default is to calculate scattering using Mie
theory, implemented in the class :class:`.Mie`. Mie theory is the exact
solution to Maxwell's equations for the scattered field from a spherical
particle, originally derived by Gustav Mie and (independently) by Ludvig
Lorenz in the early 1900s.

Multiple Spheres
~~~~~~~~~~~~~~~~

A scatterer composed of multiple spheres can exhibit multiple scattering and
coupling of the near-fields of neighbouring particles. Mie theory doesn't
include these effects, so :class:`.Spheres` objects are by default calculated
using the :class:`.Multisphere` theory, which accounts for multiple
scattering by using the SCSMFO package from `Daniel Mackowski
<http://www.eng.auburn.edu/~dmckwski/>`_.  This calculation uses
T-matrix methods to give the exact solution to Maxwell's equation for
the scattering from an arbitrary arrangement of non-overlapping spheres.

Sometimes you might want to calculate scattering from multiple spheres
using Mie theory if you are worried about computation time or if your
spheres are widely separated (such that optical coupling between the
spheres is negligible) You can specify Mie theory manually when calling
the :func:`.calc_holo` function, as the following code snippet shows:


..  testcode::

    import holopy as hp
    from holopy.core.io import get_example_data_path
    from holopy.scattering import (
        Sphere,
        Spheres,
        Mie,
        calc_holo)

    s1 = Sphere(center=(5, 5, 5), n = 1.59, r = .5)
    s2 = Sphere(center=(4, 4, 5), n = 1.59, r = .5)
    collection = Spheres([s1, s2])

    imagepath = get_example_data_path('image0002.h5')
    exp_img = hp.load(imagepath)

    holo = calc_holo(exp_img, collection, theory=Mie)

..  testcode::
    :hide:

    print(holo[0,0,0].values)

..  testoutput::
    :hide:

    1.04802354...


Note that the multisphere theory does not work with collections of
multi-layered spheres; in this case HoloPy defaults to using Mie theory
with superposition.

Non-spherical particles
~~~~~~~~~~~~~~~~~~~~~~~

HoloPy also includes scattering theories that can calculate scattering
from non-spherical particles. For cylindrical or spheroidal particles,
by default HoloPy calculates scattering from cylindrical or spheroidal
particles by using the :class:`.Tmatrix` theory, which uses the T-matrix
code from `Michael Mishchenko
<https://www.giss.nasa.gov/staff/mmishchenko/t_matrix.html>`_.


..  testcode::

    from holopy.scattering.theory import Tmatrix
    from holopy.scattering.scatterer import Spheroid

    spheroid = Spheroid(n=1.59, r=(1., 2.), center=(4, 4, 5))
    theory = Tmatrix()
    holo = calc_holo(exp_img, spheroid, theory=theory)

Holopy can also access a discrete dipole approximation (DDA) theory to model
arbitrary non-spherical objects. See the :ref:`dda_tutorial` tutorial for more
details.


Including the effect of the lens
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most of the scattering theories in HoloPy treat the fields on the detector as
a (magnified) image of the fields at the focal plane. While these theories
usually provide a good description of holograms of particles far above the
focus, when the particle is near near the focus subtle optical effects can
cause deviations between the recorded hologram and theories which do not
specifically describe the effects of the lens. To deal with this, HoloPy
currently offers two scattering theories which describe the effects of a
perfect lens on the recorded hologram. Both of these scattering theories
need information about the lens to make predictions, specifically the
acceptance angle of the lens. The acceptance angle :math:`\beta` is
related to the numerical aperture or NA of the lens by :math:`\beta =
\arcsin(NA / n_f)`, where :math:`n_f` is the refractive of the immersion
fluid. For more details on the effect of the lens on the recorded
hologram, see our papers
`here <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-28-2-1061>`_
and `here <url>`_.

The :class:`.Lens` theory allows HoloPy to include the effects of a perfect
objective lens with any scattering theory. The Lens theory works by wrapping a
normal scattering theory. For instance, to calculate the image of a sphere in
an objective lens with an acceptance angle of 1.0, do

..  testcode::

    from holopy.scattering.theory import Lens, Mie
    lens_angle = 1.0
    theory = Lens(lens_angle, Mie())

This theory can then be passed to :func:`.calc_holo` just like any other
scattering theory. However, calculations with the :class:`.Lens` theory
are very slow, orders of magnitude slower than calculations without the
lens.

To get around the slow speed of the :class:`.Lens` theory, HoloPy offers
an additional theory, :class:`.MieLens`, specifically for spherical
particles imaged with a perfect lens. For spherical particles, some
analytical simplifications are possible which greatly speed up the
description of the objective lens -- in fact, the :class:`.MieLens`
theory's implementation is slightly faster than :class:`.Mie` theory's.
The following code creates a :class:`.MieLens` theory, which can be
based to :func:`.calc_holo` just like any other scattering theory:

..  testcode::

    from holopy.scattering.theory import MieLens
    lens_angle = 1.0
    theory = MieLens(lens_angle)


My Scattering theory isn't here?!?!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add your own scattering theory to HoloPy! See :ref:`scat_theory` for
details. If you think your new scattering theory may be useful for other
users, please consider submitting a `pull request
<https://github.com/manoharan-lab/holopy/pulls>`_.

