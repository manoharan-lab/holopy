.. _theories_user:


Scattering Theories in Holopy
=============================

The HoloPy :class:`.ScatteringTheory` classes know how to calculate scattered fields from detector and scatterer information. Each scattering theory is only able to work with certain specific scatterers.

There are two broad classes of scattering theories in HoloPy: the
:ref:`Lens-free<_lens_free>` theories which treat the recorded fields as the
magnified image of the fields at the focal plane, and the
:ref:`Lens<_with_lens>` theories which use a more detailed description of the
effects of the objective lens. The lens-free theories usually do not need any
additional parameters specified, whereas the lens theories need the lens's
acceptance angle, which can be specified as either a fixed number or a
:class:`.Prior` object, representing an unknown value to be determined in an
inference calculation.

All scattering theories in holopy inherit from the :ref:`ScatteringTheory` class.


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

If a theory is asked for the raw fields, but does not have a `_raw_fields` method, the scattering theory attempts to calculate them via the scattering matrices, as called by `_raw_scat_matrs`. More than one of these methods may be implemented for performance reasons.

Be advised that the :class:`.ScatteringTheory` class is under active development, and these method names may change.


.. _lens_free:


Lens-Free Scattering Theories
-----------------------------

- :class:`.DDA`
    * Can handle every :class:`.Scatterer` object in HoloPy
    * Computes scattered fields using the discrete dipole approximation, as
      implemented by ADDA.
    * Requires the ADDA package to be installed separately, as detailed in
      the :ref:`DDA section<_dda_tutorial>`
    * Functions in two different ways, as controlled by the
      `use_indicators` flag. If the `use_indicators` flag is `True`, the
      scatterer is voxelated within HoloPy before passing to DDA. If the
      flag is `False`, ADDA's built-in scatterer geometries are used for
      things like spheres, cylinders, ellipsoids, etc.
- :class:`.Mie`
    * Can handle :class:`.Sphere` objects, :class:`.LayeredSphere` objects, or
      :class:`.Spheres` through superposision.
    * Computes scattered fields using Mie theory.
- :class:`.Multisphere`
    * Can handle :class:`.Spheres` objects.
    * Computes scattered fields using a matrix-based solution of scattering,
      accounting for multiple scattering between spheres to find a
      (numerically) exact solution.
- :class:`.Tmatrix`
    * Can handle :class:`.Sphere`, :class:`.Cylinder`, or :class:`.Spheroid`
      objects.
    * Computes scattered fields by calculating the T-matrix for axisymmetric
      scatterers, to find a (numerically) exact solution.
    * Occasionally has problems due to Fortran compilations.


.. _with_lens:


Lens-Free Scattering Theories
-----------------------------
- :class:`.Lens`
    * Create by including one of the :ref:`Lens-Free<_lens_free>` theories.
    * Can handle whatever the additional included theory can handle.
    * Considerably slower than the normal scattering theory.
    * Performance can be improved if the `numexpr` package is installed.
- :class:`.MieLens`
    * Can handle :class:`.Sphere` objects, or :class:`.Spheres` through
      superposision.
    * Computes scattered fields using Mie theory, but incorporates diffractive
      effects of a perfect objective lens.
    * Used for performance; `MieLens(lens_angle)` is much faster than calling
      `Lens(lens_angle, Mie())` and slightly faster than `Mie()`.

