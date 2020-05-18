.. _scatterers_user:

The HoloPy Scatterer
====================

The HoloPy :class:`.Scatterer` class defines objects that are described by
numerical quantities (e.g. dimension, location, refractive index) and have
known light-scattering behaviour described by a :class:`.ScatteringTheory`.

Scatterers are generally used in two scenarios:
    - All numerical properties (e.g. dimension, location, refractive index) are
      fixed to simulate a specific light scattering experiment.
    - Some numerical properties are defined as :class:`.Prior` objects,
      representing unknown values to be determined in an inference calculation.

You can find examples of these use cases in the :ref:`calc_tutorial` and
:ref:`fit_tutorial` tutorials.

Scatterer objects in HoloPy are inherited from two base classes:

- :ref:`CenteredScatterer<individual_scatterer>` describes a single object,
  with an optional location specified
- :ref:`Scatterers<composite_scatterer>` describes a collection of
  individual scatterers

Scatterer Attributes
--------------------

All HoloPy Scatterer classes have the following properties/methods:

General manipulation
    - :attr:`~.Scatterer.x`, :attr:`~.Scatterer.y`, :attr:`~.Scatterer.z`
      Components of scatterer center
    - :meth:`~.Scatterer.translated`
      New scatterer with location coordinates shifted by a vector
    - :meth:`~.Scatterer.select`
      New scatterer with the given quantities taking only the subset specified,
      e.g. refractive index at two wavelengths to refractive index at one
      wavelength only

Inference calculations
    - :attr:`~.Scatterer.parameters`
      Dictionary of all numerical values needed to describe the scatterer.
      Values described as :class:`.Prior` objects will appear that way here as
      well.
    - :attr:`~.Scatterer.guess`
      New scatterer with all varying parameter values fixed to their best guess
    - :meth:`~.Scatterer.from_parameters()`
      New scatterer built from a dictionary of parameters, with an option to
      replace only varying parameters or overwrite fixed values too

Discretization
    - :class:`indicators<.Indicators>`
      Function(s) to describe region(s) of space occupied by scatterer
      domain(s)
    - :meth:`~.Scatterer.index_at()`
      Scatterer's refractive index at given coordinates
    - :meth:`~.Scatterer.in_domain()`
      Which domain of the scatterer the given coordinates are in
    - :meth:`~.Scatterer.contains()`
      Check whether a particular point is in any domains of the scatterer
    - :attr:`~.Scatterer.num_domains`
      Number of domains of the scatterer
    - :attr:`~.Scatterer.bounds`
      Extents of the scatterer in each dimension
    - :meth:`~.Scatterer.voxelate`
      3D voxel grid representation of the scatterer containing its refractive
      index at each point

.. _individual_scatterer:

Individual Scatterers
---------------------

:class:`.CenteredScatterer` objects are not instantiated directly,
but instead in one of the subclasses:

- :class:`.Sphere`
  Can contain multiple concentric layers defined by their outer radius
- :class:`.LayeredSphere`
  Defines multiple concentric layers by their layer thickness
- :class:`.Cylinder`
- :class:`.Ellipsoid`
- :class:`.Spheroid`
- :class:`.Bisphere`
  Union of two spheres
- :class:`.Capsule`
  Cylinder with semi-spherical caps on either end
- :class:`.JanusSphere_Uniform`
  Sphere with a semi-spherical outer layer of constant thickness
- :class:`.JanusSphere_Tapered`
  Sphere with a semi-spherical outer layer that has a crescent profile
- :class:`.CsgScatterer`
  Allows for construction of an arbitrary scatterer by constructive solid
  geometry

.. _composite_scatterer:

Composite Scatterers
--------------------
:class:`.Scatterers` objects contain multiple individual scatterers,and
support the following features in addition to those shared with
:class:`.CenteredScatterer`:

Component scatterer handling
    - Support for selecting component scatterers with square brackets and
      python slicing syntax
    - :meth:`~.Scatterers.add()`
      Adds a new scatterer to the composite in-place
    - :meth:`~.Scatterers.rotated()`
      New scatterer rotated about its center according to
      :ref:`HoloPy rotation conventions<rotations>`

Tied parameters
    - Varying parameters shared between component scatterers that are the same
      python object are automatically tied together. Tied parameters appear as
      a single entry in :attr:`.Scatterers.parameters`.
    - :attr:`~.Scatterers.ties`
      Dictionary of all tied parameters, with keys indicating the shared name
    - :meth:`~.Scatterers.add_tie`
      Manually tie parameters together
    - :attr:`~.Scatterers.raw_parameters`
      Dictionary of parameters ignoring ties

There are two specific classes for working with collections of
spheres that have additional functionality:

:class:`.Spheres`
    A collection of spherical scatterers, with the following properties:

    - :attr:`~.Spheres.overlaps`
      List of pairs of component spheres that overlap
    - :attr:`~.Spheres.largest_overlap`
      Maximum overlap distance between component spheres

:class:`.RigidCluster`
    A collection of spherical scatterers in fixed relative positions.
    The entire cluster can be translated and/or rotated.
    :attr:`.RigidCluster.scatterers` and :meth:`.RigidCluster.from_parameters`
    both return :class:`.Spheres` type objects.
