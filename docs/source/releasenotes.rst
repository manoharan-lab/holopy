.. _releasenotes:

********************
HoloPy Release Notes
********************

Holopy 3.5
==========

Announcements
-------------
If you encounter errors loading prevoiusly saved HoloPy objects, try loading
them with HoloPy 3.4 and saving a new version. See deprecation notes below.

New Features
------------
- New :class:`.AberratedMieLens` allows for calculating holograms of spheres
  as imaged through a lens with spherical aberrations.

Improvements
------------
- Calling a numpy ufunc on a Prior object with name kwarg gives the resulting
  TransformedPrior object that name, e.g. clip_x = np.min(x, name='clipped').
- Cleaned up model parameter names created from TransformedPrior objects.
- CmaStrategy now scales first step size based on initial population, not
  prior.
- Inference models work with scattering theories that require
  parameters. See more in the user guide :ref:`scatterers_user`.
- Interpolation in background images is now robust to adjacent dead pixels.
- Restored ability to call scattering functions on parameterized scatterers.

Documentation
-------------
- Updated inference tutorial

Bugfixes
--------
- NmpfitStrategy now correctly accounts for non-uniform priors when optimizing.
- Functional fitting interface no longer lets alpha go to zero.
- Now able to save Model objects whose scatterer attribute contains xarrays.

Compatibility Notes
--------------------
- HoloPy now assumes dictionaries are ordered, so it requires python>=3.7.

Developer Notes
---------------
- The :class:`.ScatteringTheory` now performs scattering calculations
  only, as its single responsibility. This should make it easier to
  implement new scattering theories. Code that was previously in
  :class:`.ScatteringTheory` that calculated deterimed at which points
  the scattering matrices or scattered fields needed to be calculated is
  now in `holopy.scattering.imageformation`.
- The parameter parsing previously done by the `Model` class has now been
  broken out to a new `hp.core.parameter_mapping` module so it can be
  accessed by non-`Model` objects.
- `prior.py` has been moved from to hp.core module from hp.inference.

Deprecations
------------
- PerfectLensModel is now deprecated; lens models are now directly
  fittable with either AlphaModel or ExactModel. To do so, pass in a
  :class:`holopy.prior.Prior` object as the `lens_angle`.
- Inference-related deprecations started in 3.4 are now complete. This means
  that some old `holopy` inference objects are no longer loadable. If you still
  need to access these objects, `holopy` version 3.4 will let you load old
  inference objects and save them in the new format that is compatible with
  this (and future) versions of `holopy`.


Holopy 3.4
==========

New Features
------------
- New :class:`.Lens` scattering theory to model the effect of an objective lens
  can be applied to any other scattering theory.
- New :class:`.TransformedPrior` that applies a function to one or multiple
  component :class:`.Prior` objects and maintains ties in a :class:`.Model`.

Improvements
------------
- DDA scattering theories no longer default to printing intermediate C output.
- It is now possible to save all slices of a reconstruction stack as images.
- Rearrangement of some Scatterer properties and methods so they are now
  accessible by a broader group of classes.
- PerfectLensModel now accepts hologram scaling factor alpha as a parameter
  for inference.
- It is now possible to pass an inference strategy to the high-level fit() and
  sample() functions, either by name or as a Strategy object.
- High level inference functions fit() and sample() are now accessible in the
  root HoloPy namespace as hp.fit() and hp.sample().
- Scatterer.parameters() now matches the arguments to create the scatterer
  instead of deconstructing composite objects.
- New prior.renamed() method to create an identical prior with a new name.
- New way to easily construct scatterers from model parameters with
  ``model.scatterer_from_parameters()``.
- New ``model.initial_guess`` attribute which can be used to evaluate initial
  guess by psasing into ``model.scatterer_from_parameters()`` or
  ``model.forward()`` methods.
- Model parameters now use the names of their prior objects if present.
- Standardized parameter naming across composite objects (eg. list, dict).
- Any model parameters can now be tied, not just specific combinations within
  Scatterers objects.
- Expanded math operations of :class:`.Prior` objects, including numpy ufuncs.
- Math operations on :class:`Prior` objects now use :class:`.TransformedPrior`
  to maintain ties when used in a :class:`.Model`.


Documentation
-------------
- New user guide on :ref:`scatterers_user`.
- New user guide on :ref:`theories_user`.
- More discussion of scattering theories in tutorial.

Deprecations
------------
- The model.fit() and model.sample() methods have been deprecated in favour of
  the high-level hp.fit() and hp.sample functions().
- Adjustments to saving of Model objects (and Results objects containing them).
  Backwards compatibility is supported for now, but be sure to save new copies!
- Scatterer.guess no longer exists. Instead, you must define a model and use:
  ``model.scatterer_from_parameters(model.initial_guess)``.
- Scatterer.from_parameters() is no longer guaranteed to return a
  definite object.
- Composite scatterers no longer keep track of tied parameters.
- Scattering interface functions such as calc_holo() now require a definite
  scatterer without priors.

Bugfixes
--------
- Fortran output no longer occasionaly leaks through the output supression
  context manager used by multiple scattering theories.
- Restored ability to visualize slices through a scatterer object
- Now possible to fit only some elements of a list, eg. Scatterer center
- Models can now include xarray parameters and still support saving/loading.
- The :class:`.MieLens` scattering theory now works for both large and
  small spheres.
- The :class:`Lens` theory works for arbitrary linear polarization of
  the incoming light. This bug was not present on any releases, only on
  the development branch.

Compatibility Notes
--------------------
- Holopy's hard dependencies are further streamlined, and there is improved
  handling of missing optional dependencies.

Developer Notes
---------------
- Documentation now automatically runs sphinx apidoc when building docs.
- New Scatterer attribute ``_parameters`` provides a view into the scatterer
  and supports editing.
- :class:`.ComplexPrior` now inherits from :class:`.TransformedPrior`, but
  Model maps don't keep track of this, e.g. in `model.scatterer`.


Holopy 3.3
==========

New Features
------------
- Inference in `holopy` has been overhauled; take a look at the updated
  docs to check it out! Briefly, the inference and fitting modules have
  been combined into a unified, object-oriented interface, with several
  convenience functions available to the user both for the inference
  strategies and the inference results. One noticeable change with this
  is that the least-squares based fitting algorithms such as `Nmpfit`
  now work correctly with priors, including with non-uniform priors.
  There is also a new, user-friendly functionality for inference in
  `holopy`. Moreover, the inference pipelines can work with arbitrary
  user-defined functions instead of just holograms.
- There is a new scattering theory, `holopy.scattering.theory.MieLens`,
  which describes the effect of the objective lens on recorded holograms
  of spherical particles. This new theory is especially useful if you
  want to analyze particles below the microscope focus.
- There are two new inference strategies: a global optimizer CMA-ES
  strategy, under `holopy.inference.cmaes.CmaStrategy`, and a
  least-squares strategy which uses `scipy.optimize.leastsq` instead of
  the `Nmpfit` code.


Deprecations
------------
- The keyword argument `normals` is deprecated in `detector_points`,
  `detector_grid`, and related functions, as the old implementation was
  incorrect. This deprecation is effective immediately; calling code
  with the `normals` keyword will raise a `ValueError`.
- The old fitting interface, in `holopy.fitting`, is in the process of
  being deprecated (see "New Features" above). Calling the old fitting
  interface will raise a `UserWarning` but will otherwise work until the
  next `holopy` release.


Bugfixes
--------
In addition to many minor bugfixes, the following user-facing bugs have
been fixed:

- `load_average` now works with a cropped reference image and uses less
  memory on large image stacks.
- Issues with loss of fidelity on saving and loading objects have been
  fixed.
- A bug where `hp.propagate` failed when `gradient_filter=True` has been
  fixed.
- Tied parameters in inference calculations works correctly on edge
  cases.
- Inference should work with more generic scatterers.
- The Fortran code should be easier to build and install on Windows
  machines. This is partially done via a post-install script that
  checks that files are written to the correct location (which corrects
  some compiler differences between Windows and Linux). We still
  recommend installing Holopy with conda.


Improvements
------------
- User-facing docstrings have been improved throughout `holopy`.
- `schwimmbad` now handles parallel computations with Python's
  `multiprocessing` or `mpi`.
- More types of objects can be visualized with `hp.show`.
- DDA default behaviour now has `use_indicators=True` since it is faster
  and better tested
- The scaling of initial distributions both for Markov-Chain Monte Carlo
  and for CMA inference strategies can now be specified.


Compatibility Notes
--------------------
- We are curently phasing out support for pre-3.6 Python versions (due
  to ordered vs unordered dicts).


Developer Notes
---------------
- Test coverage has dramatically increased in `holopy`.
- Tests no longer output extraneous information on running.
- The `ScatteringTheory` class has been refactored to allow for faster,
  more flexible extension.


Miscellaneous Changes
----------------------
- Some previously required dependencies are now optional.

