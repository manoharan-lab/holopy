.. _releasenotes:


************************
Holopy 3.3 Release Notes
************************


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

