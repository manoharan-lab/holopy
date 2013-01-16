Module overview
===============

HoloPy is divided into a number of modules that each serve specific
purposes

Core
----

holopy.core defines :class:`~holopy.core.marray.Marray` (array with
metadata) and contains the basic infrastructure for loading and saving
data and results.  It also contains basic image processing routines.
Most of these are simply higher level wrappers around scipy routines.
holopy.core can be used independently of the rest of holopy.

The HoloPy core module is used at the beginning and end of your
workflow:

  1) raw image (or other data) file(s) + metadata -> :class:`~holopy.core.marray.Marray` object
  2) Raw :class:`~holopy.core.marray.Marray` object + processing -> processed :class:`~holopy.core.marray.Marray` object
  3) Computed or Processed Result -> Archival yaml text or text/binary result

Scattering
----------

Used to compute simulated scattering from defined scatterers.  The
scattering package provides objects and methods to define scatterer
geometries, and theories to compute scattering from specified
geometries.  Scattering depends on holopy.core, and certain scattering
theories may require external scattering codes.

The HoloPy scattering module is generally used to:

  1) Describe geometry as a :mod:`~holopy.scattering.scatterer` object
  2) Define the result you want as a :mod:`~holopy.core.marray.Schema` object
  3) Calculate scattering quantities with an
     :mod:`~holopy.scattering.theory` appropriate for your
     :mod:`~holopy.scattering.scatterer` ->
     :class:`~holopy.core.marray.Marray` object

Propagation
-----------

Computes light propagation from one known set of points to another
set, possibly through media.  Depends on core.

Propagation is used primarily for one operation:

  1) :class:`.Image` or :class:`.VectorImage` (Electric field) -> :class:`.Image` or :class:`.VectorImage` at another position

Fitting
-------

Fitting is used to determine the Scatterer which best reproduces some observed
data.  Fitting depends on Core and Scattering.

Fitting is used to:

  1) Define Scattering Model -> :class:`~holopy.fitting.model.Model` object
  2) Fit model to data -> :class:`.FitResult` object

Visualization
-------------

The visualization module is used to, surprise, visualize your results
or data.  If the appropriate display libraries are present, it can
show images or slices of your data and 3d renderings of volume data or
scatterers.

  1) Marray or Scatterer object -> plot or rendering


We'll go over these steps in the next section and the tutorials.
