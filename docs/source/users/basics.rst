Fundamentals
============

.. _units:

Units
-----

HoloPy does **not** enforce any particular set of units. As long as
you are consistent, you can use any set of units, for example pixels,
meters, or microns.  So if you specify the wavelength of your red imaging
laser as 658 then all other units (*x*, *y*, *z* position coordinates,
particle radii, etc.)  must also be specified in nanometers.

Data
----

HoloPy is at its core a tool for working with optical data, either
measurements from some experiment or simulations of such data.  HoloPy
works with all such data as :class:`holopy.core.marray.Marray` objects
which are arrays of point measurements along with metadata about how
to interpret those values.  Metadata might be:

:Measurement Locations:
   
   All Marray objects must know where in space their measurements were
   taken.  This is specified as a
   :class:`~holopy.core.metadata.PositionSpecification` object which can be
   by pixel spacing for a rectangular detector, angles to farfield
   detectors, or in general a list of coordinates for each measurement
   location.

:Optical Setup:
   
   Details about the optical system and optical path are specified 
   as a :class:`.metadata.Optics` object containing wavelength,
   polarization, divergence, ... of illumination light, index of
   refraction of the medium the scatterers are in, and any lenses or other optical
   elements present in the beam path.

:Other Experimental Conditions:

   HoloPy can associate arbitrary metadata with data to describe any
   relevant conditions of how the data was obtained or processing that
   has been done on the data.  

Data can be:

:1-dimensional:
   static light scattering measurements
:2-dimensional:
   :class:`~holopy.core.marray.Image`, or timeseries of 1d data sets
:3-dimensional:
   :class:`~holopy.core.marray.Volume` data (stacks) or timeseries of images
:4-dimensional:
   timeseries of volume data

HoloPy will probably work to some degree with higher dimensional data
(because the underlying numpy arrays we use are n-dimensional), but
many of its functions expect data in one of these forms.

.. _coordinate_system: 

Coordinate system
-----------------

For Image data (data points arrayed in a grid in a single plane),
HoloPy defaults to placing the origin, (0,0), at the top left corner
as shown below. The x-axis runs vertically down, the y-axis runs
horizontally to the right, and the z-axis points out of the screen,
toward you.  This corresponds to the way that images are treated by
most computer software.

.. image:: ../images/HoloPyCoordinateSystem.png
    :scale: 30 %
    :alt: Coordinate system used in holopy.

In sample space, we choose the z axis so that distances to objects
from the camera/focal plane are positive (have positive z
coordinates).  The price we pay for this choice is that the
propagation direction of the illumination light is then negative.

More complex detector geometries will define their own origin, or ask
you to define one.
	

Rotations of Scatterers
-----------------------
Certain scattering calculations in HoloPy require specifying the orientation
of a scatterer (such as a Janus sphere) relative to the HoloPy coordinate
system. We do this in the most general way possible by specifying three
Euler angles and a reference orientation. Rotating a scatterer initially
in the reference orientation through the three Euler angles :math:`\alpha`,
:math:`\beta`, and :math:`\gamma` (in the active transformation picture)
yields the desired orientation. The reference orientation is specified by the 
definition of the scatterer.

The Euler rotations are performed in the following way:

1. Rotate the scatterer an angle :math:`\alpha` about the HoloPy :math:`z` axis.
2. Rotate the scatterer an angle :math:`\beta` about the HoloPy :math:`y` axis.
3. Rotate the scatterer an angle :math:`\gamma` about the HoloPy :math:`z` axis.

The sense of rotation is as follows: each angle is a rotation in the *counterclockwise*
direction about the specified axis, viewed along the positive direction of the axis from
the origin. This is the opposite sense of how rotations are typically defined, but this
convention is adopted for compatability with software which adopts the usual convention
but in a passive transformation picture.


Overview
--------

HoloPy is divided into a number of packages that each serve specific
purposes

Core
^^^^

holopy.core defines Marray (array with metadata) and Metadata objects
and contains the basic infrastructure for loading and saving data and
results.  It also contains basic image processing routines for
cleaning up your data.  Most of these are simply higher level wrappers
around scipy routines.  holopy.core can be used independently of the
rest of holopy.

HoloPy Core is used at the beginning and end of your workflow:

  1) raw image (or other data) file(s) + metadata -> :class:`~holopy.core.marray.Marray` object
  2) Raw :class:`~holopy.core.marray.Marray` object + processing -> processed :class:`~holopy.core.marray.Marray` object
  3) Computed or Processed Result -> Archival yaml text or text/binary result

Scattering
^^^^^^^^^^

Used to compute simulated scattering from defined scatterers.  The
scattering package provides objects and methods to define scatterer
geometries, and theories to compute scattering from specified
geometries.  Scattering depends on holopy.core (and certain scattering
theories may depend on external scattering codes).

HoloPy Scattering is generally used to:

  1) Describe geometry as :mod:`~holopy.scattering.scatterer` object
  2) Define the result you want as a :mod:`~holopy.core.marray.Schema` object
  3) Calculate scattering quantities with an :mod:`~holopy.scattering.theory` appropriate for your :mod:`~holopy.scattering.scatterer` -> :class:`~holopy.core.marray.Marray` object

Propagation
^^^^^^^^^^^

Compute light propagation from one known set of points to another set
of points, possibly through media or optical elements.  Depends on
core (and on scattering if propagating through with nonuniform media).

Propagation is used primarily for one operation:

  1) :class:`.Image` or :class:`.VectorImage` (Electric field) -> :class:`.Image` or :class:`.VectorImage` at another position

Fitting
^^^^^^^

Fitting is used to determine the Scatterer which best creates some observed
data.  Fitting depends on Core and Scattering.

Fitting is used to:

  1) Define Scattering Model -> :class:`~holopy.fitting.model.Model` object
  2) Fit model to data -> :class:`.FitResult` object

Visualization
^^^^^^^^^^^^^

The visualization module is used to, surprise, visualize your results
or data.  If the appropriate display libraries are present, it can
show images or slices of your data and 3d renderings of volume data or
scatterers.

  1) Marray or Scatterer object -> plot or rendering


We'll go over these steps in the next section and the tutorials.
