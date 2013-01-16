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
works with all such data as :class:`holopy.core.marray.Marray` objects,
which are arrays of point measurements along with metadata about how
to interpret those values.  Metadata might be:

:Measurement Locations:
   
   All :class:`~holopy.core.marray.Marray` objects must know where in
   space their measurements were taken.  This is specified as a
   :class:`~holopy.core.metadata.PositionSpecification` object, which
   could specify the pixel spacing on a rectangular detector such as a
   camera or a list of angular positions taken by a far-field detector
   such as a photodiode.  In general it is a list of coordinates
   specifying each measurement location.

:Optical Setup:
   
   Details about the optical system and optical path are specified as
   a :class:`.metadata.Optics` object containing the wavelength and
   polarization of the illumination source (assumed to be a plane
   wave) and the index of refraction of the medium the scatterers are
   in.

:Other Experimental Conditions:

   HoloPy can associate arbitrary metadata with data to describe how
   the data was obtained or any processing that was been done on the
   data.

Data can be:

:1-dimensional:
   for example, data from static light scattering measurements
:2-dimensional:
   for example, data from a camera
   (:class:`holopy.core.marray.Image`), or a timeseries of 1d data sets 
:3-dimensional:
   volumetric data (:class:`holopy.core.marray.Volume`) such as image
   stacks, or a timeseries of images
:4-dimensional:
   for example, a timeseries of volume data

HoloPy will probably work with higher dimensional data (because the
underlying NumPy arrays we use are n-dimensional), but many of its
functions expect data in one of these forms.

.. _coordinate_system: 

Coordinate system
-----------------

For :class:`~holopy.core.marray.Image` data (data points arrayed in a
regular grid in a single plane), HoloPy defaults to placing the
origin, (0,0), at the top left corner as shown below. The x-axis runs
vertically down, the y-axis runs horizontally to the right, and the
z-axis points out of the screen, toward you.  This corresponds to the
way that images are treated by most computer software.

.. image:: ../images/HolopyCoordinateSystem.png
   :scale: 30 %
   :alt: Coordinate system used in HoloPy.

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

The sense of rotation is as follows: each angle is a rotation in the *clockwise*
direction about the specified axis, viewed along the positive direction of the axis from
the origin. This is the usual sense of how rotations are typically defined in math.



