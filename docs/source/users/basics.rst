Fundamentals
============

Concepts
----------

We call a :dfn:`hologram` the digital image you record with a
holographic optical train.  Sometimes when people say "hologram" they
are actually referring to the 3D image you get when you shine light
through a holographic film, like the ones you see in museums or on
your credit card.  We use the term :dfn:`reconstruction` for the 3D
image produced by shining light through a hologram.

Coordinate system
------------------

The coordinate system we use in Holopy places the origin, (0,0), at
the top left corner as shown below. The x-axis runs vertically down, 
the y-axis runs horizontally to the right, and the z-axis points out
of the screen, toward you.

In sample space, we choose the z axis so that distances to objects from the camera/focal plane are positive (have positive z coordinates).  The price we pay for this choice is that the propagation direction of the illumination light is then negative.  

.. image:: ../images/HolopyCoordinateSystem.png
    :scale: 40 %
    :alt: Coordinate system used in holopy.


	

Units
-----

Holopy does **not** enforce any particular set of units. As long as
you are consistent, you can use any set of units, for example pixels,
meters, microns.  So if you specify the wavelength of your red imaging
laser as 658 then all other units (*x*, *y*, *z* position coordinates,
particle radii, etc.)  must also be specified in nanometers.


Overview
--------

Holopy can be used to analyze holograms or calculate them.
Calculating holograms is straightforward: you tell Holopy where your
scatterers are and give it some optical data, and it calculates a
hologram.

If you're using Holopy to analyze holograms, you're probably starting
with a digital image (in any of many image formats).  To analyze it,
you will also need to have some information about the optical train
used to produce it.  

There are two workflows for analysis:

Reconstruction
^^^^^^^^^^^^^^

   1) Load raw image files, import metadata (optics) -> :class:`holopy.hologram.Hologram` object
   2) Clean up and process hologram -> processed :class:`holopy.hologram.Hologram` object
   3) Reconstruct -> :class:`holopy.analyze.reconstruct.Reconstruction`
   4) Post-process -> :class:`holopy.analyze.reconstruct.Reconstruction`
   5) Visualize or save

"Process" means any image processing operations, such as to clean up
the background or cut out a region of interest.

Routines for step 1 are in the :mod:`holopy.io` module, for steps 2
and 4 in the :mod:`holopy.process` module, and for step 3 in the
:mod:`holopy.analyze.reconstruct` modules.

Calculation
^^^^^^^^^^^

   1) Describe geometry as :mod:`scatterpy.scatterer` object
   2) Choose a :mod:`scatterpy.theory` object appropriate to the scatterer
   3) calculate a hologram of the scatterer with the theory -> `holopy.hologram.Hologram` object
	 
Fitting
^^^^^^^

   1) Load raw image files, import metadata (optics) -> :class:`holopy.hologram.Hologram` object
   2) Clean up and process hologram -> processed :class:`holopy.hologram.Hologram` object
   3) Generate initial guess
   4) Fit -> List of parameter values (index of refraction, radius, position, ...)
   5) Visualize or save

Routines for step 1 are in the :mod:`holopy.io` module, for step 2 in
the :mod:`holopy.process` module, and for steps 3 and 4 in the
:mod:`holopy.analyze.fit` module.

We'll go over these steps in the next section and the tutorials.
