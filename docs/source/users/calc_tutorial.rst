.. _calc_tutorial:

*********************
Calculating holograms
*********************

Scattering models
=================

Holopy contains two different models for calculating holograms. Those are

:Mie:

    This model calculates the scattered field of a single spherical
    particle using the Lorenz-Mie solution. The calculated field is
    the exact solution for the scattering of a single spherical
    particle. The model can also be used to calculate the hologram of
    multiple particles by adding their scattered fields. Since this
    method does not account for multiple scattering, it yields only an
    approximation to the scattering for multiple particles.  The
    approximation is a good one if the particles are sufficiently
    separated.

:T-Matrix: 

    This model calculates the scattered field of a collection of
    particles through a numerical method that accounts for multiple
    scattering and near-field effects (see [Fung2011]_).  The T-Matrix
    approach is much more accurate than Mie superposition, but it is
    more computationally intensive.  There are currently two choices
    for the configuration of the spheres:

        - dimers: two spherical particles
        - trimers: three spherical particles


Each model has a ``forward_holo`` function that will calculate a
hologram. Below we demonstrate how to calculate a hologram with each
of the available models.

Calculating holograms
=====================

In the following code snippets we calculate a hologram from a 1 micron
in diameter spherical polystyrene particle. We assume the refractive
index of the particle is 1.58 (close to that of polystyrene) and the
particle is in water. The particle is 10 microns from the focal plane
and centered in the camera's field of view.  We assume a camera size
of 256-by-256 pixels where the pixels are squares with sides of 10
microns. Using a 100x magnifications optical train the pixel size in
the imaging plane will be 0.1 microns. We also assume that the optics
information has been stored in an instance of the
:class:`holopy.optics.Optics` class called ``my_optics`` ::

    import holopy
    from holopy.optics import Optics
    my_optics = Optics(wavelen = 658e-9, \
			  index = 1.33, pixel_scale=[0.1e-6,0.1e-6])

Alternatively, the optics information could be read from a yaml
file.::

    yaml_file = 'my_experiments/100xObjective_Water.yaml'
    my_optics = Optics(**holopy.io.load_yaml(yaml_file))

In the following examples we will use units of meters and calculate
holograms created with 658 nm light.

Single sphere
-------------

Here we use the single sphere Mie calculations for computing the
hologram.  The arguments to the ``forward_holo`` function are
specified in :meth:`holopy.model.mie.forward_holo`.  They include the
size of the hologram we want to calculate (in pixels) and the
properties and position of the particle::

    from holopy.model import mie
    holo = mie.forward_holo(256, my_optics, \
		 1.58, 0., 0.5e-6, 12.8e-6, 12.8e-6, 10e-6, 0.8)

.. note::
    All units in the above code sample are in meters. This will work
    out fine if the wavelength is also specified in meters. If you
    wanted to do everything in pixels the same bit of code would look
    like::

        holo = mie.forward_holo(256, my_optics, \
			 1.58, 0., 5, 128, 128, 100, 0.8)

    Provided that the wavelength of light was specified in units of
    pixels, this will calculate the same hologram as the previous
    example.


Dimer of spheres
----------------

Dimers are more complicated. You can calculate the hologram from
scattering by a dimer from knowing the positions in space of the two
particles. But the dimer's position and orientation in space can also
be described through:

    * dimer's center of mass
    * two angles describing the dimer's orientation

        * beta
        * gamma

    * gap distance

These properties of a dimer are described below. 

Gap distance
~~~~~~~~~~~~~~~~~~~~

The center-to-center distance :math:`d` between the two particles making up the dimer is given
by :math:`d = r_1 + r_2 + \epsilon_{gap}` where :math:`\epsilon_{gap}` is the gap distance and
:math:`r_1` and :math:`r_2` are the radii of the two spheres.

Euler Angles
~~~~~~~~~~~~

See detailed description below.

The parameters given to the dimer hologram calculating function,
:meth:`holopy.model.tmatrix_dimer.forward_holo`, are:

    * size of hologram
    * instance of :class:`holopy.optics.Optics`
    * real part of both particle's refractive index
    * imaginary part of both particle's refractive index
    * radius of both particles
    * x-, y-, and z-coordinate of the dimer's center-of-mass
    * the Euler angle beta describing dimer's orientation w.r.t. y-axis
    * the Euler angle gamma describing dimer's orientation w.r.t. z-axis
    * gap distance or separation parameter as described above
    * scaling parameter, alpha, that scales the intensity of scattered field

See :meth:`holopy.model.mie.forward_holo` for the ordering of those
parameters. 

Additionally, one can pass this function the following optional
parameters:

    * dictionary of T-matrix parameters
    * ``old_coords`` can be set from default of False to True to set the origin 
      (i.e., (0,0)) at the center of the image
    * ``dimensional`` can be set from default of True to False to use parameters that have all been made
      dimensionless by scaling with the wave vector

While the code can take different values for the particles' real
and imaginary refractive indices, if both particles have the same
refractive index you can specify the second particle's real and imaginary
indices as `None`. 

Dimer example::

    ipython -pylab
    import holopy
    import holopy.io
    import holopy.model.tmatrix_dimer as tmatdimer
    opt = holopy.Optics(**holopy.io.load_yaml('/my_data/60xWater.yaml'))
    opt.index = 1.334
    model = tmatdimer.forward_holo(300,opt,1.58,1.58,0.0001,0.0001, \
                                   0.5, 0.5, 5e-6, 5e-6, 10e-6, 90, 90, 0.8)

In the example above, a 300x300 hologram is calculated. For a dimer of two
1 micron-diameter spheres of index 1.58. 


Trimer of spheres
-----------------

The function :meth:`holopy.model.tmatrix_trimer.forward_holo` will calculate
the hologram due to three touching spheres. Unlike the dimer case, there is
no separation parameter as the particles must be in contact. Additionally, because
the trimer lacks the symmetry of the dimer, three Euler angles must be
used to describe its orientation. 

Trimer example::

    ipython -pylab
    import holopy
    import holopy.model.tmatrix_trimer as tmattrimer
    opts = holopy.io.load_yaml('/my_data/60xWater.yaml')
    h=tmattrimer.forward_holo(256, opts,1.2,0.0001,6.35,6.35,6.35,0,0,10e-6,0,0,0,.6)

Euler Angles
------------
The Euler angle conventions used in holopy are based on the convention used by Daniel Mackowski's
code SCSMFO1B.FOR.  SCSMFO1B's documentation describes its :math:`zyz` Euler angle convention as a *passive transformation*, or change of basis.

It is mathematically equivalent, and in our opinion logically easier, to think of the Euler rotations as 
an *active transformation*, physically rotating a cluster (dimer or trimer) about its center-of-mass from 
a pre-defined *reference configuration* to its actual orientation in the laboratory frame. 
In the active perspective, all rotations are performed about a fixed set of axes in the lab frame.
Then, in the active perspective of holopy, Euler rotations about the angles :math:`\alpha`, 
:math:`\beta`, and :math:`\gamma` do the following:

    * Rotate the cluster from the reference configuration an angle :math:`\alpha` about the laboratory
      :math:`z` axis
    * Rotate the cluster an angle :math:`\beta` about the laboratory :math:`y` axis
    * Rotate the cluster an angle :math:`\gamma` about the laboratory :math:`z` axis

Here, positive rotations are *counterclockwise*, viewed from the origin along the positive :math:`z` or
:math:`y` direction.  It is important to remember how the coordinate axes are oriented in holopy. Positive
angles being counterclockwise is the price paid for using the active transformation perspective.

To be mathematically specific: 

.. image:: ../images/euler_matrix_eqn.png
    :scale: 100 %
    
where :math:`\mathbf{v}` is the laboratory frame vector to an arbitrary point in the cluster reference 
configuration and :math:`\mathbf{v}'''` is the vector to that point in the laboratory frame after the
Euler rotations.

For trimers, which are not axisymmetric, all three Euler angles are necessary. :math:`\alpha` and :math:`\gamma` 
are valid modulo :math:`360^\circ`; the code will give correct output regardless of the value of these angles.
:math:`\beta` is usually only considered valid from :math:`0^\circ` to :math:`180^\circ`; SCMSFO1B handles
this by effectively considering the absolute value of :math:`\beta`. So, hologram calculations will
produce the same output if given :math:`\beta` or :math:`-\beta`. This needs to be remembered in interpreting
data produced by fitting.

Dimers are axisymmetric and we can describe them with just two Euler angles, :math:`\beta` and :math:`\gamma`. 
:math:`\gamma` behaves in the usual way. So that the fitter can explore a continuous parameter space, however,
we have made negative values of :math:`\beta` valid *solely for dimers*. In particular, values of :math:`\beta` less than 0 automatically have 180 added, and values of :math:`\beta` greater than 180 have 180 automatically 
subtracted. Behavior is then consistent between -180 and 360, with the caveat that if one is fitting holograms
of two particles of dissimilar sizes, it is important not to hold both particle radii constant.



