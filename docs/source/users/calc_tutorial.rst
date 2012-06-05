.. _calc_tutorial:

*********************
Calculating holograms
*********************

The code for calculating holograms is split out into a separate package within holopy called :mod:`scatterpy`.  

Scattering Theories: :mod:`scatterpy.theory`
============================================

Holopy contains two different scattering theories for calculating holograms. Those are

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

:Multisphere: 

    This model calculates the scattered field of a collection of
    particles through a numerical method that accounts for multiple
    scattering and near-field effects (see [Fung2011]_).  The T-Matrix
    approach is much more accurate than Mie superposition, but it is
    more computationally intensive.  The Multisphere code can handle
	any	number of spheres


Each model has a ``calc_holo`` function that will calculate a
hologram. Below we demonstrate how to calculate a hologram with each
of the available models.

Scatterer Objects: :mod:`scatterpy.scatterer`
=============================================

Scatterer objects are used to describe the geometry of objects that light scatters off.  They contain information about the position, size, and index of refraction of the scatterers.  The most commonly useful scatterers are

:Sphere:

    Describes a single sphere

:SphereCluster:

    A collection of multiple spheres

:CoatedSphere:

    A sphere with multiple layers each of a different index of refraction.

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
    from holopy import Optics
    my_optics = Optics(wavelen = 658e-9, 
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
hologram.  We create a :class:`scatterpy.scatterer.sphere.Sphere` object to describe the sphere, and a :class:`scatterpy.theory.mie.Mie` object to do the calculation.  The arguments to the ``forward_holo`` function are
specified in :meth:`holopy.model.mie_fortran.forward_holo`.  They
include the size of the hologram we want to calculate (in pixels) and
the properties and position of the particle ::

    from scatterpy.theory import Mie
    from scatterpy.scatterer import Sphere
    sphere = Sphere(center=(12.8e-6, 12.8e-6, 10e-6), n = 1.58, r = 0.5e-6)
    mie_theory = Mie(my_optics, 256)
    holo = mie_theory.calc_holo(sphere, 0.8)
	
.. note::
    All units in the above code sample are in meters. This will work
    out fine if the wavelength is also specified in meters. If you
    wanted to do everything in pixels you would instead define the	sphere as::

        sphere = Sphere(center(128, 128, 100), n = 1.58, r = 5)

    Provided that the wavelength of light was specified in units of
    pixels, this will calculate the same hologram as the previous
    example.


Cluster of Spheres
------------------

Calculating a hologram from a cluster of spheres is done in a very similar manner ::

    from scatterpy.scatterer import SphereCluster
    s1 = Sphere(center=(12.8e-6, 12.8e-6, 10e-6), n = 1.58, r = 0.5e-6)
    s2 = Sphere(center=(12e-6, 11e-6, 10e-6), n = 1.58, r = 0.5e-6)
    cluster = SphereCluster([s1, s2])
    holo = mie_theory.calc_holo(cluster, 0.8)

This will do the calculation with superposition of Mie solutions, if you want to solve the actual multisphere problem for higher accuracy you would instead use ::

    from scatterpy.theory import Multisphere
    multisphere_theory = Multisphere(optics, 256)
    holo = multisphere_theory.calc_holo(cluster, 0.8)

Adding more spheres to the cluster is as simple as defining more sphere objects and passing a longer list of spheres to the :class:`scatterpy.scatterer.SphereCluster` constructor.

Coated Spheres
--------------

Coated (or layered) spheres can use the same Mie theory as normal spheres, Multisphere does not as yet work with coated spheres.  Coated spheres differ from normal spheres only in taking a list of indexes and radii corresponding to the layers ::

    from scatterpy.scatterer import CoatedSphere
    cs = CoatedSphere(center=(12.8e-6, 12.8e-6, 10e-6), n = (1.58, 1.42), r = (0.3e-6, 0.6e-6))
    holo = mie_theory.calc_holo(cs, .8)


	
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



