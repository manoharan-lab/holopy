.. _recon_tutorial:

**************************
Propagation/Reconstruction
**************************

The classical way of working with holograms is to optically
reconstruct them by shining light back through them.  This has the
mathematical effect of propagating the field stored in the hologram to
some different plane.  HoloPy generalizes this concept and allows you
to numerically propagate any hologram or electric field to another
point in space.

Propagating and viewing the result takes 3 steps:

1. Import or calculate a hologram

2. Propagate

3. View

Example
===============
.. TODO: provide a complete example and refactor the rest to follow the
.. three steps above

To begin, import: ::

    import holopy
    from holopy import propagate, show, load


Load a hologram
===============

The first thing to do is to create a :class:`.Image`
object which will store the image data of the hologram as well as
metadata about the optics used in creating the hologram, imaging laser
wavelength, the medium refractive index, and the pixel size of the
camera.  See :ref:`loading` and :ref:`metadata`.


Reconstruct the hologram
========================
To reconstruct an x-y slice you must pass (at least) the hologram and the
z-distance of the reconstructed image to the function
:func:`.propagate`

In the following bit of code, an image 10 microns from the focal plane
is reconstructed. ::

  rec_xy = propagate(holo, 10)

  rec_xy.dtype
  dtype('complex128')

  rec_vol = propagate(holo, np.linspace(5, 15, 10))


The reconstructed image is a complex array. The dimensions of the
imaging plane (x&y) are the first two dimensions.  If a range of z
distances are specified the array will be 3 dimensional, x, y, z.  

Visualizing Reconstructions
===========================

Now that you have the reconstruction, you need to view it. A few 
resources come in handy when visualizing and working with the
reconstructions.

HoloPy provides convenience wrappers around display routines from
`matplotlib <http://matplotlib.sourceforge.net/>`_ and `MayaVI
<http://code.enthought.com/projects/mayavi/>`_ though since our Data
is an numpy array, you can use the raw plotting libraries as well.

For viewing 2d slices, use :func:`hp.show <holopy.vis.show>`::

  hp.show(rec_vol)
  hp.show(rec_vol.angle())
  hp.show(rec_vol.imag())

By default hp.show shows the magnitude of a complex image.  For
volume images, you can step through slices in the z plane by using the
left and right arrow keys on your keyboard.