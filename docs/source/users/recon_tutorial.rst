.. _recon_tutorial:

**************************
Propagation/Reconstruction
**************************

The classical way of working with holograms is to optically reconstruct them by shining light back through them.  This has the mathematical effect of propagating the field stored in the hologram to some different plane.  Holopy generalizes this concept and allows you to numerically propagate any hologram or electric field to another point in space.  

Import the code
===============

To begin, import the :mod:`holopy.analyze.reconstruct` module. ::

    import holopy
    from holopy import propagate, show


:func:`holopy.propagate` is also provided as an alias to :func:`holopy.propagation.propagation`


Load a hologram
==================

The first thing to do is to create a :class:`holopy.hologram.Hologram`
object which will store the image data of the hologram as well as
metadata about the optics used in creating the hologram.  See
:ref:`loading` and :ref:`metadata`.  Usually you will specify the
imaging laser wavelength, the medium refractive index, and the pixel
size of the camera as the metadata.


Reconstruct the hologram
========================
To reconstruct an x-y slice you must pass (at least) the hologram and the
z-distance of the reconstructed image to the function
:func:`holopy.analyze.reconstruct.reconstruct`

In the following bit of code, an image 10 microns from the focal plane
is reconstructed. The reconstruct function returns a
:class:`holopy.analyze.reconstruct.Reconstruction` instance. Like with
the :class:`holopy.hologram.Hologram` instance, this can be treated
like a numpy array.

.. sourcecode:: ipython

    In [4]: rec_xy = reconstruct(holo, 10)

    In [6]: rec_xy.dtype
    Out[6]: dtype('complex128')

    In [7]: rec_vol = reconstruct(holo, np.linspace(5, 15, 10))


The reconstructed image is a complex array. The
dimensions of the imaging plane (x&y) are the first two
dimensions.  If a range of z distances are specified the array will be 3 dimensional, x, y, z.  If the hologram is a 3d time series, the reconstructed image will be a 3d x, y, t, or 4d x, y, z, t array.

Visualizing Reconstructions
============================

Now that you have the reconstruction, you need to view it. A few 
resources come in handy when visualizing and working with the
reconstructions.

Holopy provides convenience wrappers around display routines from `matplotlib <http://matplotlib.sourceforge.net/>`_ and `MayaVI <http://code.enthought.com/projects/mayavi/>`_ though since our Data is an numpy array, you can use the raw plotting libraries as well.

For viewing 2d slices, use hp.show ::

  holopy.show(rec_vol)
  holopy.show(rec_vol.angle())
  holopy.show(rec_vol.imag())

By default holopy.show shows the magnitude of a complex image.  For volume images, you can step through slices in the z plane by using the left and right arrow keys on your keyboard.

To see three dimensional renderings, use holopy.render ::

  holopy.render(rec_vol, 'contour')

This will show a contour surface plot of the reconstruction, using MayaVI.  

Propagating through Non-Homogeneous Media
=========================================

The propagation discussed above assumes propagation through free space or a homogeneous dielectric medium.  However, holopy can also propagate a field  through an optical elements :: 

  from holopy.propagation import ThinLens
  rec = progagate(holo, 1e-5, optical_train = ThinLens(f = 1e-2, z =   1e-1)

or an inhomogeneous medium ::

  medium = holopy.load('medium.yaml')
  rec = propagate(holo, 1e-5, medium = medium)

Holopy defaults to centering the medium or optical elements on the center of the data field (Or should we specify there center relative to the origin of the coordinate system (upper left corner for images)?  I think we will almost always want to center things, so it is better to make it default, there than the slight akwardness if the Data does not have a well defined center - tgd).  You can specify on offset vector if you don't want them centered ::

  ThinLens(f = 1e-2, z = 1e-1, offset = (1e-4, 1e-4))


Changing Propagation Model
==========================

Holopy defaults to a linear model of propagation by convolution with pointspread functions.  If asked to compute propagation through a nonuniform medium it switches to its DDA propagation model.  If you wish to manually control the propagation model you can use ::

  rec = propagate(holo, 1e-5, propagation = FresnelTransform)

Be aware that not all propagation models can support all kinds of data, media, and optical elements, so you may get an exception if for example you try to use FresnelTransform with nonuniform media.  If you leave the propagation model unspecified holopy will try to find one that will work for your conditions and only fail if it has no valid model.  



  