**********************
Fitting Models to Data
**********************

The most powerful use of holopy is to analyze data by fitting a model to it.  Given a correct model this can give high precision measurements of physical quantities from holographic or other data.

A fit needs:

1) A :class:`.Model` of the scattering system
   
2) The :class:`.Marray` containing data the fit is trying to match
   
3) (Optional) a :class:`~holopy.fitting.minimizer.Minimizer` (holopy defaults to using the supplied :class:`~holopy.fitting.minimizer.Nmpfit`)

The simplest fit will look like ::

  result = fit(model, data)

Scattering Model
================

Model is a set of variable parameters a scattering theory for
computing simulated scattering data, and any ancillary information
needed for this calculation.  ::

  model = Model(parametrization, theory.calc_*)

Parametrization
---------------

There are two ways to specify a
:class:`~holopy.fitting.model.Parametrization`.  The standard method is
to provide a :class:`~holopy.scattering.scatterer` object telling which
values to fix and which to vary. ::

  from holopy.fitting import par
  param_scat = Sphere(n = 1.59, r = par(guess = .5, limit = [.3, .8]),
                      center = (par(10), par(10), par(10)))

This will tell holopy that you want to model scattering from a sphere
of fixed index n = 1.59, and vary the sphere's radius and position to
attempt to match a hologram.  Initial guesses are provided for the
radius and three center coordinates, and the radius is constrained to
lie between .3 and .8.  The three radii are allowed to vary without
limit.  This example makes use of a shorthand, we encourage you use
the provided shortcut ``par`` to mean :class:`~holopy.fitting.parameter.Parameter`
because you will be typing it a lot.

If your model does not fit neatly into a parametrized scatterer like
this, Holopy provides a lower level interface ::

  param = Parametrization(make_scatterer,
                          parameters = [par(guess = .5, name = 'r'),
                                        par(guess = 0, name = euler_alpha'),
                                        par(guess = 0, name = 'euler_beta')])

Here make_scatterer needs to be a function that takes keyword
arguments of the names of the parameters and returns a scatterer.

Tying Parameters
~~~~~~~~~~~~~~~~

You may desire to fit holograms with *tied parameters*, in which
several physical quantities that could be varied independently are
constrained to have the same (but non-constant) value. A common
example involves fitting a model to a multi-particle hologram in which
all of the particles are constrained to have the same refractive
index, but the index is determined by the fitter.  This may be done by
defining a Parameter and using it in multiple places ::
  
  n = par(1.59)
  sc = Spheres([Sphere(n = nl, r = par(0.5e-6), center = array([10., 10., 20.])),
                Sphere(n = n, r = par(0.5e-6), center = array([9., 11., 21.]))])

Theory
------

The theory in a model is one of the calc_* functions provided by
scattering theories, for example :meth:`.Mie.calc_holo`.

Technically, you can use any function here as long as it takes a
scatterer and a :class:`.Schema` (and optionally additional keyword
arguments) as arguments and returns an :class:`.Marray` object.

Other information
-----------------

If you want to provide a scaling alpha, that can be done as a keyword
argument to the model ::
  
  model = Model(param_scat, Mie.calc_holo, alpha = par(.6, [0, 1]))

.. note::

   We are working on understanding the theory behind these scaling
   factors and hope to be able to eliminate this scaling paramater.
   Thus, we hope to remove this option at some point in the future
   when it becomes unnecessary.  
  
If you want to fit to information normally provided in the metadata,
you can provide a parametrized :class:`.Schema` object, any parameters
specified here will override those specified in the data ::

  model = Model(param_scat, mie.calc_holo,
                target_overlay = DataTarget(optics = Optics(divergence
				= par(0, [0, 1]))))

.. note::

   This is a feature preview.  Holopy does not currently support
   fitting to metadata.  

Data
====

An :class:`.Marray` object with a full set of metadata.  Between the model and
the provided :`.Marray`, you must specify or parametrize all of the values
needed to perform a scattering calculation.

.. note::

   We have not tested Holopy at all extensively for fitting to
   Marray's other than Image.  

Minimizer
=========

If you do not provide a minimizer, fits will default to using the
supplied Nmpfit minimizer which can be called explicitly as follows::

  fit(model, data, minimizer = Nmpfit())

You can choose another minimizer or provide non-default options to a
minimizer by passing a minimizer object to fit(), for example (To tell nmpfit to use looser tolerances and a small iteration limit
(to get a fast result to check things out).)::

  fit(model, data, minimizer = Nmpfit(ftol=1e-5, xtol = 1e-5, gtol=1e-5, niter=2))

or to use OpenOpt's ralg minimizer instead of nmpfit  (This will fail unless you have OpenOpt installed and configured so that Holopy can
find it.)::

  fit(model, data, minimizer = Ralg())

.. note::

   This is a feature preview.  Holopy currently only supports fitting
   with the supplied Nmpfit.  
  
If you need to provide information to the minimizer about specific
parameters (for example a derivative step to nmp fit) you add them to
the par call as keyword args, for example ::

  Sphere(n = par(1.59, [1, 2], step = 1e-3), ...)

Examples
========

Sphere
------

Let's compute a hologram with known parameters and then fit it to make sure we retrieve the right parameters.  Instead, you can replace the
calculated hologram (holo) with real data, if you like. TODO: result is not very accurate... why? ::

   import holopy
   from holopy.core import ImageTarget, Optics
   from holopy.fitting import Model, par, fit
   from holopy.scattering.scatterer import Sphere
   from holopy.scattering.theory import Mie

   target = ImageTarget(shape = 100, pixel_size = .1, optics = Optics(wavelen = .660, index = 1.33))
   s = Sphere(center = (10.2, 9.8, 10.3), r = .5, n = 1.58)
   holo = Mie.calc_holo(s, target)

   par_s = Sphere(center = (par(guess = 10, limit = [5,15]), par(10, [5, 15]), par(10, [5, 15])),
                  r = .5, n = 1.58)

   model = Model(par_s, Mie.calc_holo, alpha = par(.6, [.1, 1]))
   result = fit(model, holo)

Here we specify the three spatial coordinates as parameters, and fix
the index of refraction and radius of the sphere.

``result.scatterer`` is the scatterer that best matches the hologram,
``result.alpha`` is the alpha for the best fit.  ``result.chisq`` and
``result.rsq`` are statistical measures of the the goodness of the fit.
``result.model`` and ``result.minimizer`` are the Model and Minimizer
objects used in the fit, and ``result.minimization_info`` contains any
further information the minimization algorithm returned about the
minimization procedure (for nmpfit this includes things like covariance
matrices).  See the documentation of :class:`.FitResult`.

Saving Results
~~~~~~~~~~~~~~

You will most likely want to save the fit result ::

  holopy.save('result.yaml', result)

This saves all of the information about the fit to a yaml text
file.  These files are reasonably human readable and serve as our archive format for data.  They can be loaded back into python with ::

  loaded_result = holopy.load('result.yaml')

Complex Index of Refraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
You can specify a complex index with ::

  Sphere(n = ComplexParameter(real = par(1.58), imag = 1e-4))

This will fit to the real part of index of refraction while holding the imaginary part fixed.  You can fit to it as well by specifying a Parameter instead of a fixed number there.  

Spheres
-------

In this example, we fit for the parameters of two spheres ::

    import holopy
    from holopy.scattering.scatterer import Sphere
    from holopy.scattering.scatterer import Spheres
    from holopy.scattering.theory import Mie
    from holopy.core import ImageTarget, Optics
    from holopy.fitting import Model, par, fit
    from holopy.fitting.minimizer import Nmpfit

    #calculate a hologram with known particle positions to do a fit against
    target = ImageTarget(shape = 256, pixel_size = .1, 
        optics = Optics(wavelen = .660, index = 1.33))

    s1 = Sphere(center=(15, 15, 20), n = 1.59, r = 0.5)
    s2 = Sphere(center=(14, 14, 20), n = 1.59, r = 0.5)
    cluster = Spheres([s1, s2])
    holo = Mie.calc_holo(cluster, target)

    #now do the fit
    guess1 = Sphere(center = (par(guess = 15.5, limit = [5,25]), 
        par(14.5, [5, 25]), par(22, [5, 25])), r = .5, n = 1.59)
    guess2 = Sphere(center = (par(guess = 14.5, limit = [5,25]), 
        par(13.5, [5, 25]), par(22, [5, 25])), r = .5, n = 1.59)
    par_s = Spheres([guess1,guess2])

    model = Model(par_s, Mie.calc_holo, alpha = par(.6, [.1, 1]))
    result = fit(model, holo)

Hologram with Beam Tilt
-----------------------

.. note::

   This is a feature preview.  Holopy does not yet support fitting to
   metadata.  

Here we override some of the parameters specified in the Data (or in fact you can leave them as none when specifying Metadata for this data) ::

  model = Model(Sphere(...), target_overlay = DataTarget(optics = Optics(
    ilum_vector = UnitVector(beta = par(0), gamma = par(0))))

Fitting this model will vary the beam tilt.  UnitVector is a composite parameter like ComplexParameter with the special constraint that it stay normalized.  

Static Light Scattering
-----------------------

.. note::

   This is a feature preview.  A fit of this sort might work, but we
   have not tested it.  

Assuming you have recorded some static light scattering data in a file sls_data.txt and the metadata in sls_meta.yaml ::

  data = hp.load('sls_data.txt', 'sls_meta.yaml')

  model = Model(Sphere(n = par(1.58, [1, 2]), r = par(.5)), Mie.calc_scat_matr, scaling = par(1))

  result = fit(model, data)

Alternative Scatterer Parametrizations
---------------------------------------

.. note::

   This is a feature preview.  Holopy does not provide any
   Parametrizations of this sort yet.  

Holopy also provides some additional views of scatterers that may be convenient for fitting.  For example ::

  from holopy.fitting.views import Dimer
  s = Dimer([Sphere(n, r), Sphere(n, r)], gap, beta, gamma, center)

This contains the same number of paramters as a 2 sphere Spheres
object and fully specifies a Spheres object, but provides a different
set of knobs for the fitter to adjust.
