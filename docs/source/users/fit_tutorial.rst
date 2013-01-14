**********************
Fitting Models to Data
**********************

The most powerful use of holopy is to analyze data by fitting a model to it.  Given a correct model this can give high precision measurements of physical quantities from holographic or other data.

A fit needs:

1) A :class:`.Model` of the scattering system
   
2) The :class:`.Marray` containing data the fit is trying to match
   
3) (Optional) a :class:`~holopy.fitting.minimizer.Minimizer` (holopy defaults to using the supplied :class:`~holopy.fitting.minimizer.Nmpfit`)

Example
================
Let's compute a hologram with known parameters and then fit it to make sure we 
retrieve the right parameters. ::

   import holopy
   from holopy.core import ImageSchema, Optics
   from holopy.fitting import Model, par, fit
   from holopy.scattering.scatterer import Sphere
   from holopy.scattering.theory import Mie

   schema = ImageSchema(shape = 100, spacing = .1, 
       optics = Optics(wavelen = .660, polarization = [1, 0], index = 1.33))
   s = Sphere(center = (5, 5, 10.3), r = .5, n = 1.58)
   holo = Mie.calc_holo(s, schema)

   par_s = Sphere(center = (par(guess = 5.5, limit = [4,10]), par(4.5, [4, 10]), par(10, [5, 15])),
                  r = .5, n = 1.58)

   model = Model(par_s, Mie.calc_holo, alpha = par(.6, [.1, 1]))
   result = fit(model, holo)

After running this, you can inspect result.scatterer and see that the initial guess of the sphere's location
(5.5, 4.5, 10.0) was corrected by the fitter to (5.0,5.0,10.3). Success!

In this first simple fit, we specify the three spatial coordinates as variable parameters, and set
the index of refraction and radius of the sphere to constant values.

``result.scatterer`` is the scatterer that best matches the hologram,
``result.alpha`` is the alpha for the best fit.  ``result.chisq`` and
``result.rsq`` are statistical measures of the the goodness of the fit.
``result.model`` and ``result.minimizer`` are the Model and Minimizer
objects used in the fit, and ``result.minimization_info`` contains any
further information the minimization algorithm returned about the
minimization procedure (for nmpfit this includes things like covariance
matrices).  See the documentation of :class:`.FitResult`.




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
this, HoloPy provides a lower level interface ::

  from holopy.fitting import Parametrization
  param = Parametrization(make_scatterer,
                          parameters = [par(guess = .5, name = 'r'),
                                        par(guess = 0, name = 'euler_alpha'),
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


Data
====

An :class:`.Marray` object with a full set of metadata.  Between the model and
the provided :`.Marray`, you must specify or parametrize all of the values
needed to perform a scattering calculation.

.. note::

   We have not tested HoloPy at all extensively for fitting to
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

or to use OpenOpt's ralg minimizer instead of nmpfit  (This will fail unless you have OpenOpt installed and configured so that HoloPy can
find it.)::

  fit(model, data, minimizer = Ralg())

.. note::

   This is a feature preview.  HoloPy currently only supports fitting
   with the supplied Nmpfit.  
  
If you need to provide information to the minimizer about specific
parameters (for example a derivative step to nmp fit) you add them to
the par call as keyword args, for example ::

  Sphere(n = par(1.59, [1, 2], step = 1e-3), ...)

Examples
========



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
    from holopy.core import ImageSchema, Optics
    from holopy.fitting import Model, par, fit
    from holopy.fitting.minimizer import Nmpfit

    #calculate a hologram with known particle positions to do a fit against
    target = ImageSchema(shape = 256, spacing = .1, 
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

   This is a feature preview.  HoloPy does not provide any
   Parametrizations of this sort yet.  

HoloPy also provides some additional views of scatterers that may be convenient for fitting.  For example ::

  from holopy.fitting.views import Dimer
  s = Dimer([Sphere(n, r), Sphere(n, r)], gap, beta, gamma, center)

This contains the same number of paramters as a 2 sphere Spheres
object and fully specifies a Spheres object, but provides a different
set of knobs for the fitter to adjust.

Fitting Time Series of Images
-----------------------

If you are taking video holograms (one of the most useful cases), you
will probably find yourself wanting to fit long timeseries of data.
This is done with :func:`holopy.fitting.fit.fit_series` ::

  fit_series(model, dataset, prefit = None, postfit = None)

for each image in the dataset, fit series will:

1) Get the next data from the dataset, if it is a string, load an
   image by that name from the current directory -> Data object
 
2) Call prefit(data, model, **kwargs) -> Data object
   
3) Fit the model to the data -> FitResult object
   
4) Save the FitResult with holopy.save
   
5) Call postfit(fitresult, **kwargs) -> FitResult
   
6) Use the last FitResult to setup the guess for the next frame.  

The fit_series arguments are:

:model:

   Described exactly as for a single fit.  The same model is used to
   fit all of the data.  If you need to vary the model across the fit
   you can instead provide a ModelFamily object and select between
   models in prefiting

:dataset:

   The simplest dataset is a list of filenames, but it can be any
   iterable that returns either a strings or a Data objects.  If the
   dataset is string filenames, metadata will need to be provided as a
   separate metadata keyword argument to fit_series, or the model
   specified with a full set of metadata.  

:prefit (optional):

   Prefit gives you a chance to do any processing you want to do on
   images before fitting them.  A user supplied prefit should
   expect to be given a Data object and a Model object and should
   return a Data object to fit.  The prefit function should accept
   (but can ignore) arbitrary keyword arguments so that in the future
   we can pass more information into prefit.

:postfit (optional):

   postfit will be called with the FitResult object as an
   argument, should return a FitResult, and should also accept arbitrary
   keyword arguments for future proofing.  

Each frame fitted actually uses the Model from the previous fitresult.
This allows you to modify the initial guesses for the next frame, but
you can also swap out any arbitrary peices about the model if desired
(switch from Mie to Multisphere theory, change the number of particles
in a Spheres object, or anything else).  Just remember, with great
power comes great responsibility.
   
If postfit raises a :class:`holopy.fitting.series.RejectFit` or
subclass, the same data will be fit again.  The model for the retry
will be extracted from RejectFit.result_override if it is not None,
allowing you to set different guesses or model parameters.

