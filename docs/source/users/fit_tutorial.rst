.. _fit_tutorial:

**********************
Fitting Models to Data
**********************

The most powerful use of HoloPy is to analyze data by fitting a
scattering model to it.  This provides high precision measurements of
physical quantities from holographic or other types of data.

A fit generally consists of the following steps:

1. :ref:`Prepare <prepare_data>` your data for fitting.

2. :ref:`Specify a scattering model <define_model>` for the system your
   data was recorded from.

3. :ref:`Fit <run_fit>` the model to the data.

A Simple Fit
============

In the following, we fit for the position, radius, and refractive
index of a microsphere whose hologram is computed using known values
of these parameters. Fitting a model to a computed hologram is a
convenient test, as we know what answer we should get::

  import holopy as hp
  from holopy.core import ImageSchema, Optics
  from holopy.fitting import Model, par, fit
  from holopy.scattering.scatterer import Sphere
  from holopy.scattering.theory import Mie

  schema = ImageSchema(shape = 100, spacing = .1,
      optics = Optics(wavelen = .660, polarization = [1, 0],
      index = 1.33))
  sphere = Sphere(center = (5, 5, 10.3), r = .5, n = 1.58)
  data_holo = Mie.calc_holo(sphere, schema)

  par_s = Sphere(center = (par(guess = 5.5, limit = [4,10]),
       par(4.5, [4, 10]), par(10, [5, 15])), r = .5, n = 1.58)
  model = Model(par_s, Mie.calc_holo, alpha = par(.6, [.1, 1]))
  result = fit(model, data_holo)
  hp.save('result.yaml', result)


The first few lines import the HoloPy models that are needed to
compute and fit to holograms: ::

  from holopy.core import ImageSchema, Optics
  from holopy.fitting import Model, par, fit
  from holopy.scattering.scatterer import Sphere
  from holopy.scattering.theory import Mie

.. _prepare_data:

Preparing Data
--------------

Next, we compute the hologram for a microsphere using the same steps
as those in :ref:`calc_tutorial`::

  schema = ImageSchema(shape = 100, spacing = .1,
      optics = Optics(wavelen = .660, polarization = [1, 0],
      index = 1.33))
  sphere = Sphere(center = (5, 5, 10.3), r = .5, n = 1.58)
  data_holo = Mie.calc_holo(sphere, schema)

If you are working with your own data, it is important to remember to
normalize the data, since calculations return a normalized result. So
if you had ``data.tif`` and ``bg.tif`` you would use something like::

  import holopy as hp
  from holopy.core import Optics
  from holopy.core.process import normalize
  optics = Optics(wavelen = .660, polarization = [1, 0],
                  index = 1.33)
  data_holo = normalize(hp.load('data.tif', spacing = .1, optics = optics) /
                   hp.load('bg.tif', spacing = .1, optics = optics))

.. _define_model:

Define a Model
--------------

Next, we provide initial guesses for the three spatial coordinates
(`x`, `y`, and `z`, in that order) as bounded variable parameters in
``center``.  For instance, the initial guess for the `x` position of
the scatterer is 5.5 microns and bounded between 4 and 10 microns.
Guesses for the radius (``r``) and refractive index of the sphere
(``n``) are constant values also included in ``par_s``: ::

  par_s = Sphere(center = (par(guess = 5.5, limit = [4,10]),
      par(4.5, [4, 10]), par(10, [5, 15])), r = .5, n = 1.58)

Then this parametrized scatterer, along with a calculation theory, is
used to define a model::

   model = Model(par_s, Mie.calc_holo, alpha = par(.6, [.1, 1]))

``alpha`` is an additional fitting parameter first introduced by Lee
et al. in [Lee2007] (see :ref:`credits` for additional details).

To see how well the guess in your model lines up with the hologram you
are fitting to, use ::

  guess_holo = model.guess_holo(data_holo)

This will compute a hologram with the same dimensions as the data you
are attempting to fit. This simplest and best way to see what the
minimizer will be working from when it attempts to fit your hologram.

.. _run_fit:

Run the Fit
-----------

Once you have all of that set up, running the fit is almost
trivially simple::

  result = fit(model, data_holo)


You can examine the fitted position in ``result.scatterer.center``. We
see that the initial guess of the sphere's position (5.5, 4.5, 10.0)
was corrected by the fitter to (5.0,5.0,10.3). Success!

From the fit,
``result.scatterer`` gives the scatterer that best matches the hologram,
``result.alpha`` is the alpha for the best fit.  ``result.chisq`` and
``result.rsq`` are statistical measures of the the goodness of the fit.

You can also compute a hologram of the final fit result to compare to
the data with ::

  result_holo = result.fitted_holo(data_holo)

.. note::

   ``result.model`` and ``result.minimizer`` are the Model and
   Minimizer objects used in the fit, and ``result.minimization_info``
   contains any additional information the minimization algorithm
   returned about the minimization procedure (for
   :class:`~holopy.fitting.minimizer.Nmpfit` this includes things like
   covariance matrices).  Additional details are included in the
   documentation for :class:`.FitResult`.

Finally, we save the result with::

  hp.save('result.yaml', result)

This saves all of the information about the fit to a yaml text file.
These files are reasonably human readable and serve as our archive
format for data.  They can be loaded back into python with ::

  loaded_result = hp.load('result.yaml')

.. TODO additional examples require testing


Fitting Multiple Spheres
========================

In this example, we fit for the parameters of two spheres ::

    from holopy.scattering.scatterer import Sphere
    from holopy.scattering.scatterer import Spheres
    from holopy.scattering.theory import Mie
    from holopy.core import ImageSchema, Optics
    from holopy.fitting import Model, par, fit

    #calculate a hologram with known particle positions to do a fit against
    target = ImageSchema(shape = 256, spacing = .1,
        optics = Optics(wavelen = .660, index = 1.33, polarization = (1, 0)))

    s1 = Sphere(center=(15, 15, 20), n = 1.59, r = 0.5)
    s2 = Sphere(center=(14, 14, 20), n = 1.59, r = 0.5)
    cluster = Spheres([s1, s2])
    data_holo = Mie.calc_holo(cluster, target)

    #now do the fit
    guess1 = Sphere(center = (par(guess = 15.5, limit = [5,25]),
        par(14.5, [5, 25]), par(22, [5, 25])), r = .5, n = 1.59)
    guess2 = Sphere(center = (par(guess = 14.5, limit = [5,25]),
        par(13.5, [5, 25]), par(22, [5, 25])), r = .5, n = 1.59)
    par_s = Spheres([guess1,guess2])

    model = Model(par_s, Mie.calc_holo, alpha = par(.6, [.1, 1]))
    result = fit(model, data_holo)


Fitting a Time Series of Images
===============================

If you are taking video holograms (one of the most useful cases), you
will probably find yourself wanting to fit long timeseries of data.
This is done with :func:`.fit_series` ::

   from holopy.core import ImageSchema, Optics
   from holopy.fitting import Model, par, fit_series
   from holopy.scattering.scatterer import Sphere
   from holopy.scattering.theory import Mie

   schema = ImageSchema(shape = 100, spacing = .1,
       optics = Optics(wavelen = .660, polarization = [1, 0],
       index = 1.33))
   sphere1 = Sphere(center = (5, 5, 10.3), r = .5, n = 1.58)
   sphere2 = Sphere(center = (5, 5, 10.5), r = .5, n = 1.58)
   data_holos = [Mie.calc_holo(s, schema) for s in (sphere1, sphere2)]

   par_s = Sphere(center = (par(guess = 5.5, limit = [4,10]),
       par(4.5, [4, 10]), par(10, [5, 15])), r = .5, n = 1.58)
   model = Model(par_s, Mie.calc_holo, alpha = par(.6, [.1, 1]))
   results = fit_series(model, data_holos)

This is very similar to fit a single hologram (and this explanation
only calls out the differences), except instead we calculate and fit
two holograms. In the code below we define two spheres with the second
offset slightly (as if it was moving)::

   sphere1 = Sphere(center = (5, 5, 10.3), r = .5, n = 1.58)
   sphere2 = Sphere(center = (5, 5, 10.5), r = .5, n = 1.58)

And then compute two holograms from them using a `list comprehension
<http://docs.python.org/2/tutorial/datastructures.html#list-comprehensions>`_::

   data_holos = [Mie.calc_holo(s, schema) for s in (sphere1, sphere2)]

And finally, fit the holograms::

   results = fit_series(model, data_holos)

The results are a list of :class:`.FitResult` objects.

Speeding up Fits with Random Subset Fitting
===========================================

A hologram usually contains far more information than is needed to
determine the number of parameters you are interested in. Because of
this, you can often get a significantly faster fit with no little or
no loss in accuracy by fitting to only a random fraction of the pixels
in a hologram. ::

  result = fit(model, data, use_random_fraction=.1)

You will want to do some testing to make sure that you still get
acceptable answers with your data, but our investigations have shown
that you can frequently use random fractions of .1 or .01 with little
effect on your results and gain a speedup of 10x or greater.

Advanced Parameter Specification
================================

Complex Index of Refraction
---------------------------

You can specify a complex index with ::

  from holopy.fitting import ComplexParameter

  Sphere(n = ComplexParameter(real = par(1.58, step = 0.01), imag = 1e-4))

This will fit to the real part of index of refraction while holding
the imaginary part fixed.  You can fit to it as well by specifying
``imag = par(1e-4)`` instead of ``imag = 1e-4``. In a case like this
where we are providing a small imaginary part for numerical stability,
you would not want to fit to it. However fitting to an imaginary index
component could be useful for a metal particle. Setting the key word argument ``step = 0.01`` specifies the the step size used in calculating
the numerical derivatives of this parameter. Specifying a small step
size is often necessary when fitting for an index of refraction.

Tying Parameters
----------------

You may desire to fit holograms with *tied parameters*, in which
several physical quantities that could be varied independently are
constrained to have the same (but non-constant) value. A common
example involves fitting a model to a multi-particle hologram in which
all of the particles are constrained to have the same refractive
index, but the index is determined by the fitter.  This may be done by
defining a Parameter and using it in multiple places ::

  n1 = par(1.59)
  sc = Spheres([Sphere(n = nl, r = par(0.5e-6), \
    center = array([10., 10., 20.])), \
    Sphere(n = n1, r = par(0.5e-6), center = array([9., 11., 21.]))])

Telling the Minimizer More About a Parameter
--------------------------------------------

If you need to provide information to the minimizer about specific
parameters (for example a derivative step to nmp fit) you add them to
the par call as keyword args, for example ::

  Sphere(n = par(1.59, [1, 2], step = 1e-3), ...)


Custom Parametrization
----------------------

So far you have been specifying parametrizations of a scatterer by
directly specifying parameters within the scatterer. This is
the most convenient method, but sometimes it is not flexible enough.

If you need more control over how parameters define a scatterer,
HoloPy provides a lower level interface, the
:class:`.Parametrization` class. This allows you to do things like define a
cluster and fit by rotating it::

  from holopy.fitting import Parametrization
  s1 = Sphere(center=(15, 15, 20), n = 1.59, r = 0.5)
  s2 = Sphere(center=(14, 14, 20), n = 1.59, r = 0.5)
  cluster = Spheres([s1, s2])
  def make_scatterer(euler_alpha, euler_beta):
     return cluster.rotated(alpha = euler_alpha, beta = euler_beta)
  param = Parametrization(make_scatterer,
    parameters = [par(guess = 0, name = 'euler_alpha'),
    par(guess = 0, name = 'euler_beta')])

.. TODO fix rotations so that this example works

Here ``make_scatterer`` needs to be a function that takes keyword
arguments of the names of the parameters and returns a scatterer. In
this example, that is a function which rotates a reference cluster
through a given set of angles.


Using a Different Theory
========================

If you are fitting to a cluster of closely spaced spheres, you will
probably want to use the :class:`.Multisphere` theory instead of
Mie. This requires changing only the model from the `Fitting Multiple
Spheres`_ example to::

  model = Model(par_s, Multisphere.calc_holo, alpha = par(.6, [.1, 1]))

HoloPy is not limited to fitting holograms, you can change which
scattering calculation is used to compare with data. For example when
fitting against static light scattering data you might use a model
like one of these::

  model = Model(parametrization, Mie.calc_scat_matr)
  model = Model(parametrization, Mie.calc_scat_intensity)

Technically, you can use any function here as long as it takes a
scatterer and a :class:`.Schema` (and optionally additional keyword
arguments) as arguments and returns an :class:`.Marray` object.


Using a Different Minimizer
===========================

If you do not provide a minimizer, fits will default to using the
supplied :class:`.Nmpfit` minimizer with a set of sensible defaults.

You can choose another minimizer or provide non-default options to a
minimizer by passing a minimizer object to fit(), for example (to tell
nmpfit to use looser tolerances and a small iteration limit to get a
fast result to loosely check things out)::

  fit(model, data, minimizer = Nmpfit(ftol=1e-5, xtol = 1e-5,
                                      gtol=1e-5, maxiter=2))

or if you have OpenOpt and DerApproximator installed, you can use to
use one of OpenOpt's minimizers instead::

  fit(model, data, minimizer = OpenOpt(algorithm = 'ralg'))
