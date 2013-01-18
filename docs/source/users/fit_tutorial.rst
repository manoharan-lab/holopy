**********************
Fitting Models to Data
**********************

The most powerful use of HoloPy is to analyze data by fitting a
scattering model to it.  This provides high precision measurements of
physical quantities from holographic or other types of data.

Fitting requires a :class:`.Model` of the scattering system and
:class:`.Marray` containing data the fit is trying to match (most
commonly an :class:`.Image`).

A Simple Fit
============

In the following, we fit for the position, radius, and refractive
index of a microsphere whose hologram is computed using known values
of these parameters::

   from holopy.core import ImageSchema, Optics
   from holopy.fitting import Model, par, fit
   from holopy.scattering.scatterer import Sphere
   from holopy.scattering.theory import Mie

   schema = ImageSchema(shape = 100, spacing = .1,
      optics = Optics(wavelen = .660, polarization = [1, 0],  
      index = 1.33))
   sphere = Sphere(center = (5, 5, 10.3), r = .5, n = 1.58)
   holo = Mie.calc_holo(sphere, schema) 

   par_s = Sphere(center = (par(guess = 5.5, limit = [4,10]), 
       par(4.5, [4, 10]), par(10, [5, 15])), r = .5, n = 1.58) 
   model = Model(par_s, Mie.calc_holo, alpha = par(.6, [.1, 1])) 
   result = fit(model, holo)
   holopy.save('result.yaml', result)

The first few lines import the HoloPy models that are needed to
compute and fit for holograms: ::

  from holopy.core import ImageSchema, Optics
  from holopy.fitting import Model, par, fit
  from holopy.scattering.scatterer import Sphere
  from holopy.scattering.theory import Mie

Next, we compute the hologram for a microsphere using the same steps
as those in :ref:`calc_tutorial`::

  schema = ImageSchema(shape = 100, spacing = .1,
      optics = Optics(wavelen = .660, polarization = [1, 0],  
      index = 1.33))
  sphere = Sphere(center = (5, 5, 10.3), r = .5, n = 1.58)
  holo = Mie.calc_holo(sphere, schema) 

Next, we provide initial guesses for the three spatial coordinates
(`x`, `y`, and `z`, in that order) as bounded variable parameters in
``center``.  For instance, the initial guess for the `x` position of
the scatterer is 5.5 microns and bounded between 4 and 10 microns.
Guesses for the radius (``r``) and refractive index of the sphere
(``n``) are constant values included also in ``par_s``: ::

  par_s = Sphere(center = (par(guess = 5.5, limit = [4,10]), 
      par(4.5, [4, 10]), par(10, [5, 15])), r = .5, n = 1.58) 

Then, using the scattering model specified in ``model`` where
``alpha`` is a fitting parameter first introduced by Lee et al. in
[Lee2007] (see :ref:`credits` for additional details), the ``fit``
function performs the fit to the hologram computed in `holo`: ::
  
  result = fit(model, holo)

After performing the fit, you can inspect the fitted values for `n`,
`r`, and the position given by ``center`` in
``result.scatterer``. Note that the output values for ``center`` do
not give the spatial coordinates of the fitted scatterer.  We see that
the initial guess of the sphere's position (5.5, 4.5, 10.0) was
corrected by the fitter to (5.0,5.0,10.3). Success!

From the fit,
``result.scatterer`` gives the scatterer that best matches the hologram,
``result.alpha`` is the alpha for the best fit.  ``result.chisq`` and
``result.rsq`` are statistical measures of the the goodness of the fit.

.. note::

        ``result.model`` and ``result.minimizer`` are the Model and
        Minimizer objects used in the fit, and
        ``result.minimization_info`` contains any additional
        information the minimization algorithm returned about the
        minimization procedure (for
        :class:`~holopy.fitting.minimizer.Nmpfit` this includes things
        like covariance matrices).  Additional details are included in
        the documentation for :class:`.FitResult`.

Finally, we save the result with::

  holopy.save('result.yaml', result)

This saves all of the information about the fit to a yaml text file.
These files are reasonably human readable and serve as our archive
format for data.  They can be loaded back into python with ::

  loaded_result = holopy.load('result.yaml')

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
    holo = Mie.calc_holo(cluster, target)

    #now do the fit
    guess1 = Sphere(center = (par(guess = 15.5, limit = [5,25]), 
        par(14.5, [5, 25]), par(22, [5, 25])), r = .5, n = 1.59)
    guess2 = Sphere(center = (par(guess = 14.5, limit = [5,25]), 
        par(13.5, [5, 25]), par(22, [5, 25])), r = .5, n = 1.59)
    par_s = Spheres([guess1,guess2])

    model = Model(par_s, Mie.calc_holo, alpha = par(.6, [.1, 1]))
    result = fit(model, holo)


Fitting a Time Series of Images
===============================

If you are taking video holograms (one of the most useful cases), you
will probably find yourself wanting to fit long timeseries of data.
This is done with :func:`holopy.fitting.fit.fit_series` ::

   from holopy.core import ImageSchema, Optics
   from holopy.fitting import Model, par, fit_series
   from holopy.scattering.scatterer import Sphere
   from holopy.scattering.theory import Mie

   schema = ImageSchema(shape = 100, spacing = .1,
       optics = Optics(wavelen = .660, polarization = [1, 0], 
       index = 1.33))
   sphere1 = Sphere(center = (5, 5, 10.3), r = .5, n = 1.58)
   sphere2 = Sphere(center = (5, 5, 10.5), r = .5, n = 1.58)
   holos = [Mie.calc_holo(s, schema) for s in (sphere1, sphere2)]

   par_s = Sphere(center = (par(guess = 5.5, limit = [4,10]), 
       par(4.5, [4, 10]), par(10, [5, 15])), r = .5, n = 1.58)
   model = Model(par_s, Mie.calc_holo, alpha = par(.6, [.1, 1])) 
   results = fit_series(model, holos)
   
This is very similar to fit a single hologram (and this explanation
only calls out the differences), except instead we calculate and fit
two holograms. In the code below we define two spheres with the second
offset slightly (as if it was moving).::

   sphere1 = Sphere(center = (5, 5, 10.3), r = .5, n = 1.58)
   sphere2 = Sphere(center = (5, 5, 10.5), r = .5, n = 1.58)

And then compute two holograms from them using a `list comprehension
<http://docs.python.org/2/tutorial/datastructures.html#list-comprehensions>`_.::

   holos = [Mie.calc_holo(s, schema) for s in (sphere1, sphere2)]

And finally, fit the holograms::

   results = fit_series(model, holos)

the results are a list of :class:`.FitResult` objects. 


Advanced Parameter Specification
================================

Complex Index of Refraction
---------------------------
  
You can specify a complex index with ::

  Sphere(n = ComplexParameter(real = par(1.58), imag = 1e-4))

This will fit to the real part of index of refraction while holding
the imaginary part fixed.  You can fit to it as well by specifying a
Parameter instead of a fixed number there.


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
directly directly specifying parameters within the scatterer. This is
the most convenient method, but sometimes it is not flexible enough.

If you need more control over how parameters define a scatterer,
HoloPy provides a lower level interface the
:class:`.Parametrization`. This allows you to do things like define a
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

Here make_scatterer needs to be a function that takes keyword
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

  model = Model(paremetrization, Mie.calc_scat_matr)
  model = Model(paremetrization, Mie.calc_scat_intensity)

Technically, you can use any function here as long as it takes a
scatterer and a :class:`.Schema` (and optionally additional keyword
arguments) as arguments and returns an :class:`.Marray` object.


Using a Different Minimizer
===========================

If you do not provide a minimizer, fits will default to using the
supplied Nmpfit minimizer with a set of sensible defaults. 

You can choose another minimizer or provide non-default options to a
minimizer by passing a minimizer object to fit(), for example (To tell
nmpfit to use looser tolerances and a small iteration limit (to get a
fast result to check things out).)::

  fit(model, data, minimizer = Nmpfit(ftol=1e-5, xtol = 1e-5, gtol=1e-5, niter=2))

or to use OpenOpt's ralg minimizer instead of nmpfit (This will fail
unless you have OpenOpt installed and configured so that HoloPy can
find it.)::

  fit(model, data, minimizer = Ralg())
  
