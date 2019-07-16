.. _fit_tutorial:

Fitting Models to Data
======================

As we have seen, we can use HoloPy to perform :ref:`calc_tutorial` from many
types of objects. Here, the goal is to compare these calculated holograms to a
recorded hologram, and adjust the parameters of the simulated scatterer to get
a good fit for the real hologram.

A Simple Least Squares Fit
~~~~~~~~~~~~~~~~~~~~~~~~~~

We start by loading and processing data using many of the functions outlined
in the tutorial on :ref:`load_tutorial`.

..  testcode::

    import holopy as hp
    from holopy.core.io import get_example_data_path, load_average
    from holopy.core.process import bg_correct, subimage, normalize
    from holopy.scattering import Sphere, Spheres, calc_holo
    from holopy.inference import (least_squares_fit, point_estimate,
                                  prior, ExactModel, CmaStrategy)

    # load an image
    imagepath = get_example_data_path('image01.jpg')
    raw_holo = hp.load_image(imagepath, spacing = 0.0851, medium_index = 1.33, illum_wavelen = 0.66, illum_polarization = (1,0))
    bgpath = get_example_data_path(['bg01.jpg','bg02.jpg','bg03.jpg'])
    bg = load_average(bgpath, refimg = raw_holo)
    data_holo = bg_correct(raw_holo, bg)

    # process the image
    data_holo = subimage(data_holo, [250,250], 200)
    data_holo = normalize(data_holo)


Next we define a scatterer that we wish to model as our initial guess. We can
calculate the hologram that it would produce if it were placed in our
experimental setup, as in the previous tutorial on :ref:`calc_tutorial`.
Fitting works best if your initial guess is close to the correct result. You
can find guesses for `x` and `y` coordinates with :func:`.center_find`, and
guess `z` with :func:`.propagate`.

..  testcode::

    guess_sphere = Sphere(n=1.58, r=0.5, center=[24,22,15])
    initial_guess = calc_holo(data_holo, guess_sphere)
    hp.show(data_holo)
    hp.show(initial_guess)

Finally, we can adjust the parameters of the sphere in order to get a good fit
to the data. Here we adjust the center coordinates (x, y, z) of the sphere and
its radius, but hold its refractive index fixed.

..  testcode::

    ls_sphere = least_squares_fit(s, data_holo, parameters = ['x','y','z','r'])
    ls_hologram = calc_holo(data_holo, ls_sphere)
    
The initial guess of the sphere's position (24, 22, 15) was corrected by
fitting to (24.17,21.84,16.42). Note that we have achieved sub-pixel position
resolution!

.. _point_estimate:
Customizing the model
~~~~~~~~~~~~~~~~~~~~~
Sometimes you might want a bit more control over how the parameters are varied.
You can customize the parameters with a :class:`.Model` object that describes
parameters as :class:`.Prior` objects. For example, we can set bounds on the
coordinate parameters and and use a Gaussian prior (std=0.05) for radius.

..  testcode::

    x = prior.Uniform(lower_bound=15, upper_bound=30, guess=24)
    y = prior.Uniform(15, 30, 22)
    z = prior.Uniform(10, 20)
    par_sphere = Sphere(n=1.58, r=prior.Gaussian(0.5, 0.05), center=[x, y, z])
    model = ExactModel(scatterer=par_sphere, calc_func=calc_holo)
    fit_result = point_estimate(model, data_holo)
    
Here we have used an :class:`.ExactModel` which takes a function ``calc_func``
to apply on the scatterer (we have used :func:`.calc_holo` here). HoloPy also
includes :class:`.AlphaModel` and :class:`.PerfectLensModel` which describe
specific models of hologram image formation. They take additional parameters
that allow for a more sophisticated model than a basic call to
:func:`.calc_holo`. Of course, these extra parameters can also be fit by
defining them as :class:`.Prior` objects.

The model in our example has read some metadata from ``data_holo``
(illumination wavelength & polarization, medium refractive index, and image
noise level), but we could have specified those with keywords when defining the
:class:`.Model` instead.

The calculation :func:`.point_estimate` returns a :class:`.FitResult` object.
It knows how to summarize the results of the fitting calculation in various
ways, or you can save it to a file with ``hp.save`` :

..  testcode::

    best_fit_dictionary = fit_result.parameters
    best_fit_sphere = fit_result.scatterer
    best_fit_hologram = fit_result.best_fit
    best_fit_lnprob = fit_result.max_lnprob
    hp.save('results_file.h5', fit_result)  

Sampling a Posterior
~~~~~~~~~~~~~~~~~~~~

Often, we aren't just interested in the best-fit parameter values, but in the
full range of parameter values that provide a reasonable fit to an observed
hologram. This is best expressed as a Bayesian posterior distribution, which we
can sample with a Markov Chain Monte Carlo (MCMC) algorithm. The approach and
formalism used by HoloPy are described in more detail in [Dimiduk2016]_. For
more information on Bayesian inference in general, see [Gregory2005]_.

A sampling calculation uses the same model and data as the point estimate
calculation in the preceding seciont, but we replace the function
:func:`.point_estimate` with :func:`.sample_posterior` instead. Note that this
calculation without further modifications might take an unreasonably long time!
There are some tips on how to speed up the calculation further down on this
page.

The :func:`.sample_posterior` calculation returns a :class:`.SamplingResult`
object, which is similar to the :class:`.FitResult` returned by
:func:`.point_estimate`, but with some additional features. We can access the
sampled parameter values and calculated log-probabilities with
:attr:`.SamplingResult.samples` and :attr:`.SamplingResult.lnprobs`,
respectively. Usually, the MCMC samples will take some steps to converge or
"burn-in" to a stationary distribution from your initial guess. You can remove
these samples with the built-in method :meth:`.SamplingResult.burn_in`, which
returns a new :class:`.SamplingResult` with only the burned-in samples. To
reduce the burn in time, provide an initial guess position and width that is as
close as possible to the eventual posterior distribution. You can use 
:meth:`.BaseModel.generate_guess` to calculate an initial sampling for passing
in as an initial guess.

Customizing the algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~
By default, :func:`.point_estimate` uses the same fitting algorithm as
:func:`.least_squares_fit` and :func:`.sample_posterior` doesn't use any sort
of tempering. You can specify different algorithms  by passing a *method*
keyword into either function. Options for :func:`.point_estimate` currently
include cma-es (``method='cma'``) and scipy least squares (``method='scipy'``).
Options for :func:`.sample_posterior` include tempering by changing the number
of pixels (``method='subset tempering'``) or temperature
(``method='parallel tempering'``) [not currently implemented].

For even more control over the fitting process you can define an
:class:`.InferenceStrategy` object for the algorithm you would like to use.
This lets you specify hyperparameters, set a random seed, or control parallel
computations. You can save the strategy to a file for use in future
calculations or modify it in place during an interactive session.

..  testcode::

    cma_fit_strategy = CmaStrategy(popsize=15, parallel=None)
    cma_fit_strategy.seed = 1234
    hp.save('strategy_file.h5', cma_fit_strategy)
    strategy_result = cma_fit_strategy.fit(model, data_holo)
    
The method of the :meth:`.CmaStrategy.fit` returns a :class:`.FitResult`
object just like we saw from :func:`.point_estimate`. Similarly,
:meth:`.EmceeStrategy.sample` returns a :class:`.SamplingResult` object. 

Advanced Parameter Specification
--------------------------------
You can use the :class:`.Model` framework to more finely control parameters,
such as specifying a complex refractive index :

..  testcode::

  n = prior.ComplexPrior(real=prior.Gaussian(1.58, 0.02), imag=1e-4)
  
When this is used to define a :class:`.Sphere`, :func:`.point_estimate` will
fit to the real part of index of refraction while holding the imaginary part
fixed. You could fit it as well by specifying a :class:`.Prior` for ``imag``.

You may desire to fit holograms with *tied parameters*, in which
several physical quantities that could be varied independently are
constrained to have the same (but non-constant) value. A common
example involves fitting a model to a multi-particle hologram in which
all of the particles are constrained to have the same refractive
index, but the index is determined by the fitter.  This may be done by
defining a parameter and using it in multiple places.

..  testcode::

  n1 = prior.Gaussian(1.58, 0.02)
  sphere_cluster = Spheres([
    Sphere(n = n1, r = 0.5, center = [10., 10., 20.]),
    Sphere(n = n1, r = 0.5, center = [9., 11., 21.])])

Random Subset Fitting
---------------------
A hologram usually contains far more information than is needed to
determine the number of parameters you are interested in. Because of
this, you can often get a significantly faster fit with no little or
no loss in accuracy by fitting to only a random fraction of the pixels
in a hologram. ``data_holo`` is a 200x200 pixel array, but we can fit to just
2000 pixels from the image.

..  testcode::

  subset_result = point_estimate(model, data_holo, npixels=2000)

You will want to do some testing to make sure that you still get
acceptable answers with your data, but our investigations have shown
that you can frequently use random fractions of .1 or .01 with little
effect on your results and gain a speedup of 10x or greater.
