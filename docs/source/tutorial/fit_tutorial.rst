.. _fit_tutorial:

Fitting Models to Data
======================

As we have seen, we can use HoloPy to perform :ref:`calc_tutorial` from many
types of objects. Here, the goal is to compare these calculated holograms to a
recorded experimental hologram, and adjust the parameters of the simulated
scatterer to get a good fit for the real hologram.


A Simple Least Squares Fit
~~~~~~~~~~~~~~~~~~~~~~~~~~

We start by loading and processing data using many of the functions outlined
in the tutorial on :ref:`load_tutorial`.

..  testcode::

    import holopy as hp
    import numpy as np
    from holopy.core.io import get_example_data_path, load_average
    from holopy.core.process import bg_correct, subimage, normalize
    from holopy.scattering import Sphere, Spheres, calc_holo
    from holopy.inference import prior, ExactModel, CmaStrategy, EmceeStrategy

    # load an image
    imagepath = get_example_data_path('image01.jpg')
    raw_holo = hp.load_image(imagepath, spacing = 0.0851, medium_index = 1.33,
                             illum_wavelen = 0.66, illum_polarization = (1,0))
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
can find guesses for `x` and `y` coordinates with :func:`.center_find`, and a
guess for `z` with :func:`.propagate`.

..  testcode::

    guess_sphere = Sphere(n=1.58, r=0.5, center=[24,22,15])
    initial_guess = calc_holo(data_holo, guess_sphere)
    hp.show(data_holo)
    hp.show(initial_guess)

Finally, we can adjust the parameters of the sphere in order to get a good fit
to the data. Here we adjust the center coordinates (x, y, z) of the sphere and
its radius, but hold its refractive index fixed. By default :func:`.fit` will
adjust all parameters, so we can omit the argument if that is what we want.

..  testcode::

    parameters_to_fit = ['x', 'y', 'z', 'r']
    fit_result = hp.fit(data_holo, guess_sphere, parameters=parameters_to_fit)

The :func:`.fit` function automatically runs :func:`.calc_holo` on many
different sets of parameter values to find the combination that gives the best
match to the experimental ``data_holo``. We get back a :class:`.FitResult`
object that knows how to summarize the results of the fitting calculation in
various ways, and can be saved to a file with :func:`.hp.save`` :

..  testcode::

    best_fit_values = fit_result.parameters
    initial_guess_values = fit_result.guess_parameters
    best_fit_sphere = fit_result.scatterer
    best_fit_hologram = fit_result.hologram
    best_fit_lnprob = fit_result.max_lnprob
    hp.save('results_file.h5', fit_result)

If we look at ``best_fit_values`` or ``best_fit_sphere``, we see that our
initial guess of the sphere's position of (24, 22, 15) was corrected to
(24.16, 21.84, 16.35). Note that we have achieved sub-pixel position
resolution!


Customizing the model
~~~~~~~~~~~~~~~~~~~~~
Sometimes you might want a bit more control over how the parameters are varied.
You can customize the parameters with a :class:`.Model` object that describes
parameters as :class:`.Prior` objects instead of simply passing in your best
guess scatterer and the names of the parameters you wish to vary. For example,
we can set bounds on the coordinate parameters and use a Gaussian prior for the
radius - here, with a mean of 0.5 and standard deviation of 0.05 micrometers.

..  testcode::

    x = prior.Uniform(lower_bound=15, upper_bound=30, guess=24)
    y = prior.Uniform(15, 30, 22)
    z = prior.Uniform(10, 20)
    par_sphere = Sphere(n=1.58, r=prior.Gaussian(0.5, 0.05), center=[x, y, z])
    model = ExactModel(scatterer=par_sphere, calc_func=calc_holo)
    fit_result = hp.fit(data_holo, model)

Here we have used an :class:`.ExactModel` which takes a function ``calc_func``
to apply on the :class:`.Scatterer` (we have used :func:`.calc_holo` here).
The :class:`.ExactModel` isn't actually the default when we call :func:`.fit`
directly. Instead, HoloPy uses an :class:`.AlphaModel`, which includes an
additional fitting parameter to control the hologram contrast intensity - the
same as calling :func:`.calc_holo` with a `scaling` argument. You can
fit for the extra parameters in these models by defining them as
:class:`.Prior` objects. Likewise, if the scattering theory you are
using requires fittable parameters (such as the `lens_angle` for the
:class:`.MieLens` theory or the `spherical_aberration` for the
:class:`.AberratedMieLens` theory), you can fit for these by defining
them as :class:`.Prior` objects as well.

The model in our example has read in some metadata from ``data_holo``
(illumination wavelength & polarization, medium refractive index, and image
noise level). If we want to override those values, or if we loaded an image
without specifying metadata, we can pass them directly into the
:class:`.Model` object by using keywords when defining it.


Advanced Parameter Specification
--------------------------------
You can use the :class:`.Model` framework to more finely control parameters,
such as specifying a complex refractive index :

..  testcode::

    n = prior.ComplexPrior(real=prior.Gaussian(1.58, 0.02), imag=1e-4)

When this refractive index is used to define a :class:`.Sphere`, :func:`.fit`
will fit to the real part of index of refraction while holding the imaginary
part fixed. You could fit it as well by specifying a :class:`.Prior` for
``imag``.

You may desire to fit holograms with *tied parameters*, in which
several physical quantities that could be varied independently are
constrained to have the same (but non-constant) value. A common
example involves fitting a model to a multi-particle hologram in which
all of the particles are constrained to have the same refractive
index, but the index is determined by the fitter. This may be done by
defining a parameter and using it in multiple places. Other tools for handling
tied parameters are described in the user guide on :ref:`scatterers_user`.

..  testcode::

    n1 = prior.Gaussian(1.58, 0.02)
    sphere_cluster = Spheres([
    Sphere(n = n1, r = 0.5, center = [10., 10., 20.]),
    Sphere(n = n1, r = 0.5, center = [9., 11., 21.])])


.. _transformed_priors:

Transforming Priors
~~~~~~~~~~~~~~~~~~~
Sometimes you might want to apply mathematical operations to transform one
prior into another, for example in the following use cases:

* You want two parameters to vary together with values that are related but
  unequal, such as the length and radius of a cylinder with known aspect ratio,
  or the z-coordinates of two vertically stacked particles.

* You want a parameter that is distrbuted according to some other distribution,
  such as a cylinder axis evenly distributed in spherical coordinates or a
  sphere with uniformly distributed volume (not radius).

* You want to reparamaterize a problem to reduce covariances between fitting
  parameters, such as finding the separation distance between two closely
  interacting particles or to find the positions of particles confined to a
  plane with a slight tilt as compared to the x-y plane.

If you have an explicit transformation function you can use it to define a
:class:`.TransformedPrior` object, for example to define a polar angle with
the appropriate distribution we might do:

..  testcode::

    def uniform2polar(u):
        # we want polar angle \theta distributed according to sin(\theta)
        # Use inverse transform sampling, correct for \theta in [0, pi]
        symmetric_polar = np.arcsin(u)
        return symmetric_polar + np.pi/2
    director = prior.Uniform(-1, 1, name='director')
    polar_angle = prior.TransformedPrior(uniform2polar, director, name='polar')

You can use :class:`.TransformedPrior` objects when defining a
:class:`.Scatterer` just like regular priors. They share some attributes with
basic priors as well, such as the :meth:`.TransformedPrior.sample` method, but
you cannot directly calculate probabilities or log-probabilities of a
:class:`.TransformedPrior` taking on a particular value.

Besides explicitly defining them, you can also create
:class:`.TransformedPrior` objects by using numpy ufuncs and built-in operators
on priors:

..  testcode::

    sphere_area = prior.Uniform(1, 2, name='base_area')
    diameter = np.sqrt(sphere_area / np.pi)
    radius = diameter / 2
    shell = radius + 0.1

All of our derived objects are :class:`.TransformedPrior` objects, even though
we didn't explicitly define them that way. If we use more than one of
``sphere_area``, ``diameter``, ``radius``, or ``shell`` in a fitting or
inference calculation, they will all be derived from a single parameter
(``sphere_area`` in this case) even though they take on different values. Note
that we could have expressed ``shell`` in one line if we didn't care about the
intermediate values:

..  testcode::

    shell = np.sqrt(prior.Uniform(1, 2, name='base_area')) / 2 + 0.1
    shell.name = 'shell'

It's always a good idea to assign your priors names when working with
:class:`.TransformedPrior` objects to keep track of their relationships.
Parameter names will be generated if none are provided but they might not be
very informative. To help with naming, you can even assign names to
:class:`.TransformedPrior` objects when using numpy ufuncs!

..  testcode::

    diameter = np.sqrt(sphere_area / np.pi, name='diameter')


.. _infer_tutorial:

Bayesian Parameter Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Often, we aren't just interested in the best-fit (MAP) parameter values, but
in the full range of parameter values that provide a reasonable fit to an
observed hologram. This is best expressed as a Bayesian posterior distribution,
which we can sample with a Markov Chain Monte Carlo (MCMC) algorithm. The
approach and formalism used by HoloPy are described in more detail in
[Dimiduk2016]_. For more information on Bayesian inference in general,
see [Gregory2010]_.

A sampling calculation uses the same model and data as the fitting calculation
in the preceding section, but we replace the function :func:`.fit` with
:func:`.sample` instead. Note that this calculation without further
modifications might take an unreasonably long time! There are some tips on how
to speed up the calculation further down on this page.

The :func:`.sample` calculation returns a :class:`.SamplingResult`
object, which is similar to the :class:`.FitResult` returned by
:func:`.fit`, but with some additional features. We can access the
sampled parameter values and calculated log-probabilities with
:attr:`.SamplingResult.samples` and :attr:`.SamplingResult.lnprobs`,
respectively. Usually, the MCMC samples will take some steps to converge or
"burn-in" to a stationary distribution from your initial guess. This is most
easily seen in the values of :attr:`.SamplingResult.lnprobs`, which will
rise at first and then fluctuate around a stationary value after having burned
in. You can remove the early samples with the built-in method
:meth:`.SamplingResult.burn_in`, which returns a new :class:`.SamplingResult`
with only the burned-in samples.

Customizing the algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~
The :func:`.fit` and :func:`.sample` functions follow algorithms that determine
which sets of parameter values to simulate and compare to the experimental
data. You can specify a different algorithm by passing a *strategy* keyword
into either function. Options for :func:`.fit` currently include the default
Levenberg-Marquardt (``strategy="nmpfit"``), as well as cma-es
(``strategy="cma"``) and scipy least squares (``strategy="scipy lsq"``).
Options for :func:`.sample` include the default without tempering
(``strategy="emcee"``), tempering by changing the number of pixels evaluated
(``strategy="subset tempering"``), or parallel tempered MCMC
(``strategy="parallel tempering"``) [not currently implemented]. You can see
the available strategies in your version of HoloPy by calling
``hp.inference.available_fit_strategies`` or
``hp.inference.available_sampling_strategies``.

Each of these algorithms runs with a set of default values, but these may need
to be adjusted for your particular situation. For example, you may want to set
a random seed, control parallel computations, customize an initial guess, or
specify hyperparameters of the algorithm. To use non-default settings, you must
define a *Strategy* object for the algorithm you would like to use. You can
save the strategy to a file for use in future calculations or modify it during
an interactive session. ::

    cma_fit_strategy = CmaStrategy(popsize=15, parallel=None)
    cma_fit_strategy.seed = 1234
    hp.save('cma_strategy_file.h5', cma_fit_strategy)
    strategy_result = fit(data_holo, model, cma_fit_strategy)

In the example above, we have adjusted
the ``popsize`` hyperparameter of the cma-es algorithm, prevented the
calculation from running as a parallel computation, and set a random seed for
reproducibility. The calculation returns a :class:`.FitResult` object, just
like a direct call to :func:`.fit`.

Similarly, we can customize a MCMC computation to sample a posterior by calling
:func:`.sample` with a :class:`.EmceeStrategy` object. Here we perform a
MCMC calculation that uses only 500 pixels from the image and runs 50 walkers
each for 2000 samples. We set the initial walker distribution to be one tenth
of the prior width. In general, the burn-in time for a MCMC calculation will
be reduced if you provide an initial guess position and width that is as close
as possible to the eventual posterior distribution. You can use
:meth:`.Model.generate_guess` to generate an initial sampling to pass in as an
initial guess to your :class:`.EmceeStrategy` object. ::

        nwalkers = 50
        initial_guess = model.generate_guess(nwalkers, scaling=0.1)
        emcee_strategy = EmceeStrategy(npixels=500, nwalkers=nwalkers,
            nsamples=2000, walker_initial_pos=initial_guess)
        hp.save('emcee_strategy_file.h5', emcee_strategy)
        emcee_result = hp.sample(data_holo, model, emcee_strategy)

Random Subset Fitting
---------------------
In the most recent example, we evaluated the holograms at the locations of only
500 pixels in the experimental image. This is because a hologram usually
contains far more information than is needed to estimate your parameters of
interest. You can often get a significantly faster fit with little or no loss
in accuracy by fitting to only a random fraction of the pixels in a hologram.

You will want to do some testing to make sure that you still get
acceptable answers with your data, but our investigations have shown
that you can frequently use random fractions of 0.1 or 0.01 with little
effect on your results and gain a speedup of 10x or greater.
