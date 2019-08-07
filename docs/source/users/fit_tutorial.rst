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
    from holopy.core.io import get_example_data_path, load_average
    from holopy.core.process import bg_correct, subimage, normalize
    from holopy.scattering import Sphere, Spheres, calc_holo
    from holopy.inference import (find_MAP, prior, ExactModel, CmaStrategy)

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

    fit_result = find_MAP(data_holo, guess_sphere, parameters=['x', 'y', 'z', 'r'])
    
The :func:`.find_MAP` function automatically runs :func:`.calc_holo` on many
different sets of parameter values to find the combination that gives the best
match to the experimental ``data_holo``, called the
*maximum a-posteriori probability* or MAP values of the fitting parameters. We
get back a :class:`.FitResult` object that knows how to summarize the results
of the fitting calculation in various ways, and can be saved to a file with
``hp.save`` :

..  testcode::

    best_fit_dictionary = fit_result.parameters
    best_fit_sphere = fit_result.scatterer
    initial_guess_hologram = fit_result.initial_guess
    best_fit_hologram = fit_result.best_fit
    best_fit_lnprob = fit_result.max_lnprob
    hp.save('results_file.h5', fit_result)

If we look at ``best_fit_dictionary`` or ``best_fit_sphere``, we see that our
initial guess of the sphere's position of (24, 22, 15) was corrected to
(24.17, 21.84, 16.42). Note that we have achieved sub-pixel position
resolution!


Customizing the model
~~~~~~~~~~~~~~~~~~~~~
Sometimes you might want a bit more control over how the parameters are varied.
You can customize the parameters with a :class:`.Model` object that describes
parameters as :class:`.Prior` objects instead of simply passing in your best
guess scatterer and the names of the parameters you wish to vary. For example,
we can set bounds on the coordinate parameters and and use a Gaussian prior
(std=0.05) for radius.

..  testcode::

    x = prior.Uniform(lower_bound=15, upper_bound=30, guess=24)
    y = prior.Uniform(15, 30, 22)
    z = prior.Uniform(10, 20)
    par_sphere = Sphere(n=1.58, r=prior.Gaussian(0.5, 0.05), center=[x, y, z])
    model = ExactModel(scatterer=par_sphere, calc_func=calc_holo)
    fit_result = find_MAP(data_holo, model)

Here we have used an :class:`.ExactModel` which takes a function ``calc_func``
to apply on the scatterer (we have used :func:`.calc_holo` here).
The :class:`.ExactModel` is the default for :func:`.find_MAP` but HoloPy also
includes :class:`.AlphaModel` and :class:`.PerfectLensModel`, which describe
specific models of hologram image formation. They take additional parameters
that allow for a more sophisticated calculation than a basic call to
:func:`.calc_holo`. Of course, these extra parameters can also be fit by
defining them as :class:`.Prior` objects.

The model in our example has read in some metadata from ``data_holo``
(illumination wavelength & polarization, medium refractive index, and image
noise level), but we could have specified those with keywords when defining our
:class:`.Model` instead.


Advanced Parameter Specification
--------------------------------
You can use the :class:`.Model` framework to more finely control parameters,
such as specifying a complex refractive index :

..  testcode::

    n = prior.ComplexPrior(real=prior.Gaussian(1.58, 0.02), imag=1e-4)

When this is used to define a :class:`.Sphere`, :func:`.find_MAP` will fit to
the real part of index of refraction while holding the imaginary part fixed.
You could fit it as well by specifying a :class:`.Prior` for ``imag``.

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


Sampling a Posterior
~~~~~~~~~~~~~~~~~~~~

Often, we aren't just interested in the best-fit (MAP) parameter values, but
in the full range of parameter values that provide a reasonable fit to an
observed hologram. This is best expressed as a Bayesian posterior distribution,
which we can sample with a Markov Chain Monte Carlo (MCMC) algorithm. The
approach and formalism used by HoloPy are described in more detail in
[Dimiduk2016]_. For more information on Bayesian inference in general,
see [Gregory2005]_.

A sampling calculation uses the same model and data as the fitting calculation
in the preceding section, but we replace the function :func:`.find_MAP` with
:func:`.sample_posterior` instead. Note that this calculation without further
modifications might take an unreasonably long time! There are some tips on how
to speed up the calculation further down on this page.

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
:meth:`.Model.generate_guess` to generate an initial sampling to pass in as
an initial guess.


Customizing the algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~
The :func:`.find_MAP` and :func:`sample_posterior` functions follow algorithms
that determine which sets of parameter values to simulate and compare to the
experimental data. You can specify a different algorithm by passing a
*strategy* keyword into either function. Options for :func:`.find_MAP`
currently include the default Levenberg-Marquardt (``strategy="nmpfit"``), as
well as cma-es (``strategy="cma"``) and scipy least squares
(``strategy="scipy lsq"``). Options for :func:`.sample_posterior` include the
default without tempering (``strategy="emcee"``) or tempering by changing the
number of pixels evaluated (``strategy="subset tempering"``) or Monte Carlo
temperature (``strategy="parallel tempering"``) [not currently implemented].

Each of these algorithms runs with a set of default values, but these may need
to be adjusted for your particular situation. For example, you may want to set
a random seed, control parallel computations, customize an initial guess, or
specify hyperparameters of the algorithm. To use non-default settings, you must
define an :class:`.InferenceStrategy` object for the algorithm you would like
to use. You can save the strategy to a file for use in future calculations or
modify it in place during an interactive session.

..  testcode::

    cma_fit_strategy = CmaStrategy(popsize=15, parallel=None)
    cma_fit_strategy.seed = 1234
    hp.save('cma_strategy_file.h5', cma_fit_strategy)
    strategy_result = cma_fit_strategy.find_MAP(data_holo, model)
    
Running the :meth:`.CmaStrategy.find_MAP` method is the same as calling
:func:`.find_MAP`, but with the option to customize how the algorithm runs. In
the example above, we have adjusted the ``popsize`` hyperparameter of the
cma-es algorithm, prevented the calculation from running as a paralell
computation, and set a random seed for reproducibility. The calculation returns
a :class:`.FitResult` object, just like a direct call to :func:`.find_MAP`.

Similarly, we can customize a MCMC computation to sample a posterior by
instantiating an appropriate class and calling its ``sample_posterior`` method.
Here we perform a MCMC calculation that uses only 500 pixels in the image and
runs for 2000 samples. We set the initial walker distribution to be one tenth
of the prior width.

..  testcode::

        emcee_strategy = EmceeStrategy(npixels=500, nsamples=2000)
        emcee_strategy.initial_distribution_scaling = 0.1
        hp.save('emcee_strategy_file.h5', emcee_strategy)
        emcee_result = emcee_strategy.sample_posterior(data_holo, model)

Random Subset Fitting
---------------------
In the most recent example, we only evaluated the holograms at the locations of
only 500 pixels in the experimental image. This is because a hologram usually
contains far more information than is needed to estimate your parameters of
interest. Because of this, you can often get a significantly faster fit with
little or no loss in accuracy by fitting to only a random fraction of the
pixels in a hologram. 

You will want to do some testing to make sure that you still get
acceptable answers with your data, but our investigations have shown
that you can frequently use random fractions of .1 or .01 with little
effect on your results and gain a speedup of 10x or greater.
