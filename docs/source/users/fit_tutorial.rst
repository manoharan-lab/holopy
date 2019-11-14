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
    from holopy.inference import (
        fit, sample, prior, ExactModel, CmaStrategy, EmceeStrategy)

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
its radius, but hold its refractive index fixed.

..  testcode::

    fit_result = fit(data_holo, guess_sphere, parameters=['x', 'y', 'z', 'r'])

..  testoutput::
    :options: +NORMALIZE_WHITESPACE
    
    Iter       1    CHI-SQUARE =  122.2991772  DOF =  40000
       r = 1  
       center.0 = 1  
       center.1 = 1  
       center.2 = 1  
    Iter       2    CHI-SQUARE =  73.1743179  DOF =  40000
       r = 0.8985179252  
       center.0 = 1.003214606  
       center.1 = 0.9978521719  
       center.2 = 1.014361104  
    Iter       3    CHI-SQUARE =  40.63411872  DOF =  40000
       r = 0.9073623251  
       center.0 = 1.005863599  
       center.1 = 0.9957062809  
       center.2 = 1.045412721  
    Iter       4    CHI-SQUARE =  24.45376276  DOF =  40000
       r = 0.9452170024  
       center.0 = 1.006911124  
       center.1 = 0.9939173215  
       center.2 = 1.071988733  
    Iter       5    CHI-SQUARE =  21.16623209  DOF =  40000
       r = 0.958813881  
       center.0 = 1.006953814  
       center.1 = 0.9931590158  
       center.2 = 1.084211067  
    Iter       6    CHI-SQUARE =  20.74045935  DOF =  40000
       r = 0.9589420606  
       center.0 = 1.006880604  
       center.1 = 0.9929427187  
       center.2 = 1.088112117  
    Iter       7    CHI-SQUARE =  20.68330482  DOF =  40000
       r = 0.9574523224  
       center.0 = 1.006838202  
       center.1 = 0.9928860496  
       center.2 = 1.089302504  
    Iter       8    CHI-SQUARE =  20.67511947  DOF =  40000
       r = 0.9565873408  
       center.0 = 1.006819234  
       center.1 = 0.9928712692  
       center.2 = 1.089696776  
    Iter       9    CHI-SQUARE =  20.67392811  DOF =  40000
       r = 0.9562066479  
       center.0 = 1.006811419  
       center.1 = 0.9928673007  
       center.2 = 1.089836786  
    Iter      10    CHI-SQUARE =  20.6737543  DOF =  40000
       r = 0.9560531311  
       center.0 = 1.006808321  
       center.1 = 0.9928661771  
       center.2 = 1.089888524  
    Iter      11    CHI-SQUARE =  20.67372897  DOF =  40000
       r = 0.955993013  
       center.0 = 1.006807111  
       center.1 = 0.9928658359  
       center.2 = 1.089907898  
    Iter      12    CHI-SQUARE =  20.67372525  DOF =  40000
       r = 0.9559701628  
       center.0 = 1.006806653  
       center.1 = 0.992865729  
       center.2 = 1.089915406  
    Iter      13    CHI-SQUARE =  20.67372471  DOF =  40000
       r = 0.9559613745  
       center.0 = 1.006806477  
       center.1 = 0.9928656913  
       center.2 = 1.08991827  
    Iter      14    CHI-SQUARE =  20.67372464  DOF =  40000
       r = 0.9559575958  
       center.0 = 1.0068064  
       center.1 = 0.9928656735  
       center.2 = 1.089919146  
    Iter      15    CHI-SQUARE =  20.67372462  DOF =  40000
       r = 0.9559565899  
       center.0 = 1.006806381  
       center.1 = 0.992865673  
       center.2 = 1.089919683  
    
The :func:`.fit` function automatically runs :func:`.calc_holo` on many
different sets of parameter values to find the combination that gives the best
match to the experimental ``data_holo``. We get back a :class:`.FitResult`
object that knows how to summarize the results of the fitting calculation in
various ways, and can be saved to a file with ``hp.save`` :

..  testcode::

    best_fit_dictionary = fit_result.parameters
    best_fit_sphere = fit_result.scatterer
    initial_guess_values = fit_result.initial_guess
    best_fit_hologram = fit_result.best_fit
    best_fit_lnprob = fit_result.max_lnprob
    hp.save('results_file.h5', fit_result)

If we look at ``best_fit_dictionary`` or ``best_fit_sphere``, we see that our
initial guess of the sphere's position of (24, 22, 15) was corrected to
(24.16, 21.84, 16.35). Note that we have achieved sub-pixel position
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
    fit_result = fit(data_holo, model)

..  testoutput::
    :options: +NORMALIZE_WHITESPACE

    Iter       1    CHI-SQUARE =  2427226.876  DOF =  40000
       r = 1  
       center.0 = 1  
       center.1 = 1  
       center.2 = 1  
    Iter       2    CHI-SQUARE =  1452264.079  DOF =  40000
       r = 0.8985182479  
       center.0 = 1.003214606  
       center.1 = 0.9978521719  
       center.2 = 1.014361151  
    Iter       3    CHI-SQUARE =  806449.2325  DOF =  40000
       r = 0.9073632039  
       center.0 = 1.005863599  
       center.1 = 0.9957062963  
       center.2 = 1.045412918  
    Iter       4    CHI-SQUARE =  485324.4648  DOF =  40000
       r = 0.9452177765  
       center.0 = 1.006911119  
       center.1 = 0.9939173232  
       center.2 = 1.071988856  
    Iter       5    CHI-SQUARE =  420078.7832  DOF =  40000
       r = 0.9588137738  
       center.0 = 1.006953808  
       center.1 = 0.9931590149  
       center.2 = 1.084210946  
    Iter       6    CHI-SQUARE =  411628.4238  DOF =  40000
       r = 0.9589421222  
       center.0 = 1.006880601  
       center.1 = 0.9929427139  
       center.2 = 1.088112056  
    Iter       7    CHI-SQUARE =  410499.8879  DOF =  40000
       r = 0.9573892952  
       center.0 = 1.006836959  
       center.1 = 0.9928857322  
       center.2 = 1.089270357  
    Iter       8    CHI-SQUARE =  410332.343  DOF =  40000
       r = 0.9565857634  
       center.0 = 1.006819085  
       center.1 = 0.9928712451  
       center.2 = 1.089690649  
    Iter       9    CHI-SQUARE =  410308.0635  DOF =  40000
       r = 0.9562089803  
       center.0 = 1.006811431  
       center.1 = 0.9928673068  
       center.2 = 1.089835341  
    Iter      10    CHI-SQUARE =  410304.5247  DOF =  40000
       r = 0.9560544563  
       center.0 = 1.006808338  
       center.1 = 0.9928661809  
       center.2 = 1.089888106  
    Iter      11    CHI-SQUARE =  410304.0088  DOF =  40000
       r = 0.9559938191  
       center.0 = 1.006807121  
       center.1 = 0.9928658359  
       center.2 = 1.089907851  
    Iter      12    CHI-SQUARE =  410303.9336  DOF =  40000
       r = 0.9559703942  
       center.0 = 1.006806654  
       center.1 = 0.9928657253  
       center.2 = 1.089915322  
    Iter      13    CHI-SQUARE =  410303.9226  DOF =  40000
       r = 0.9559615064  
       center.0 = 1.006806476  
       center.1 = 0.9928656888  
       center.2 = 1.089918209  
    Iter      14    CHI-SQUARE =  410303.921  DOF =  40000
       r = 0.9559580159  
       center.0 = 1.006806405  
       center.1 = 0.9928656745  
       center.2 = 1.089919266  
    Iter      15    CHI-SQUARE =  410303.9208  DOF =  40000
       r = 0.9559568753  
       center.0 = 1.006806383  
       center.1 = 0.9928656725  
       center.2 = 1.08991977  
    Iter      16    CHI-SQUARE =  410303.9207  DOF =  40000
       r = 0.9559568244  
       center.0 = 1.006806348  
       center.1 = 0.9928656621  
       center.2 = 1.089919643  


Here we have used an :class:`.ExactModel` which takes a function ``calc_func``
to apply on the :class:`.Scatterer` (we have used :func:`.calc_holo` here).
The :class:`.ExactModel` is the default for :func:`.fit` but HoloPy also
includes :class:`.AlphaModel` and :class:`.PerfectLensModel`, which describe
specific models of hologram image formation. They take additional parameters
that allow for a more sophisticated calculation than a basic call to
:func:`.calc_holo`. Of course, these extra parameters can also be fit by
defining them as :class:`.Prior` objects.

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

When this is used to define a :class:`.Sphere`, :func:`.fit` will fit to
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
(``strategy="emcee"``) or tempering by changing the number of pixels evaluated (``strategy="subset tempering"``) or Monte Carlo temperature
(``strategy="parallel tempering"``) [not currently implemented]. You can see
the available strategies in your version of HoloPy by calling
`hp.inference.available_fit_strategies` or
`hp.inference.available_sampling_strategies`.

Each of these algorithms runs with a set of default values, but these may need
to be adjusted for your particular situation. For example, you may want to set
a random seed, control parallel computations, customize an initial guess, or
specify hyperparameters of the algorithm. To use non-default settings, you must
define a *Strategy* object for the algorithm you would like to use. You can
save the strategy to a file for use in future calculations or modify it in
place during an interactive session. ::

    cma_fit_strategy = CmaStrategy(popsize=15, parallel=None)
    cma_fit_strategy.seed = 1234
    hp.save('cma_strategy_file.h5', cma_fit_strategy)
    strategy_result = model.fit(data_holo, cma_fit_strategy)
    
Running the :meth:`.Model.fit` method is the same as calling
:func:`.fit`, but with the option to customize how the algorithm runs through
the :class:`.CmaStrategy` object. In the example above, we have adjusted
the ``popsize`` hyperparameter of the cma-es algorithm, prevented the
calculation from running as a parallel computation, and set a random seed for
reproducibility. The calculation returns a :class:`.FitResult` object, just
like a direct call to :func:`.fit`.

Similarly, we can customize a MCMC computation to sample a posterior by calling
:meth:`.Model.sample` with a :class:`.EmceeStrategy` object. Here we perform a
MCMC calculation that uses only 500 pixels from the image and runs 50 walkers
each for 2000 samples. We set the initial walker distribution to be one tenth
of the prior width.  In general, the burn-in time for a MCMC calculation will
be reduced if you provide an initial guess position and width that is as close
as possible to the eventual posterior distribution. You can use
:meth:`.Model.generate_guess` to generate an initial sampling to pass in as an
initial guess to your :class:`.EmceeStrategy` object. ::

        nwalkers = 50
        initial_guess = model.generate_guess(nwalkers, scaling=0.1)
        emcee_strategy = EmceeStrategy(npixels=500, nwalkers=nwalkers,
            nsamples=2000, walker_initial_pos=initial_guess)
        hp.save('emcee_strategy_file.h5', emcee_strategy)
        emcee_result = model.sample(data_holo, emcee_strategy)

Random Subset Fitting
---------------------
In the most recent example, we evaluated the holograms at the locations of only
500 pixels in the experimental image. This is because a hologram usually
contains far more information than is needed to estimate your parameters of
interest. You can often get a significantly faster fit with little or no loss
in accuracy by fitting to only a random fraction of the pixels in a hologram. 

You will want to do some testing to make sure that you still get
acceptable answers with your data, but our investigations have shown
that you can frequently use random fractions of .1 or .01 with little
effect on your results and gain a speedup of 10x or greater.
