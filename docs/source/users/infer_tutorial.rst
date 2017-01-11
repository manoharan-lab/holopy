.. _infer_tutorial

Bayesian inference of Parameter Values
======================================

:ref:`calc_tutorial` can inform us about the hologram produced by a specific scatterer,
but they can't tell us anything about what type of scatterer produced an experimentally measured hologram.
For this reverse problem, we turn to a Bayesian inference approach. We can calculate the
holograms produced by many similar scatterers, and evaluate which ones are closest to
our measured hologram. We can then use known information about the scatterers to determine
which exact scatterer parameters were most likely to have produced the observed hologram.

In this example, we will infer the size, refractive index, and position of a spherical scatterer::

    import holopy as hp
    import numpy as np
    from holopy.core.io import get_example_data_path, load_average
    from holopy.core.process import bg_correct, subimage, normalize
    from holopy.scattering import Sphere, calc_holo
    from holopy.inference import prior, AlphaModel, tempered_sample

    # load an image
    imagepath = get_example_data_path('image01.jpg')
    raw_holo = hp.load_image(imagepath, spacing = 0.0851, medium_index = 1.33, illum_wavelen = 0.66, illum_polarization = (1,0))
    bgpath = get_example_data_path(['bg01.jpg','bg02.jpg','bg03.jpg'])
    bg = load_average(bgpath, refimg = raw_holo)
    data_holo = bg_correct(raw_holo, bg)

    # process the image
    data_holo = subimage(data_holo, [250,250], 200)
    data_holo = normalize(data_holo)

    # Set up the prior
    s = Sphere(n=prior.Gaussian(1.5, .1), r=prior.BoundedGaussian(.5, .05, 0, np.inf),
             center=prior.make_center_priors(data_holo))

    # Set up the noise model
    noise_sd = data_holo.std()
    model = AlphaModel(s, noise_sd=noise_sd, alpha=1)

    result = tempered_sample(model, data_holo)

    result.values()
    hp.save('example-sampling.h5', result)

The first few lines import the code needed to compute holograms and do parameter inference

..  testcode::

    import holopy as hp
    import numpy as np
    from holopy.core.io import get_example_data_path, load_average
    from holopy.core.process import bg_correct, subimage, normalize
    from holopy.scattering import Sphere, calc_holo
    from holopy.inference import prior, AlphaModel, tempered_sample

Preparing Data
~~~~~~~~~~~~~~

Next, we load a hologram from a file using the same steps
as those in :ref:`load_tutorial`

..  testcode::

    # load an image
    imagepath = get_example_data_path('image01.jpg')
    raw_holo = hp.load_image(imagepath, spacing = 0.0851, medium_index = 1.33, illum_wavelen = 0.66, illum_polarization = (1,0))
    bgpath = get_example_data_path(['bg01.jpg','bg02.jpg','bg03.jpg'])
    bg = load_average(bgpath, refimg = raw_holo)
    data_holo = bg_correct(raw_holo, bg)

You will notice that the hologram data is localized to a region near the center of the image.
We only want to compare calculated holograms to this region, so we will crop our image with :func:`.subimage`.
We also need to normalize the data so that its mean is 1, since calculations return a normalized result. Since our
image is background divided, its mean is already very close to 1, but it is good to get in the habit of normalizing anyway.

..  testcode::

    # process the image
    data_holo = subimage(data_holo, [250,250], 200)
    data_holo = normalize(data_holo)

..  note::

    It is often useful to test an unfamiliar technique on data for which you know the expected outcome.
    Instead of actual data, you could use a hologram calculated from :func:`.calc_holo`, and modulated
    by random noise with :func:`.add_noise`.

Defining a Probability Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Priors
------

We know that the hologram was produced by a spherical scatterer, so we want to 
define a :class:`.Sphere` object like we did in the :ref:`calc_tutorial` tutorial.
However, in this case we don't know what parameters to specify for the sphere (since that is what we're trying to find out).
Instead, we write down a probabilistic statement of our prior information about the sphere. 
In statistics, we call this a prior. For the case we are
investigating here, you would probably have some best guess and
uncertainty about the size and index of your particle, obtained from the supplier or from prior
work with the particle. We will guess radius to be 0.5 microns (with 50 nm error) and refractive index to be 1.5 (with 0.1 error).
We also need to provided a prior for the position of the sphere.
We can use a :func:`.hough` transform to get a pretty good guess
of where the particle is in x and y, but it is difficul to determine where it is in z.

..  note::
    One trick to get a better estimate of z position is to numerically propagate the hologram backwards in space 
    with :func:`.propagate`, and look for where the interference fringes vanish.

Let's turn our information about priors into code by defining our scatterer:

..  testcode::

    s = Sphere(n=prior.Gaussian(1.5, .1), r=prior.BoundedGaussian(.5, .05, 0, np.inf),
             center=prior.make_center_priors(data_holo))

The Gaussian distribution is the prior used to describe a value for which all we
know is some expected value and some uncertainty on that expected value. For the
radius we also know that it must be nonnegative, so we can bound the Gaussian at
zero. The :func:`.make_center_priors` function automates generating priors for a sphere
center using :func:`.center_finder` (based on a hough transform). It assigns Gaussian priors for x and y, and picks a large
uniform prior for z to represent our ignorance about how far the particle is from the imaging plane. In this case the center prior will be::
    
    [Gaussian(mu=11.4215, sd=0.0851),
    Gaussian(mu=9.0945, sd=0.0851),
    Uniform(lower_bound=0, upper_bound=170.2)]

..  testcode::
    :hide:

    print(s.center[0])

..  testoutput::
    :hide:

    Gaussian(mu=24.186546323529495, sd=0.08510000000000062)


Likelihood
----------

Next we need to define a model that tells HoloPy how probable it is that we
would see the data we observed given some hypothetical scatterer position, size
and index. In the language of statistics, this is referred to as a likelihood.
In order to compute a likelihood, you need some estimate of how noisy your data
is (so that you can figure out how likely it is that the differences between
your model and data could be explained by noise). Here we use the standard
deviation of the data, which is an overestimate of the true noise, since it also includes variaion due to our signal. 

..  testcode::

  noise_sd = data_holo.std()
  model = AlphaModel(s, noise_sd=noise_sd, alpha=1)

..  note::

    ``alpha`` is a model parameter that scales the scattered beam intensity relative to the reference beam.
    It is often less than 1 for reasons that are poorly understood. If you aren't sure what value it should take
    in your system, you can allow ``alpha`` to vary by giving it a prior like the sphere parameters. 

Sampling the Posterior
~~~~~~~~~~~~~~~~~~~~~~

Finally, we can sample the posterior probability for this model. 
Essentially, a set of proposed scatterers are randomly generated according to the priors we specified.
Each of these scatterers is then evaluated in terms of how well it matches the experimental hologram ``data_holo``.
A Monte Carlo algorithm iteratively produces and tests sets of scatterers to find the scatterer parameters 
that best reproduce the target hologram. We end up with a distribution of values for each parameter (the posterior)
that represents our updated knowledge about the scatterer when accounting for the expected experimental hologram.
To do the actual sampling, we use :func:`.tempered_sample` (ignoring any RuntimeWarnings about invalid values):

    result = tempered_sample(model, data_holo)

The above line of code may take a long time to run (it takes 10-15 mins on our 8-core machines).
If you just want to quickly see what results look like, try:

..  testcode::

    result = tempered_sample(model, data_holo, nwalkers=10, samples=100, max_pixels=100)

This code should run very quickly, but its results cannot be trusted for any actual data.
Nevertheless, it can give you an idea of what format results will take.
In our last line of code, we have adjusted three parameters to make the code run faster: 
``nwalkers`` describes the number of scatterers produced in each generation.
``samples`` describes how many generations of scatterers to produce. 
Together, they define how many scatterering calculations must be performed. 
For the values chosen inthe fast code, a Monte Carlo steady state will not yet have been achieved, so the resulting posterior distribution is not very meaningful.
``max_pixels`` describes the maximum number of pixels compared between the experimental holgoram and the test holograms.
It turns out that holograms contain a lot of redundant information (e.g. radial symmetry), so a subset of pixels can be analyzed without loss of accuracy.
However, 100 pixels is probably too few to capture all of the relevant information in the hologram. 

You can get a quick look at our obtained values with::

..  testcode::

    result.values()

result.values() gives you the maximum a posteriori probability (MAP) value as well as 1 sigma (or you can request any
other sigma with an argument to the function) credibility intervals. You can also look only at central measures::

    result.MAP
    result.mean
    result.median

Since calculation of useful results takes a long time, you will usually want to save them to an hdf5 file::

..  testcode::

   hp.save('example-sampling.h5', result)

References
~~~~~~~~~~

.. [Dimiduk2016] Dimiduk, T. G., Manoharan, V. N. (2016) Bayesian approach to analyzing holograms of colloidal particles. Optics Express

.. [Gregory2005] Gregory, P. (2005) Bayesian Logical Data Analysis. Cambridge University Press
