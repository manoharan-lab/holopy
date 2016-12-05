.. _infer_tutorial

Bayesian inference of Parameter Values
======================================

Here we infer the size, refractive index, and position of a spherical scatterer::

  import holopy as hp
  import numpy as np
  from holopy.inference import prior, AlphaModel, tempered_sample
  from holopy.scattering import Sphere, calc_holo
  from holopy.core.process import bg_correct
  from holopy import detector_grid

  # Make a simulated hologram
  d = detector_grid((100, 100), .1)
  s = Sphere(r=.5, n=1.6, center=(5, 5, 5))
  h= calc_holo(d, s, illum_wavelen=.66, medium_index=1.33, illum_polarization=(0, 1))

  # Set up the prior
  s = Sphere(n=prior.Gaussian(1.5, .1), r=prior.BoundedGaussian(.5, .05, 0, np.inf), center=prior.make_center_priors(h))

  # Set up the noise model
  noise_sd = .1
  model = AlphaModel(s, noise_sd=noise_sd, alpha=1)

  r = tempered_sample(model, h, nwalkers=100, samples=800, seed=40, min_pixels=10, max_pixels=1000, stages=5)
  hp.save('example-sampling.h5', r)

  r.MAP
  r.values()


The first few lines import the code needed to compute holograms and do parameter inference::

  import holopy as hp
  import numpy as np
  from holopy.inference import prior, AlphaModel, tempered_sample
  from holopy.scattering import Sphere, calc_holo
  from holopy import detector_grid

Preparing Data
--------------

Next, we compute the hologram for a microsphere using the same steps
as those in :ref:`calc_tutorial`::

  d = detector_grid((100, 100), .1)
  s = Sphere(r=.5, n=1.6, center=(5, 5, 5))
  h= calc_holo(d, s, illum_wavelen=.66, medium_index=1.33, illum_polarization=(0, 1))

If you are working with your own data, it is important to remember to
normalize the data, since calculations return a normalized result. So
if you had ``data.tif`` and ``bg.tif`` you would use something like::

  import holopy as hp
  from holopy.core.process import normalize
  data_holo = normalize(bg_correct(hp.load_image('data.tif', spacing = .1, illum_wavelen=.66,
                                                 illum_polarization=[1, 0], medium_index=1.33),
                                   hp.load_image('bg.tif', spacing = .1)))

Defining a Probability Model
----------------------------

Priors
------

For the Bayesian inference approach that holopy uses, the first thing we need to
do is write down a probabilistic statement of our prior information about what
we are trying to infer. In statistics, we call this a prior. For the case we are
investigating here, you would probably have some guess at a value and
uncertainty about the size and index of your particle from the supplier or prior
work with the particle. We can use a hough transform to get a pretty good guess
of where the particle is in x and y, but, if the hologram was from actual data,
you probably would not have a very good guess of where it is in z. So lets turn
this information into code::

  s = Sphere(n=prior.Gaussian(1.5, .1), r=prior.BoundedGaussian(.5, .05, 0, np.inf), center=prior.make_center_priors(h))

The Gaussian distribution is the prior used to describe a value for which all we
know is some expected value and some uncertainty on that expected value. For the
radius we also know that it must be nonnegative, so we can bound the Gaussian at
zero. Finally for z, we have chosen to represent our ignorance in where the
particle might be in z by assigning it equal probability that it might be
anywhere between 0 and 100 microns from the focal plane. The
prior.make_center_priors(h) function automates generating priors for a sphere
center using a hough transform centerfinder for x and y, and picks a large
uniform prior for z. In this case the prior will be::

  [Gaussian(mu=5.00013, sd=0.1),
   Gaussian(mu=5.00010, sd=0.1),
   Uniform(lower_bound=0, upper_bound=100.0)]

Likelihood
----------

Next we need to define a model that tells HoloPy how probable it is that we
would see the data we observed given some hypothetical scatterer position, size
and index. In the language of statistics, this is referred to as a likelihood::

  noise_sd = .1
  model = AlphaModel(s, noise_sd=noise_sd, alpha=1)

Sampling the Posterior
----------------------

Finally, we can sample the posterior probability for this model and save the results to an hdf5 file::

  r = tempered_sample(model, h)
  hp.save('example-sampling.h5', r)

You can get a quick look at the values with::

  r.MAP
  r.values()

r.MAP gives you the Maximium a Posteriori probability (values we observed while sampling that has the highest probability of being the correct parameter values). r.values() gives you the MAP value as well as 1 sigma (or you can request any other sigma with an argument to the function) credibility intervals. 

References
----------

.. [Dimiduk2016] Dimiduk, T. G., Manoharan, V. N. (2016) Bayesian approach to analyzing holograms of colloidal particles. Optics Express

.. [Gregory2005] Gregory, P. (2005) Bayesian Logical Data Analysis. Cambridge University Press
