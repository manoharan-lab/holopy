.. _fit_tutorial:

Fitting Models to Data
======================

In addition to Bayesian inference, HoloPy can also do simpler least-squares fits to determine the scatterer parameters that best match
an experimentally measured hologram. The main advantage of this technique is that it can be much faster.
The drawback is that good intial guesses of each parameter are required to obtain accurate results.

..  note::

    The HoloPy fitting methods have been superseded by the Bayesian inference
    techniques described in the :ref:`infer_tutorial` tutorial. We strongly
    recommend that approach unless you have a good reason that fitting is
    preferable in your particular situation.

A Simple Fit
~~~~~~~~~~~~

We start by loading and processing data just as we did for the parameter inference in the previous tutorial.

..  testcode::

    import holopy as hp
    import numpy as np
    from holopy.core.io import get_example_data_path, load_average
    from holopy.core.process import bg_correct, subimage, normalize
    from holopy.scattering import Sphere, calc_holo

    # load an image
    imagepath = get_example_data_path('image01.jpg')
    raw_holo = hp.load_image(imagepath, spacing = 0.0851, medium_index = 1.33, illum_wavelen = 0.66, illum_polarization = (1,0))
    bgpath = get_example_data_path(['bg01.jpg','bg02.jpg','bg03.jpg'])
    bg = load_average(bgpath, refimg = raw_holo)
    data_holo = bg_correct(raw_holo, bg)

    # process the image
    data_holo = subimage(data_holo, [250,250], 200)
    data_holo = normalize(data_holo)

Define a Model
--------------
The model specification is a little bit different from the inference case.
First, we define a parameterized scatterer including initial guesses and absolute bounds
using the :class:`.Parameter` class. Note that the bounds here are not uncertainty values as in
the inference case, but instead represent the full allowed range of a parameter (like the :class:`.Uniform` prior).
The ``center`` coordinates must be specified as (`x`, `y`, and `z`, in that order).
Here, we will keep particle radius and refractive index fixed. Fitting works best when there are only a few uncertain parameters.
You can find guesses for `x` and `y` coordinates with :func:`.center_find`, and guess `z` with :func:`.propagate`.
In this image (uncropped version), the particle's center is near (24, 22, 15), with coordinates in microns. 

..  testcode::

    from holopy.fitting import fit, Model
    from holopy.fitting import Parameter as par
    par_s = Sphere(center = (par(guess = 24, limit = [15,30]),
      par(22, [15, 30]), par(15, [10, 20])), r = .5, n = 1.58)

Then this parametrized scatterer, along with a desired scattering calculation, is
used to define a model:

..  testcode::

   model = Model(par_s, calc_holo, alpha = par(.6, [.1, 1]))

``alpha`` is an additional fitting parameter first introduced in [Lee2007] (see :ref:`credits` for additional details).

To see how well the guess in your model lines up with the hologram you
are fitting to, use :

..  testcode::

    guess_holo = calc_holo(data_holo, par_s, scaling=model.alpha)

Run the Fit
-----------

Once you have all of that set up, running the fit is almost
trivially simple::

    result = fit(model, data_holo)

We can see just the fit results with ``result.scatterer.center``.
The initial guess of the sphere's position (24, 22, 15)
was corrected by the fitter to (24.17,21.84,16.42). Notice that we have achieved sub-pixel position resolution!

From the fit,
``result.scatterer`` gives the scatterer that best matches the hologram,
``result.alpha`` is the alpha for the best fit.  ``result.chisq`` and
``result.rsq`` are statistical measures of the the goodness of the fit.

You can also compute a hologram of the final fit result to compare to
the data with::

  result_holo = calc_holo(data_holo, result.scatterer, scaling=result.alpha)

Finally, we save the result with::

  hp.save('result.h5', result)

.. _random_subset:

Speeding up Fits with Random Subset Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A hologram usually contains far more information than is needed to
determine the number of parameters you are interested in. Because of
this, you can often get a significantly faster fit with no little or
no loss in accuracy by fitting to only a random fraction of the pixels
in a hologram. ::

  result = fit(model, data_holo, random_subset=.01)

You will want to do some testing to make sure that you still get
acceptable answers with your data, but our investigations have shown
that you can frequently use random fractions of .1 or .01 with little
effect on your results and gain a speedup of 10x or greater.

Advanced Parameter Specification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Complex Index of Refraction
---------------------------

You can specify a complex index with:

..  testcode::

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
defining a Parameter and using it in multiple places :

..  testcode::

  from holopy.scattering import Spheres
  n1 = par(1.59)
  sc = Spheres([Sphere(n = n1, r = par(0.5e-6), \
    center = [10., 10., 20.]), \
    Sphere(n = n1, r = par(0.5e-6), center = [9., 11., 21.])])
