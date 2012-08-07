**********************
Fitting Models to Data
**********************

The most powerful use of holopy is to analyze data by fitting a model to it.  Given a correct model this can give high precision measurements of physical quantities from holographic or other data.

A fit needs:

1) A model of the scattering system
   
2) The data the fit is trying to match
   
3) (Optional) a fitting algorithm (holopy defaults to using the supplied nmpfit)

The simplest fit will look like ::

  result = fit(model, data)

Scattering Model: :class:`scatterpy.fitting.model.Model`
========================================================

Model is a set of variable parameters a scattering theory for computing simulated scattering data, and any ancillary information needed for this calculation.  ::

  model = Model(scatterer_parametrization, theory)

scatterer_parametrization
-------------------------

There are two ways to specify a scatterer_parametrization.  The standard method is to provide a :class:`holopy.scattering.scatterer` object telling which values to fix and which to vary. ::

  param_scat = Sphere(n = 1.59, r = par(guess = .5, limit = [.3, .8]),
                      center = (par(10), par(10), par(10))

This will tell holopy that you want to model scattering from a sphere of fixed index n = 1.59, and vary the sphere's radius and position to attempt to match a hologram.  Initial guesses are provided for the radius and three center coordinates, and the radius is constrained to lie between .3 and .8.  The three radii are allowed to vary without limit.

If your model does not fit neatly into a parametrized scatterer like this, Holopy provides a lower level interface ::

  param = Parametrization(make_scatterer,
                          parameters = [par(guess = .5, name = 'r'),
                                        par(guess = 0, name = euler_alpha'),
                                        par(guess = 0, name = 'euler_beta')])

Here make_scatterer needs to be a function that takes keyword arguments of the names of the parameters and returns a scatterer.  

theory
------

The theory in a model is one of the calc_* functions provided by scattering theories, for example Mie.calc_holo.

Technically, you can use any function here as long as it takes a scatterer and a DataTarget (and optionally additional keyword arguments) as arguments and returns a Data object.

Other information
-----------------

If you want to provide a scaling alpha, that can be done as a keyword argument to the model ::
  
  model = Model(param_scat, Mie.calc_holo, alpha = par(.6, [0, 1]))

If you want to fit to information normally provided in the Data Metadata, you can provide a parametrized Metadata object, any parameters specified here will override those specified in the data ::

  model = Model(param_scat, mie.calc_holo, metadata = Metadata(divergence = par(0, [0, 1])))

Data
====

Any Data object with a full set of metadata.  Between the model and the provided Data, you must specify or parametrize all of the values needed to perform a scattering calculation.

Minimizer
=========

If you do not provide a minimizer, fits will default to using the supplied Nmpfit minimizer ::

  fit(model, data, minimizer = Nmpfit())

You can choose another minimizer or provide non-default options to a minimizer by passing a minimizer object to fit, for example ::

  fit(model, data, minimizer = Nmpfit(ftol=1e-5, xtol = 1e-5, gtol=1e-5, niter=2))

To tell nmpfit to use looser tolerances and a small iteration limit (to get a fast result to check things out), or ::

  fit(model, data, minimizer = Ralg())

To use OpenOpt's ralg minimizer instead of nmpfit.  (This will fail unless you have nmpfit installed and configured so that Holopy can find it).  

If you need to provide information to the minimizer about specific parameters (for example a derivative step to nmp fit) you add them to the par call as keyword args, for example ::

  Sphere(n = par(1.59, [1, 2], step = 1e-3), ...)

Examples
========

Sphere
------

Here let's compute a hologram and then fit it.  You can replace the
calculated hologram with real data, if you like ::

   from holopy import Metadata, DataTarget
   from holopy.fititng import Model, par, fit
   from holopy.scattering.scatterer import Sphere
   from holopy.scattering.theory import Mie

   target = holopy.DataTarget(points = 100, (wavelen = 658, index = 1.33, pixel_scale=0.1))
   s = Sphere(center = (10.2, 9.8, 10.3), r = .5, n = 1.58)
   holo = mie.calc_holo(s, target)

   par_s = Sphere(center = (par(guess = 10, limit = [5,15]), par(10, [5, 15]), par(10, [5, 15])),
                  r = .5, n = 1.58)

   model = Model(par_s, Mie.calc_holo, alpha = par(.6, [.1, 1]))
   result = fit(model, holo)

Here we specify the three spatial coordinates as parameters, and fix
the index of refraction and radius of the sphere.

`result.scatterer` is the scatterer that best matches the hologram,
`result.alpha` is the alpha for the best fit.  `result.chisq` and
`result.rsq` are statistical measures of the the goodness of the fit.
`result.model` and `result.minimizer` are the Model and Minimizer
objects used in the fit, and `result.minimization_info` contains any
further information the minimization algorithm returned about the
minimization procedure (for nmpfit this includes things like covariance
matrices). 

You will most likely want to save the fit result ::

  holopy.save('result.yaml', result)

This saves all of the information about the fit to a yaml text
file.  These files are reasonably human readable and serve as our archive format for data.  They can be loaded back into python with ::

  loaded_result = holopy.load('result.yaml')

You can specify a complex index with ::

  Sphere(n = ComplexParameter(real = par(1.58), imag = 1e-4))

This will fit to the real part of index of refraction while holding the imaginary part fixed.  You can fit to it as well by specifying a Parameter instead of a fixed number there.  

Tying Parameters
----------------
You may desire to fit holograms with *tied parameters*, in which several 
physical quantities that could be varied independently are constrained to have
the same (but non-constant) value. A common example involves fitting a model
to a multi-particle hologram in which all of the particles are constrained to
have the same refractive index, but the index is determined by the fitter.
This may be done by defining a Parameter and using it in multiple places ::
  
  n_real = par(1.59)
  sc = SphereCluster([Sphere(n = n_real, r = par(0.5e-6), 
                             center = array([10., 10., 20.]) * 1e-6),
                      Sphere(n = n_real, r = par(0.5e-6),
                             center = array([9., 11., 21.] * 1e-6))])

Hologram with Beam Tilt
-----------------------

Here we override some of the parameters specified in the Data (or in fact you can leave them as none when specifying Metadata for this data) ::

  model = Model(Sphere(...), metadata = Metadata(
    ilum_vector = UnitVector(beta = par(0), gamma = par(0))))

Fitting this model will vary the beam tilt

Static Light Scattering
-----------------------

Assuming you have recorded some static light scattering data in a file sls_data.txt and the metadata in sls_meta.yaml ::

  data = hp.load('sls_data.txt', 'sls_meta.yaml')

  model = Model(Sphere(n = par(1.58, [1, 2]), r = par(.5)), Mie.calc_intensity, scaling = par(1))

  result = fit(model, data)

Alternative Scatterer Parameterizations
---------------------------------------

Holopy also provides some additional views of scatterers that may be convenient for fitting.  For example ::

  from holopy.fitting.views import Dimer
  s = Dimer([Sphere(n, r), Sphere(n, r)], gap, beta, gamma, center)

This contains the same number of parameters as a 2 sphere SphereCluster and fully specifies a SphereCluster, but provides a different set of knobs for the fitter to adjust.  
