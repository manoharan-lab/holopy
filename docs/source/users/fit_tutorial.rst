Fitting holograms
=================

The :func:`holopy.analyze.fit.fit_new` fits a model of the object to a given set of data.

:model:
   A description of the scattering system to fit to your hologram.  This consists of

   :parameters:

	  The set of parameters the fitter should vary to attempt to match
	  the hologram.  These are typically things like coordinates of
	  spheres, radii, and indices of refraction. They might also
	  include rotation angles of a cluster or inter-particle
	  separations.  For most basic use, you will do this by passing in
	  a :class:`scatterpy.scatterer.Scatterer` object with
	  :class:`holopy.analyze.fit_new.Parameter` objects instead of
	  numbers for any values you want to vary in the fit.

   :theory:

	   The scattering theory to be used to compute holograms for
	   comparison with the data
	   
   :scatterer generator (optional):

	   A function that takes as its arguments the parameters and
	   returns a scatterer which the theory can use to compute a
	   hologram.  

:data:

   The measured hologram to which you are fitting the model

:algorithm (optional):

   The fitting algorithm.  This defaults to the supplied nmpfit
   algorithm, but if you have OpenOpt installed you can use minimizers
   from that package as well.

Here let's compute a hologram and then fit it.  You can replace the
calculated hologram with real data, if you like ::

   from holopy import Optics
   from holopy.analyze.fit_new import Model, par, fit
   from scatterpy.scatterer import Sphere
   from scatterpy.theory import Mie

   optics = Optics(wavelen = .658, index = 1.33, pixel_scale=0.1)
   mie = Mie(optics, 100)
   s = Sphere(center = (10.2, 9.8, 10.3), r = .5, n = 1.58)
   holo = mie.calc_holo(s, 1.0)

   par_s = Sphere(center = (par(guess = 10, limit = [5,15]), par(10, [5, 15]), par(10, [5, 15])),
                  r = .5, n = 1.58)
   

   alpha = par(.6, [.1, 1], 'alpha')
	   
   model = Model((par_s, alpha), Mie)
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
file.  These files are reasonably human readable and can be loaded
back as an object with ::

  loaded_result = holopy.io.yaml_io.load('result.yaml')
