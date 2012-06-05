Fitting holograms
=================

The :func:`holopy.analyze.fit.fit_new` fits a model of the object to a given set of data.

:model:
   A description of the scattering system to fit to your hologram.  This consists of

   :parameters:

	  The set of parameters the fitter should vary to attempt to match
	  the hologram.  These are typically things like coordinates of
	  spheres, radii, and indices's of refraction, but can be things
	  like rotation angles of a cluster, inter-particle separations,
	  or pretty much anything.

   :theory:

	   The scattering theory that can be used to compute holograms of
	   the scatterer for comparison with the data
	   
   :scatterer generator:

	   A function that takes as its arguments the parameters and
	   returns a scatterer which the theory can use to compute a
	   hologram

:data:

   The measure hologram which you are fitting the model to

:algorithm (optional):

   The fitting algorithm to use to fit the model.  This defaults to
   the supplied nmpfit algorithm, but if you have OpenOpt installed
   you can use fitters from that package as well.

Here lets compute a hologram and then fit it.  You could replace the
hologram calculation with loading some real data ::

   from holopy import Optics
   from holopy.analyze.fit_new import Model, Parameter, fit
   from scatterpy.scatterer import Sphere
   from scatterpy.theory import Mie
   optics = Optics(wavelen = .658, index = 1.33, pixel_scale=0.1)
   mie = Mie(optics, 100)
   s = Sphere(center = (10.2, 9.8, 10.3), r = .5, n = 1.58)
   holo = mie.calc_holo(s, 1.0)
   
   parameters = [Parameter(name = 'x', guess = 10, limit = [5, 15]),
                 Parameter('y', 10, [5, 15]),
                 Parameter('z', 10, [5, 15]),
                 Parameter('alpha', 1.0)]
   def make_scatterer(x, y, z):
       return Sphere(center = (x, y, z), n = 1.58, r = .5)
   model = Model(parameters, Mie, make_scatterer=make_scatterer)
   result = fit(model, holo)

Here we specify the three spatial coordinates as parameters, and fix
the index of refraction and radius of the sphere.

`result.scatterer` is the scatterer that best matches the hologram,
`result.alpha` is the alpha for the best fit.  `result.chisq` and
`result.rsq` are statistical measures of the the goodness of the fit.
`result.model` and `result.minimizer` are the Model and Minimizer
objects used in the fit, and `result.minimization_info` contains any
further information the minimization algorithm returned about the
minimization procedure (for nmpfit this is things like covariance
matrixies). 

You will most likely want to save the fit result ::

  holopy.save('result.yaml', result)

This saves the entirety of information about the fit to a yaml text
file.  These files are reasonably human readable and can be loaded
back as an object with ::

  loaded_result = holopy.io.yaml_io.load('result.yaml')