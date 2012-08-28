**************************
Fitting to Timeseries Data
**************************

If you are taking video holograms (one of the most useful cases), you
will probably find yourself wanting to fit long timeseries of data.
This is done with :func:`holopy.fitting.fit.fit_series` ::

  fit_series(model, dataset, prefit = None, postfit = None)

for each image in the dataset, fit series will:

1) Get the next data from the dataset, if it is a string, load an
   image by that name from the current directory -> Data object
 
2) Call prefit(data, model, **kwargs) -> Data object
   
3) Fit the model to the data -> FitResult object
   
4) Save the FitResult with holopy.save
   
5) Call postfit(fitresult, **kwargs) -> FitResult
   
6) Use the last FitResult to setup the guess for the next frame.  

The fit_series arguments are:

:model:

   Described exactly as for a single fit.  The same model is used to
   fit all of the data.  If you need to vary the model across the fit
   you can instead provide a ModelFamily object and select between
   models in prefiting

:dataset:

   The simplest dataset is a list of filenames, but it can be any
   iterable that returns either a strings or a Data objects.  If the
   dataset is string filenames, metadata will need to be provided as a
   separate metadata keyword argument to fit_series, or the model
   specified with a full set of metadata.  

:prefit (optional):

   Prefit gives you a chance to do any processing you want to do on
   images before fitting them.  A user supplied prefit should
   expect to be given a Data object and a Model object and should
   return a Data object to fit.  The prefit function should accept
   (but can ignore) arbitrary keyword arguments so that in the future
   we can pass more information into prefit.

:postfit (optional):

   postfit will be called with the FitResult object as an
   argument, should return a FitResult, and should also accept arbitrary
   keyword arguments for future proofing.  

Each frame fitted actually uses the Model from the previous fitresult.
This allows you to modify the initial guesses for the next frame, but
you can also swap out any arbitrary peices about the model if desired
(switch from Mie to Multisphere theory, change the number of particles
in a Spheres object, or anything else).  Just remember, with great
power comes great responsibility.
   
If postfit raises a :class:`holopy.fitting.series.RejectFit` or
subclass, the same data will be fit again.  The model for the retry
will be extracted from RejectFit.result_override if it is not None,
allowing you to set different guesses or model parameters.



