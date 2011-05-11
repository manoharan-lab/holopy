Fitting holograms
=================

The fitting code is in the :mod:`holopy.analyze.fit` module.
So to start working working with it let's ::

    import holopy
    from holopy.analyze import fit

:func:`holopy.fit` is also provided as an alias to :func:`holopy.analyze.fit.fit`

The :func:`holopy.analyze.fit.fit` function takes a fit yaml file
that specifies what you data want fit, the intial guesses, bounds
on the parameters to be fitted and other relevant fit options. We
refer to information within this fit yaml file as the ``input deck``. 

.. note::

    For fitting holograms, :mod:`holopy` assumes that they are stored
    in images files labelled ``image`` followed by a number and
    then the file suffix, for example ``.tif``. So you might have
    files named ``image0001.tif`` through ``image1000.tif``.

Setting up the fit yaml file
----------------------------

The first thing to put in this file relates to where and what the
data you want fit is.

.. code-block:: yaml

    data_directory: Z:/rjm/kaz_2011-04-30/05
    image_range: [1001,1101]
    background_file:  Z:/rjm/kaz_2011-04-30/05/image2501.tif
    results_directory: .
    optics_file: uber100x.yaml
    fit_file: fit_params.yaml


data_directory
    This is the directory in which your data is stored.

image_range
    Data in files labelled from ``imageA.*`` to ``imageB.*`` are
    fit where ``A`` and ``B`` are the first and second elements
    in the range.

background_file
    Filename of image containing the background to be, by default,
    divided out of the each hologram.

results_directory
    Directory to which the fit results will be saved.

optics_file
    File containing the optics metadata (wavelength of light, pixel size, etc.)

fit_file
    File with fit bounds, step sizes and other fit-related options. 

Available models
^^^^^^^^^^^^^^^^

We should next specify the model we want our data fit to. Our options
are ``mie``, ``dimer``, or ``trimer.``

.. note::

    These models each have their separate ``forward_holo`` function
    in the :mod:`holopy.model` module.

You specifiy the model by setting the ``cluster_type``

.. code-block:: yaml

    cluster_type: 'mie'

Parameters to vary
^^^^^^^^^^^^^^^^^^

Next in the yaml file that's passed to :meth:`holopy.analyze.fit.fit` you
should specify the initial guesses used for the parameters. Or, if there
are parameters you will not vary specify what those parameters should be
set to. In the examples below we demonstrate what parameters should be
listed for the different models one might fit to (``mie``, ``dimer``
or ``trimer``).

.. code-block:: yaml

   n_particle_real : 1.6

Parameters needed for calculating holograms can be held constant. Those
parameters should be listed under ``hold_constant:`` with the syntax
demonstrated below where the radius and imaginary refractive index are
held constant. 

.. code-block:: yaml

    hold_constant:
        - n_particle_imag
        - radius

Parameters can also be tied. For example, when fitting a hologram from
a dimer of spherical particles, one may want to enforce that while
the particle refractive index may vary, it must be the same for both
particles. To do so, one should put in the yaml file the following.

.. code-block:: yaml

    tied:
        - n_particle_real_2 : n_particle_real_1
        - n_particle_imag_2 : n_particle_imag_1

The pair of tied parameters must be listed under ``tied:`` and the
code then will then tie the first parameter to the second. If, as
perhaps in the case of fitting a trimer, you would like to tie together
all three particles' radii you could do so with the following. In this
example we tie to the first particle's radius the remaining two
particles' radii.

.. code-block:: yaml

    tied:
        - radius2 : radius1
        - radius3 : radius1

When the function :meth:`holopy.analyze.fit.fit` is run the initial guesses
will be used in fitting the first hologram specified as the first entry
in ``image_range``. For each subsequent hologram to get fit, the results of
the previous fit will be used as the initial guesses. So, for example,
the best-fit parameters found for the hologram in ``image0001.tif`` will be
used to start the fit of ``image0002.tif``. However, if you would like
to reset certain parameters to the value specified in your fitting yaml
file, just list those parameters under the heading ``reset_to_intial``.
So, to reset the parameter for the gap distance between the two particles
making up a dimer add the following to your fit yaml file.

.. code-block:: yaml

    reset_to_initial:
        - gap_distance

Setting region to fit
^^^^^^^^^^^^^^^^^^^^^

You may have large holograms and not want to fit the entire frames. The
following keywords may be used to reduce the size of the hologram to fit.

    * ``subimage_center`` sets the x- and y-coordinate of the center of the
      region to fit to.
    * ``subimage_size`` sets the dimension of the square region to fit to
    * ``resample`` can be used to down sample the hologram. After the
      keyword ``resample:`` put the number of pixels you would like the
      full or subimaged hologram to be resampled to. 

The following could be added to only fit a 128x128 region center at the point
(380, 620) of a larger image.

.. code-block:: yaml

    subimage_center: [380, 620]
    subimage_size: 128

The following fits also fits a 128x128 sized hologram after first subimaging
and then resampling the data.

.. code-block:: yaml

    subimage_center: [550, 610]
    subimage: 512
    resample: 128



Setting fit tolerances
^^^^^^^^^^^^^^^^^^^^^^

Under the keyword ``tols`` the following tolerances may be set

    * ``gtol``. Fitting will stop when the calculated derivative
      drops below a threshold determined by this number. 
    * ``xtol``. Fitting will stop when the relatives error between
      two iteractions is this value or less. 
    * ``ftol``. Fitting will stop when the sum of the squares of
      the differences are at most this value. 



Setting the fitting parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to the fit yaml file described above and passed to the
method :meth:`holopy.analyze.fit.fit`, one should have a yaml file
specifying aspects of how the hologram will be fit. This file is
listed in the main fit yaml file under ``fit_file``.

Properties specified in this file are:

    * ``max_iter`` which sets the maximum number of iterations
      the fitter will perform to find the best-fit parameters
    * ``bounds`` for each fitting parameter specified with a
      lower and upper bound, such as ``[0., 1.]`` or, if the parameter
      can be varied with no bounds then ``[none, none]``.
    * ``max_step`` for setting the maximum step size used by the fitter
      in varying each of the parameters. If no value is given for a parameter
      then it will be automically determined by the fitter.
    * ``step`` for setting the step size used by the fitter in calculating
      derivatives with respect to each parameter.

.. note::

    If you can make reasonable guesses for the bounds and step sizes of
    the parameters to fit, we strongly suggest you do so. The more
    information the fitter receives, the better the results. 


Finding initial guess
---------------------

The following methods are useful for finding the initial guess to use
when fitting a hologram.

    * :meth:`holopy.analyze.fit.get_initial_guess` takes the fit yaml
      as input and calculates a hologram based on the initial parameters
      given.
    * :meth:`holopy.analyze.fit.get_target` can be used to load the
      hologram that you wish to fit to.

By useing both these functions you can compare the hologram to fit to
and the calculated hologram based on the initial guesses. 

Outputs
-------

The :meth:`holopy.analyze.fit.fit` method saves a file, called
``fit_results.tsv`` where the results of the fit are stored. To load
the data from this file::

    results = loadtxt('fit_results.tsv', skiprows=2, usecols=[1,2,3,4,5,6,7,8])

To calculate a hologram from the best fit parameters, you can use the
method :meth:`holopy.analyze.fit.get_fit_result` with the fit result yaml
associated with the hologram you fit. For every hologram fit, an output yaml
file is created in a ``fits`` directory. The name of this created yaml file is
the name of the hologram image file with ``_fit.yaml`` appended. 


Examples
--------

Single spheres
^^^^^^^^^^^^^^

The parameters needed for fitting a single sphere (using the
the ``cluster_type`` of ``mie``) are:

    * n_particle_real
    * n_particle_imag
    * radius
    * x
    * y
    * z
    * scaling_alpha

So here is an example of the fit yaml file we need to fit
a 100x100 region of 200 holograms. We are only varying the
position (*x*, *y* and *z*) and the ``scaling_alpha`` parameter.
If the following is stored in a file called ``my_mie_fit.yaml`` then
the command to begin fitting is::

    holopy.analyze.fit.fit('my_mie_fit.yaml')


.. code-block:: yaml

    data_directory : /home/me/my_data/this_date
    image_range: [101,300]
    background_file:  /home/me/my_data/this_date/image2501.tif
    results_directory: .
    optics_file: uber100x.yaml
    fit_file: fit_params.yaml

    cluster_type: 'mie'
    medium_index: 1.414
    scaling_alpha: .629
    radius: 0.9751e-6
    n_particle_real: 1.573289
    n_particle_imag: 0.
    x: 3.4e-06
    y: 3.4e-06
    z: 13.e-6

    hold_constant:
      - medium_index
      - n_particle_imag
      - n_particle_real
      - radius

    subimage_center : [106,128]
    subimage_size : 100

The following is an example of the ``fit_params.yaml`` file which
is pointed to in the above file after the keyword ``fit_file``. 

.. code-block:: yaml

    max_iter: 60

    bounds:
      scaling_alpha: [0.0, 1.0]
      radius: [0, none]
      x: [0, none]
      y: [0, none]
      z: [0, none]
      n_particle_real: [1, none]
      n_particle_imag: [0, none]

    max_step:
      scaling_alpha:
      radius:
      x:
      y:
      z:
      n_particle_real:
      n_particle_imag:

    step:
      radius: 0.05e-6
      x: 0.1e-6
      y: 0.1e-6
      z: 0.1e-6
      n_particle_real: 0.001
      n_particle_imag:
      scaling_alpha: 

Sphere dimers
^^^^^^^^^^^^^

The parameters needed for fitting a dimer of spherical particles
(using the ``cluster_type`` of ``dimer``) are:

    * n_particle_real_1
    * n_particle_real_2
    * n_particle_imag_1
    * n_particle_imag_2
    * radius_1
    * radius_2
    * x_com
    * y_com
    * z_com
    * scaling_alpha
    * euler_beta
    * euler_gamma
    * gap_distance
    


Sphere trimers
^^^^^^^^^^^^^^

The parameters needed for fitting a trimer of spherical particles
(using the ``cluster_type`` of ``trimer``) are:

    * n_particle_real_1
    * n_particle_real_2
    * n_particle_real_3
    * n_particle_imag_1
    * n_particle_imag_2
    * n_particle_imag_3
    * radius_1
    * radius_2
    * radius_3
    * x_com
    * y_com
    * z_com
    * scaling_alpha
    * euler_alpha
    * euler_beta
    * euler_gamma
