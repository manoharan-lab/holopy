.. _dev_tutorial:

Developer's Guide
=================

.. _dev_install:

Installing HoloPy for Developers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you are going to hack on holopy, you probably only want to compile the
scattering extensions.

**For Mac and Linux:**

Download or clone the latest version of HoloPy from Git Hub at `https://github.com/manoharan-lab/holopy <https://github.com/manoharan-lab/holopy>`_.

Let's say you downloaded or cloned HoloPy to
``/home/me/holopy``. Then open a terminal, ``cd`` to ``/home/me/holopy`` and run::

    python setup.py develop

This puts the extensions inside the source tree, so that you can work
directly from ``/home/me/holopy`` and have the changes reflected in the version
of HoloPy that you import into python.

**Note for Mac users:** gfortran may put its library in a place python can't find it. If you get errors including something like ``can't find /usr/local/libgfortran.3.dynlib`` you can symlink them in from your install. You can do this by running::

  sudo ln -s /usr/local/gfortran/lib/libgfortran.3.dynlib /usr/local/lib
  sudo ln -s /usr/local/gfortran/lib/libquadmath.3.dynlib /usr/local/lib

**For Windows:**
Installation on Windows is still a work in progress, but we have been able to get HoloPy working on Windows 10 with an AMD64 architecture (64-bit) processor.

1. Install `Anaconda <https://www.continuum.io/downloads>`_ with Python 3.6 and make sure it is working.
2. Install the C compiler. It's included in `Visual Studio 2015 Community <https://www.visualstudio.com/downloads/>`_. Make sure it is working with a C helloworld.
3. From now on, make sure any command prompt window invokes the right environment conditions for compiling with VC. To do this, make sure ``C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat`` is added to the system path variable. This batch detects your architecture, then runs another batch that sets the path include the directory with the correct version of the VC compiler.
4. Install cython and made sure it works.
5. Install `Intel's Fortran compiler <https://software.intel.com/en-us/fortran-compilers/try-buy>`_. A good place to start is the trial version of Parallel Studio XE. Make sure it is working with a Fortran helloworld.
6. Install `mingw32-make <https://sourceforge.net/projects/mingw/files/MinGW/Extension/make/>`_, which does not come with Anaconda by default.
7. Download or clone the master branch of HoloPy from `https://github.com/manoharan-lab/holopy <https://github.com/manoharan-lab/holopy>`_. 
8. Open the command prompt included in Intel's Parallel Studio. Run ``holopy/setup.py``. It is necessay to use Intel's Parallel Studio command prompt to avoid compiling errors.
9. Install the following dependencies that don't come with Anaconda::
    
        conda install xarray dask netCDF4 bottleneck
        conda install -c astropy emcee=2.2.1

10. Open an iPython console where holopy is installed and try ``import holopy``.

If the above procedure doesn't work, or you find something else that does, please `let us know <https://github.com/manoharan-lab/holopy/issues>`_ so that we can improve these instructions.

..  _xarray:

How HoloPy Stores Data
~~~~~~~~~~~~~~~~~~~~~~
Images in HoloPy are stored in the format of xarray `DataArrays
<http://xarray.pydata.org/en/stable/data-structures.html#dataarray>`_. Spatial
information is tracked in the DataArray's ``dims`` and ``coords`` fields
according to the HoloPy :ref:`coordinate_system`. Additional dimensions are
sometimes specified to account for different z-slices, times, or field
components, for example. Optical parameters like refractive index and
illumination wavelength are stored in the DataArray's ``attrs`` field.

The :func:`.detector_grid` function simply creates a 2D image composed entirely
of zeros. In contrast, the :func:`.detector_points` function creates a DataArray
with a single dimension named 'point'. Spatial coordinates (in either Cartesian
or spherical form) track this dimension, so that each data value in the array
has its own set of coordinates unrelated to its neighbours. This type of
one-dimensional organization is sometimes used for 2D images as well. Inference
and fitting methods typically use only a subset of points in an image (see
:ref:`random_subset`), and so it makes sense for them to keep track of lists of
location coordinates instead of a grid. Furthermore, HoloPy's scattering
functions accept coordinates in the form of a 3xN array of coordinates. In both
of these cases, the 2D image is flattened into a 1D DataArray like that created
by :func:`.detector_points`. In this case the single dimension is 'flat' instead
of 'point'. HoloPy treats arrays with these two named dimensions identically,
except that the 'flat' dimension can be unstacked to restore a 2D image or 3D
volume.

HoloPy's use of DataArrays sometimes assigns smaller DataArrays in ``attrs``,
which can lead to problems when saving data to a file. When saving a DataArray
to file, HoloPy converts any DataArrays in ``attrs`` to numpy arrays, and keeps
track of their dimension names separately. HoloPy's :func:`.save_image` writes a
yaml dump of ``attrs`` (along with spacing information) to the
``imagedescription`` field of .tif file metadata.

:ref:`infer_tutorial` returns a lot of information, which is stored in the form of a :class:`.SamplingResult` object.
This object stores the model and :class:`.EmceeStrategy` that were used in the inference calculation as attributes. 
An additional attribute named ``dataset`` is an `xarray Dataset <http://xarray.pydata.org/en/stable/data-structures.html#dataset>`_ 
that contains both the data used in the inference calculation, as well as the raw output.
The parameter values at each step of the sampling chain and the calculated log-probabilities at each step are stored here under the ``samples`` and ``lnprobs`` namespaces.

.. _scat_theory:

Adding a new scattering theory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adding a new scattering theory is relatively straightforward. You just need to
define a new scattering theory class and implement one or two methods to compute
the raw scattering values::

  class YourTheory(ScatteringTheory):
    def _raw_fields(self, positions, scatterer, medium_wavevec, medium_index, illum_polarization):
      # Your code here

    def _raw_scat_matrs(self, scatterer, pos, medium_wavevec, medium_index):
      # Your code here

    def _raw_cross_sections(self, scatterer, medium_wavevec, medium_index, illum_polarization):
      # Your code here

You can get away with just defining one of _raw_scat_matrs or _raw_fields if you
just want holograms, fields, or intensities. If you want scattering matrices
you will need to implement _raw_scat_matrs, and if you want cross sections, you
will need to implement _raw_cross_sections. We seperate out _raw_fields from
_raw_scat_matrs because we want to provide a faster fields implementation for
mie and multisphere (and you might want to for your theory).

You can look at the Mie theory in HoloPy for an example of calling Fortran
functions to compute scattering (C functions will look similar from the python
side) or DDA for an an example of calling out to an external command line tool
by generating files and reading output files.

.. _infer_model:

Adding a new inference model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To perform inference, you need a noise model. You can make a new noise model by inheriting from :class:`~holopy.inference.noise_model.NoiseModel`. This class has all the machinery to compute likelihoods of observing data given some set of parameters and assuming Gaussian noise. 

To implement a new model, you just need to implement one function: ``forward``.
This function receives a dictionary of parameter values and a data shape schema (defined by :func:`.detector_grid`, for example) and needs to return simulated data of shape specified. See the ``_forward`` function in :class:`~holopy.inference.noise_model.AlphaModel` for an example of how to do this. 

If you want to use some other noise model, you may need to override _lnlike and define the probablity given your uncertainty. You can reference _lnlike in :class:`~holopy.inference.noise_model.NoiseModel`.

.. _nose_tests:

Running Tests
~~~~~~~~~~~~~
HoloPy comes with a suite of tests that ensure everything has been
built correctly and that it's able to perform all of the calculations
it is designed to do.  To run these tests, navigate to the root of the
package (e.g. ``/home/me/holopy``) and run:

.. sourcecode:: bash

   python run_nose.py

