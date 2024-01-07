.. _dev_tutorial:

Developer's Guide
=================

.. _dev_install:

Installing HoloPy for developers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you are going to hack on holopy, you probably want to compile the
scattering extensions.

First, download or clone the latest version of HoloPy from GitHub at `https://github.com/manoharan-lab/holopy <https://github.com/manoharan-lab/holopy>`_.

To gather all the dependencies needed to build HoloPy, the simplest approach is
to use the included environment.yml file to make a new conda environment::

  conda env create -f ./environment.yml

Then activate the new environment::

  conda activate holopy-devel

For Windows, if you don't already have Fortran and C compilers installed, you
can edit the environment file to install the ``m2w64-toolchain`` package.

Now you can build and install the package. Let's say you downloaded or cloned
HoloPy to ``/home/me/holopy``. Then open a terminal, ``cd`` to
``/home/me/holopy`` and run::

  python -m pip install --no-build-isolation --editable .

This will build the package and scattering extensions, and it will install a
stub in your current environment that loads the package from the build
directory. If you change the code and re-import holopy, it will be automatically
rebuilt by meson.

**Note for Mac users:**
gfortran may put its library in a place python can't find it. If you get errors
including something like ``can't find /usr/local/libgfortran.3.dynlib`` you can
add symlinks to fix them::

  sudo ln -s /usr/local/gfortran/lib/libgfortran.3.dynlib /usr/local/lib
  sudo ln -s /usr/local/gfortran/lib/libquadmath.3.dynlib /usr/local/lib

**Note for Windows users:**
The above build instructions *should* work with Windows, but if not, you can try
the following, which should work on Windows 10 with an AMD64 architecture
(64-bit) processor.

1. Install `Anaconda <https://www.continuum.io/downloads>`_ with Python 3.6 and
   make sure it is working.
2. Install the C compiler. It's included in `Visual Studio 2015 Community
   <https://www.visualstudio.com/downloads/>`_. Make sure it is working with a C
   helloworld.
3. From now on, make sure any command prompt window invokes the right
   environment conditions for compiling with VC. To do this, make sure
   ``C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat`` is
   added to the system path variable. This batch detects your architecture, then
   runs another batch that sets the path include the directory with the correct
   version of the VC compiler.
4. Install cython and made sure it works.
5. Install `Intel's Fortran compiler
   <https://software.intel.com/en-us/fortran-compilers/try-buy>`_. A good place
   to start is the trial version of Parallel Studio XE. Make sure it is working
   with a Fortran helloworld.
6. Install `mingw32-make
   <https://sourceforge.net/projects/mingw/files/MinGW/Extension/make/>`_, which
   does not come with Anaconda by default.
7. Download or clone the master branch of HoloPy from
   `https://github.com/manoharan-lab/holopy
   <https://github.com/manoharan-lab/holopy>`_.
8. Open the command prompt included in Intel's Parallel Studio. Run
   ``holopy/setup.py``. It is necessay to use Intel's Parallel Studio command
   prompt to avoid compiling errors.
9. Install the following dependencies that don't come with Anaconda::

        conda install xarray dask netCDF4 bottleneck
        conda install -c astropy emcee=2.2.1

10. Open an iPython console where holopy is installed and try ``import holopy``.

If the above procedure doesn't work, or you find something else that does,
please `let us know <https://github.com/manoharan-lab/holopy/issues>`_ so that
we can improve these instructions.

..  _xarray:

How HoloPy stores data
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

:ref:`infer_tutorial` returns a lot of information, which is stored in the form
of a :class:`.SamplingResult` object. This object stores the model and
:class:`.EmceeStrategy` that were used in the inference calculation as
attributes. An additional attribute named ``dataset`` is an `xarray Dataset
<http://xarray.pydata.org/en/stable/data-structures.html#dataset>`_ that
contains both the data used in the inference calculation, as well as the raw
output. The parameter values at each step of the sampling chain and the
calculated log-probabilities at each step are stored here under the ``samples``
and ``lnprobs`` namespaces.

.. _scat_theory:

Adding a new scattering theory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adding a new scattering theory is relatively straightforward. You just need to
define a new scattering theory class and implement one or two methods to compute
the raw scattering values::

  class YourTheory(ScatteringTheory):
    def can_handle(self, scatterer):
      # Your code here

    def raw_fields(self, positions, scatterer, medium_wavevec, medium_index, illum_polarization):
      # Your code here

    def raw_scat_matrs(self, scatterer, pos, medium_wavevec, medium_index):
      # Your code here

    def raw_cross_sections(self, scatterer, medium_wavevec, medium_index, illum_polarization):
      # Your code here

You can get away with just defining one of either ``raw_scat_matrs`` or
``raw_fields`` if you just want holograms, fields, or intensities. If
you want scattering matrices you will need to implement
``raw_scat_matrs``, and if you want cross sections, you will need to
implement ``raw_cross_sections``. We separate out ``raw_fields`` from
``raw_scat_matrs`` to allow for faster fields calculation for specific
cases, such as the Mie, MieLens, and Multisphere theories (and you might
want to do so for your theory as well); the base
:class:`.ScatteringTheory` class calculates the fields from the
scattering matrices by default.

You can look at the Mie theory in HoloPy for an example of calling Fortran
functions to compute scattering (C functions will look similar from the python
side) or DDA for an an example of calling out to an external command line tool
by generating files and reading output files.

If you want to fit parameters in your scattering theory, you also need
to define a class attribute `parameter_names` that contains the fittable
attributes of the scattering theory. Once you do this, fitting should
work natively with your new scattering theory: you should be able to
specify the parameters as a :class:`prior.Prior` object and `holopy`'s
inference :class:`Model` will auto-detect them as fittable parameters.
For an example of this, see the :class:`.Lens`, :class:`.MieLens`, or
:class:`.AberratedMieLens` classes.


.. _infer_model:

Adding a new inference model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To perform inference, you need a noise model. You can make a new noise model by
inheriting from :class:`~holopy.inference.noise_model.NoiseModel`. This class
has all the machinery to compute likelihoods of observing data given some set of
parameters and assuming Gaussian noise.

To implement a new model, you just need to implement one function: ``forward``.
This function receives a dictionary of parameter values and a data shape schema
(defined by :func:`.detector_grid`, for example) and needs to return simulated
data of shape specified. See the ``_forward`` function in
:class:`~holopy.inference.noise_model.AlphaModel` for an example of how to do
this.

If you want to use some other noise model, you may need to override ``_lnlike``
and define the probablity given your uncertainty. You can reference ``_lnlike``
in :class:`~holopy.inference.noise_model.NoiseModel`.

.. _nose_tests:

Running tests
~~~~~~~~~~~~~
HoloPy comes with a suite of tests that ensure everything has been built
correctly and that it can perform all of the calculations it is designed to do.
To run these tests, navigate to the root of the package (e.g.,
``/home/me/holopy``) and run

.. sourcecode:: bash

   python run_nose.py

Note that you can download the full test holograms by installing ``git lfs`` and
doing::

  git lfs pull

You don't need to do this, but it can be helpful to inspect differences between
the test calculations and the expected holograms if the tests fail.

Notes on the build system
~~~~~~~~~~~~~~~~~~~~~~~~~

We use the `meson build system <https://mesonbuild.com/>`_ with the
`meson-python extension <https://meson-python.readthedocs.io/en/latest/>`_ to
build the scattering extensions and install the package. Previously we used
``numpy.distutils`` but this module has been deprecated. There are a few things
you need to know about the build system in order to ensure that your changes to
HoloPy will work properly.

1. Note that all meson builds are done "out of tree". That means compiled
extensions are not installed into the same directory as their sources.
Note that holopy has several extensions that must be installed in a way that
makes it possible to do (for example)

.. sourcecode:: python

   import holopy.scattering.theory.mie_f.scsmfo_min

To enable this functionality, we need to tell meson to copy the extensions to
the appropriate point in the installation tree, *and* to copy the python files
too. In the subdirectories, you'll see ``meson.build`` files that call
``install_sources()``, which installs the .py files of holopy, and that call
``extension_module(subdir=...)``, which tells meson where to install the
compiled scattering extensions. All files have to be specified, so if you add a
Python file somewhere, you need to update the relevant ``meson.build`` file to
include it in the installation. Having to specify all the files is a big change
from how we did things with ``numpy.distutils``, but it's supposed to make the
build process more efficient.

Currently we do not add the test files or the example data to the installation,
because we'd need to specify a lot of files, and it would add an extra step to
writing new tests.

2. Unlike ``numpy.distutils``, meson doesn't run f2py automatically to
compile the scattering extensions. There is some code in the ``meson.build``
file in the ``mie_f`` directory that will automatically run f2py to generate the
C and Fortran wrappers for the scattering extensions. Have a look at this file
if you're adding a new Fortran extension.


**Gotchas**

* If you open a Python interpreter or Jupyter notebook in the root of the
  repository, remember that Python will see the subdirectory ``holopy`` as a
  package. So even if you haven't built the package with meson, ``import
  holopy`` might work, and will probably give you a lot of unexpected results
  (like the scattering theories being missing). Remember that meson builds do
  not happen in the source tree. To check whether you have actually built and
  installed the package, try to import it from a directory that
  does not have the ``holopy`` source tree as a subdirectory.
* To run the tests, however, you *do* need your current working directory to be
  inside the source tree. This is because the tests are not installed with the
  package.
* All python files that include tests that use multiprocessing *must* be added
  to ``install_sources()`` in the relevant ``meson.build`` file. This is because
  the multiprocessing module needs to do some pickling, and it tries to import
  the test file as a module. This is the exception to the rule that we do not
  include test files in the installation. If you don't install the file
  containing the tests, you might see that ``pytest`` hangs on the test. Doing
  ``pytest -s`` is a good way to debug any hanging tests. It runs pytest, but it
  shows all the output (stdout and stderr) from the code. If you see a
  ``ModuleNotFoundError`` from the ``multiprocessing`` package, you need to
  include your test file in the installation. So, for example, we have to
  include ``/holopy/inference/tests/test_cma.py`` in
  ``/holopy/inference/tests/meson.build`` because it relies on the ``cmaes``
  module, which uses ``multiprocessing``.
