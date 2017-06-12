..  _dev_tutorial

Developer's Guide
=================

.. _xarray:

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
yaml dump of `attrs`` (along with spacing information) to the
``imagedescription`` field of .tif file metadata.

-TODO: how inference results are saved

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
just want holograms, fields, or intensities. If you want scattering matricies
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
TODO by Tom.
Also need to refer to this somewhere in the inference tutorial.

.. _nose_tests:

Running Tests
~~~~~~~~~~~~~
HoloPy comes with a suite of tests that ensure everything has been
built correctly and that it's able to perform all of the calculations
it is designed to do.  To run these tests, navigate to the root of the
package (e.g. ``/home/me/holopy``) and run:

.. sourcecode:: bash

   python run_nose.py

