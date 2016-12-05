.. _dev_tutorial

Developer's Guide
=================

Adding a new scattering theory
------------------------------

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

You can look at the Mie theory in holopy for an example of calling fortran
functions to compute scattering (c functions will look similar from the python
side) or DDA for an an example of calling out to an external command line tool
by generating files and reading output files.

Useful Functions
----------------

Holopy.core (particularly holopy.core.math, holopy.core.utils, and
holopy.core.process) contain a number of useful utility functions. If you are
writing code adjacent to holopy, consider looking through those functions to see
if any of them will save you trouble.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

