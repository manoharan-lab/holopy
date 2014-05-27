.. _changes:

**********
What's New
**********

2.1
***

Random Subset fitting
---------------------

Speed up fits to holograms by ~10x with::

   result = fit(model, data, random_subset=.1)

This will conduct while only computing a random ten percent of the
pixels in the hologram.


Internal Fields with Mie Theory
-------------------------------

Model.guess_holo and FitResult.fitted_holo
------------------------------------------

Improved time series fitting
----------------------------

GUI for playing with holograms
------------------------------
