Installing Holopy
=========================

Dependencies
------------

Holopy requires:

* python 2.6 or 2.7
* numpy
* scipy
* `PyYAML <http://pypi.python.org/pypi/PyYAML/>`_
* `ordereddict <http://pypi.python.org/pypi/ordereddict>`_ (not required if you have python 2.7 or above)

For interactive use we suggest:

* ipython (better python terminal)
* matplotlib (plotting for python)
* mayavi2 (if you want to do 3d plotting)

Optional dependencies for certain calculations:

* `a-dda <http://code.google.com/p/a-dda/>`_ (Discrete Dipole calculations of arbitrary scatterers)
* `OpenOpt <http://openopt.org>`_ (More minimizers)

And to run the tests you need:

* python-nose

For windows users, the `Enthought Python Distribution <http://www.enthought.com/products/epd.php>`_
should have everything you need for holopy and more (except a-dda). 

If you want to build Holopy from source there are a few other python
dependencies.  You will also need C and Fortran compilers.  Please see
:ref:`building`.

Installing
----------

The simplest way to install is to download a binary build.  [TODO:
Some info on downloading binary packages (.tgz and .zip) for various
platforms] 

.. _building:

Building
--------

In addition to the python packages above, you will need the following
to build:

* a fortran 90 compiler (gfortran works, but not version 4.6)
* a fortran 77 compiler (gfortran works, but not version 4.6)
* a c compiler (gcc works)
* numpy.distutils (should ship with numpy)
* f2py (should ship with numpy)
* python development package (specifically python.h)

If you want to generate the documentation, you'll also need

* sphinx (python package for generating documentation from docstrings)
* a LaTeX distribution (to generate the equations in the documentation) - Note: many smaller distributions do not include utf8.def which is required for sphinx, so you may need to install extra packages
* dvipng

To build holopy run ``python setup.py build`` in the root directory.
This will generate a build directory and put all the modules there.

To compile the documentation run ``make html`` from the docs directory
(or type ``make`` to see the different kinds of formats you can
create).  This will generate documentation in the ``docs/build``
directory.




Instructions for Users
^^^^^^^^^^^^^^^^^^^^^^

If you want to compile and install the binary build of holopy on your
system, unpack the source archive and run

``python setup.py install``

as root.  This will build the packages and install holopy in the local
dist-packages directory of your python installation, so that python
will automatically be able to find it when you type "import holopy".

If you are a developer, you might not want to do this because you
might eventually find yourself with two versions of holopy on your
system, one installed globally and one installed locally.  An
alternative is below.


Instructions for Developers
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are going to hack on holopy, you probably only want to compile
the scattering extensions, but not install the module globally on your
system.  Let's say you unpack the source archive in
``/home/me/holopy``.  Then cd to ``/home/me/holopy`` and run

``python setup.py build_ext --inplace``

This puts the extensions inside the source tree, so that you can work
directly from ``/home/me/holopy``.  You will need to add
``/home/me/holopy`` to your ``python_path`` for python to find the
module when you import it.

Testing
-------
Holopy comes with a suite of tests that will ensure that all everything has built correctly and it is able to preform all of the calculations it should be able to.
To run these tests navigate to the root of the package and run:

.. sourcecode:: bash

    $ nosetests -a '!slow' scatterpy/tests holopy/tests


