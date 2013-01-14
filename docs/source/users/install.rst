.. _install:

Installing HoloPy
=========================

Dependencies
------------

HoloPy requires:

* python 2.7 (or python 2.6 + `ordereddict <http://pypi.python.org/pypi/ordereddict>`_)
* numpy
* scipy
* `PyYAML <http://pypi.python.org/pypi/PyYAML/>`_

For interactive use we suggest (highly suggest in the case of ipython and matplotlib):

* `ipython <http://ipython.org>`_ (better python terminal)
* `matplotlib <http://matplotlib.org>`_ (plotting for python)
* `mayavi2 <http://docs.enthought.com/mayavi/mayavi/>`_ (if you want to do 3d plotting)

Optional dependencies for certain calculations:

* `a-dda <http://code.google.com/p/a-dda/>`_ (Discrete Dipole calculations of arbitrary scatterers)
* `OpenOpt <http://openopt.org>`_ (More minimizers)

For Ubuntu/Debian users::
  
  sudo apt-get install mayavi2 python-scipy ipython python-matplotlib python-yaml

  For windows users or mac users, the `Enthought Python Distribution <http://www.enthought.com/products/epd.php>`_ should have the basics to get you started. 

If you want to build HoloPy from source there are a few other python
dependencies.  You will also need C and Fortran compilers.  Please see
:ref:`building`.

Installing
----------

The simplest way to install is to download a binary build.  Download one from our `Downloads Page <http://www.manoharan.seas.harvard.edu/holopy/downloads.html>`_

Then from a terminal as root/adiminstrator, in the archive directory, run::
  
  python setup.py install

Or put the archive directory in your PYTHONPATH to import it directly from the archive. 

.. _building:

Building
--------

See :ref:`build_env` for platform specific instructions. In general what you need to build holopy (in addition to the packages above) is:

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
HoloPy comes with a suite of tests that will ensure that all everything has built correctly and it is able to preform all of the calculations it should be able to.
To run these tests navigate to the root of the package and run:

.. sourcecode:: bash

    $ python run_tests.py

or 

.. sourcecode:: bash
 
    $ nosetests -a '!slow'


