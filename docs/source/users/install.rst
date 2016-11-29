.. _install:

Installing HoloPy
=================


******************
UNDER CONSTRUCTION
******************
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

.

Quick Start
~~~~~~~~~~~

If you do not already have scientific python, we suggest using
`Anaconda <https://www.continuum.io/downloads>`_. You will want python
2.7 (HoloPy is not yet python 3 compatible).

HoloPy has not had a release in years, so you will probably just want
to download a `zip of the master
<https://github.com/manoharan-lab/holopy/archive/master.zip>`_. We
make a reasonable effort to keep the master in a usable state, so
hopefully it will work for you.

Unpack the archive, then, from a terminal, as root/adiminstrator, in
the archive directory, run::

  python setup.py install

Or put the archive directory in your PYTHONPATH to import it directly
from the archive.

Once you have done that, start up python (we would suggest ipython or the jupyter notebook) and run::

  import holopy

If you get your prompt back without errors, congratulations! You have
successfully installed HoloPy. Proceed to the :ref:`tutorials`. If you
get errors or just want to learn more, keep reading.

.. _dependencies:

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

* `mayavi2 <http://docs.enthought.com/mayavi/mayavi/>`_ (if you want to do 3D plotting)

Optional dependencies for certain calculations:

* `a-dda <http://code.google.com/p/a-dda/>`_ (Discrete Dipole calculations of arbitrary scatterers)

* `OpenOpt <http://openopt.org>`_ (More minimizers)

If you want to build HoloPy from source there are a few other python
dependencies.  You will also need C and Fortran compilers.  Please see
:ref:`building`.

Linux (Ubuntu/Debian)
~~~~~~~~~~~~~~~~~~~~~
.. code-block:: bash

  sudo apt-get install python-scipy ipython python-matplotlib python-yaml mayavi2

Other flavors of linux might have slightly different package names.

Windows/Mac
~~~~~~~~~~~

The `Enthought Python Distribution
<http://www.enthought.com/products/epd.php>`_ should have the basics
to get you started.

.. _building:

Building
--------
Set up to build HoloPy
======================

In general what you need to build holopy (in addition to the basic
:ref:`dependencies`) is:

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

On Linux (Ubuntu)
-----------------

You can install all of the packages you need to build HoloPy and its
documentation and run the HoloPy's tests with::

  sudo apt-get install mayavi2 python-scipy bzr gfortran ipython \
    python-matplotlib python-yaml python-dev texlive-fonts-extra \
    texlive-fonts-recommended texlive-latex-extra texlive-science 


It will likely be a similar list of packages for other linuxes. 

On Windows
----------

#. Install the `MingW compiler
   <http://sourceforge.net/projects/mingw/files/Installer/mingw-get-inst/mingw-get-inst-20120426/mingw-get-inst-20120426.exe/download>`_. Make
   sure to install gfortran when prompted to select compilers.

#. Install the `Enthought Python Distribution
   <http://www.enthought.com/products/epd.php>`_

#. Install `PyYAML <http://pypi.python.org/pypi/PyYAML/>`_

You are on your own with respect to building the documentation on
windows, let us know what it took if you succeed.

On MacOS
--------

#. Install the `XCode Tools
   <https://developer.apple.com/xcode/>`_. This includes an incomplete
   version of gcc, the Gnu compiler collection. You will need also to
   download the optional "Command Line Tools" (see the"Downloads"
   panel under "Preferences").

#. Install gfortran. Follow the instructions to install the
   appropriate binaries from HPC `here
   <http://hpc.sourceforge.net/>`_. (these aren't available through
   macports)

#. Install the required python packages one of two ways

   #. **With prepared packages** 

        #. `Enthought Python Distribution <http://www.enthought.com/products/epd.php>`_ 

        #. `PyYAML <http://pypi.python.org/pypi/PyYAML/>`_

   #. **With macports** 

      This will give you a cleaner setup overall, but you are on your
      own getting macports working (someone should check me that these
      are the correct port packages to install, I don't have a mac to
      check them on)::

        port install py27-scipy py27-ipython py27-matplotlib py27-yaml
        

You are on your own with respect to building the documentation on mac,
let us know what it took if you succeed.
`Download
<https://github.com/manoharan-lab/holopy/archive/master.zip>`_ and
unpack a source build, or
check out the source from launchpad::

  bzr branch lp:holopy

To build HoloPy run (in the root of HoloPy)::

  python setup.py build

This will generate a build directory and put all the modules
there. You can then install HoloPy by running (as administrator)::

  python setup.py install


If you are a developer, you might not want use ``python setup.py
install`` because you might eventually find yourself with two versions
of HoloPy on your system, one installed globally and one installed
locally.  Thus, if you are going to hack on HoloPy, you probably only
want to compile the scattering extensions, but not install the module
globally on your system.  Let's say you unpack the source archive in
``/home/me/holopy``.  Then cd to ``/home/me/holopy`` and run

``python setup.py build_ext --inplace``

This puts the extensions inside the source tree, so that you can work
directly from ``/home/me/holopy``.  You will need to add
``/home/me/holopy`` to your ``python_path`` for python to find the
module when you import it.

Testing
~~~~~~~

HoloPy comes with a suite of tests that ensure everything has been
built correctly and that it's able to perform all of the calculations
it is designed to do.  To run these tests, navigate to the root of the
package (e.g. ``/home/me/holopy``) and run:

.. sourcecode:: bash

   python run_nose.py

or

.. sourcecode:: bash

   nosetests -a '!slow'

There is some extra test data that is not distributed with HoloPy but
can help catch some kinds of bugs. The tests will run just fine
without it, but should you want to run a slightly more thorough test
you can retrieve this data with a script in the ``management`` directory::

  python get_test_golds.py

Building the Docs
~~~~~~~~~~~~~~~~~

To compile the documentation run (from the docs directory)::

  make html

(or type ``make`` to see the different kinds of formats you can
create).  This will generate documentation in the ``docs/build``
directory.  Building the docs requires matplotlib version 1.1
or newer.
