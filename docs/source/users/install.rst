.. _install:

Installing HoloPy
=================

Quick Start
-----------

If you already have scientific python installed, getting holopy set up
should be as simple as grabbing a binary package from our `Downloads Page
<http://www.manoharan.seas.harvard.edu/holopy/downloads.html>`_

Unpack the archive, then, from a terminal, as root/adiminstrator, in
the archive directory, run::
  
  python setup.py install

Or put the archive directory in your PYTHONPATH to import it directly
from the archive.

Once you have done that, start up python (we would suggest as
``ipython --pylab`` or the pylab console from EPD or similar) and run::

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

.. toctree::
   build_env

`Download
<http://www.manoharan.seas.harvard.edu/holopy/downloads.html>`_ and
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
