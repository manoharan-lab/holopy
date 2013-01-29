.. _build_env:

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

