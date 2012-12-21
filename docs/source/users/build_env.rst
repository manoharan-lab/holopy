.. _build_env:

Setting up a Holopy Build Environment
=====================================

Linux (Ubuntu)
--------------

You can install all of the packages you need to build and test holopy and its documentation with::

  sudo apt-get install mayavi2 python-scipy bzr gfortran ipython python-matplotlib python-yaml python-dev texlive-fonts-extra texlive-fonts-recommended texlive-latex-extra texlive-science 


It will likely be a similar list of packages for other linuxes. 

Windows
-------

#. Install the `MingW compiler <http://sourceforge.net/projects/mingw/files/Installer/mingw-get-inst/mingw-get-inst-20120426/mingw-get-inst-20120426.exe/download>`_. Make sure to install gfortran when prompted to select compilers.
#. Install the `Enthought Python Distribution <http://www.enthought.com/products/epd.php>`_
#. Install `PyYAML <http://pypi.python.org/pypi/PyYAML/>`_

You are on your own with respect to building the documentation on windows, let us know what it took if you succeed. 

Mac
---

#. Install the `XCode Tools <https://developer.apple.com/xcode/>`_. This includes an incomplete version of gcc, the Gnu compiler collection. You will need also to download the optional "Command Line Tools" (see the"Downloads" panel under "Preferences").
#. Install gfortran. Follow the instructions to install the appropriate binaries from HPC `here <http://hpc.sourceforge.net/>`_. (these aren't available through macports)
#. Install the required python

   #. With prepared packages (Easier)

        #. `Enthought Python Distribution <http://www.enthought.com/products/epd.php>`_ 
        #. `PyYAML <http://pypi.python.org/pypi/PyYAML/>`_

   #. With macports. This will give you a cleaner setup overall, but
      you are on your own getting macports working (someone should check me that these are the correct port packages to install, I don't have a mac to check them on)::

        port install py27-scipy py27-ipython py27-matplotlib py27-yaml
        

You are on your own with respect to building the documentation on mac, let us know what it took if you succeed. 

