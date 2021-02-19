.. _install:

Getting Started
===============

Installation
~~~~~~~~~~~~

As of version 3.0, HoloPy supports only Python 3. We recommend using the
`anaconda <https://www.continuum.io/anaconda-overview>`_ distribution of Python,
which makes it easy to install the required dependencies. HoloPy is available on
`conda-forge <https://conda-forge.github.io/>`_, so you can install it with::

  conda install -c conda-forge holopy

in a shell, terminal, or command prompt. Once you have HoloPy installed, open an
IPython console or Jupyter Notebook and run::

  import holopy

If this line works, skip to :ref:`usage` before diving into the tutorials.

You can also build HoloPy from source by following the instructions for :ref:`dev_install`.

.. _dependencies:

Dependencies
------------

HoloPy's hard dependencies can be found in `requirements.txt <https://github.com/manoharan-lab/holopy/blob/master/requirements.txt>`_.
Optional dependencies for certain calculations include:

* `a-dda <http://code.google.com/p/a-dda/>`_ (Discrete Dipole calculations of arbitrary scatterers)

* `mayavi2 <http://docs.enthought.com/mayavi/mayavi/>`_ (if you want to do 3D plotting [experimental])

.. _usage:

Using HoloPy
~~~~~~~~~~~~

You will probably be most comfortable using HoloPy in Jupyter (resembles
Mathematica) or Spyder (resembles Matlab) interfaces. HoloPy is designed to be used with an
interactive backend. In the console, try running::

    from holopy import check_display
    check_display()

You should see an image, and you should be able to change
the square to a circle or diamond by using the left/right arrow keys. If you
can, then you're all set! Check out our :ref:`load_tutorial` tutorial to start
using HoloPy. If you don't see an image, or if the arrow keys don't do anything,
you can try setting your backend with *one* of the following::

    %matplotlib tk
    %matplotlib qt
    %matplotlib gtk
    %matplotlib gtk3

Note that these commands will only work in an IPython console or Jupyter
Notebook. If the one that you tried gave an ``ImportError``, you should restart
your kernel and try another. Note that there can only be one matplotlib backend
per ipython kernel, so you have the best chance of success if you restart your
kernel and immediately enter the ``%matplotlib`` command before doing anything
else. Sometimes a backend will be chosen for you (that cannot be changed later)
as soon as you plot something, for example by running ``test_disp()`` or
:func:`.show`. Trying to set to one of the above backends that is not installed
will result in an error, but will also prevent you from setting a different
backend until you restart your kernel.

An additional option in Spyder is to change the backend through the menu: Tools
> Preferences > IPython console > Graphics. It will not take effect until you
restart your kernel, but it will then remember your backend for future sessions,
which can be convenient.

Additional options for inline interactive polts in jupyter are::

    %matplotlib nbagg
    %matplotlib widget
