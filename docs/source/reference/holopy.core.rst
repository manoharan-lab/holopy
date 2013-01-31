core Package
============

.. automodule:: holopy.core

:mod:`.marray` Module
---------------------

Classes
^^^^^^^
* :class:`.Image`
* :class:`.ImageSchema`
* :class:`.Marray`
* :class:`.Schema`
* :class:`.Volume`
* :class:`.VolumeSchema`

Functions
^^^^^^^^^
* :func:`.arr_like`
* :func:`.resize`
* :func:`.subimage`
* :func:`.zeros_like`

:mod:`.metadata` Module
-----------------------

* :class:`.Optics` class
* :class:`.Angles` class

	   

:mod:`io` Package
-----------------

.. autofunction:: holopy.core.io.io.load
	:noindex:
				  
.. autofunction:: holopy.core.io.io.save   
	:noindex:

:mod:`math` Module
------------------

Functions for mathematical transformation

* :func:`.fft`
* :func:`.ifft`
* :func:`.rotation_matrix`

:class:`.HoloPyObject` Class
----------------------------

.. autoclass:: holopy.core.holopy_object.HoloPyObject

:mod:`.process` Module
----------------------

Most important functions

* :func:`.center_find`
* :func:`.normalize`


.. toctree::
    :hidden:

    holopy.core.marray
    holopy.core.metadata
    holopy.core.math
    holopy.core.process

