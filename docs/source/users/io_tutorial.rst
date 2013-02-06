
.. _yaml_ref:

*********************************
Saving and Loading HoloPy Objects
*********************************

HoloPy can save and load all of its objects using `YAML
<http://www.yaml.org/>`_ files.  These are designed to be both human-
and computer-readable. This makes it easy to store results of a
calculation or fit.

Saving Metadata
===============

HoloPy yaml files provide a handy way of dealing with metadata.  For
example, you can save an optics object for future use with::
  
  import holopy as hp
  optics = hp.core.Optics(.66, 1.33, (1, 0))
  hp.save('optics.yaml', optics)

optics.yaml is a plain text file that will look something like the
following (though yours will not contain the explanatory comments
beginning with #):

.. sourcecode:: yaml
  
  !Optics
  wavelen: 0.66     # Wavelength of light (in vacuum) used in creating holograms
  index: 1.33       # Index of medium
  polarization: [1.0, 0.0]
  divergence: 0.0

You can also simply write this file by hand.  Having a file containing
your optics metadata is convenient since you don't need to enter in
your metadata in every python script you run to analyze or calcuate
data.  You can simply create an :class:`.Optics` object just by loading
the file ::

  optics = holopy.load('optics.yaml')
  holo = holopy.load('image.tif', spacing = .1,  optics = optics)

In fact, it's even easier than that.  :func:`holopy.load
<holopy.core.io.io.load>` will accept the filename of a metadata yaml
file as the argument for the optics parameter and automatically load
the yaml file. ::

  holo = holopy.load('image.tif', spacing = .1, optics='optics.yaml')

This is handy if you have a lot of data that was all created using the
same optical train.

Saving Images
=============

If you have a hologram called ``holo`` from a calculation or preprocessing
that you want to save, you can use::

  holopy.save('holo.yaml', holo)
    
This will save your hologram, including all metadata associated with
it such as its optics and spacing (pixel size), to the file
``holo.yaml``.  (Opening this file with a text editor, however, may be
difficult; see [#marray_yaml]_) To reload a saved hologram, simply
type ::

  holo = holopy.load('holo.yaml')

  
You do not have to specify optics or spacing for your hologram when
loading it from a saved hologram, as all of this information is already
specified in the yaml file.

Saving Fit Results
==================

In the :ref:`fit tutorial <fit_tutorial>` you saved the result of a fit with::

  hp.save('result.yaml', result)

If you examine that file, it will contain things like:

.. sourcecode:: yaml

  !FitResult
  parameters: {alpha: 1.0, 'center[0]': 5.000000000000003, 'center[1]': 5.000000000000004,
    'center[2]': 10.299999999999969}
  scatterer: !Sphere
    n: 1.58
    r: 0.5
    center: [5.000000000000003, 5.000000000000004, 10.299999999999969]
  chisq: 2.8721763211759494e-25
  rsq: 1.0
  converged: true
  time: 5.249035120010376
  model: !Model
    scatterer: !ParameterizedObject
      obj: !Sphere
        n: 1.58
        r: 0.5
        center:
        - !Parameter
          guess: 5.5
          limit: [4, 10]
          name: center[0]
        - !Parameter
          guess: 4.5
          limit: [4, 10]
          name: center[1]
        - !Parameter
          guess: 10
          limit: [5, 15]
          name: center[2]
    theory: !method 'calc_holo of !Mie {compute_escat_radial: true, }'
    alpha: !Parameter
      guess: 0.6
      limit: [0.1, 1]
      name: alpha
  # file truncated
               
You can notice that the result yaml contains the fitted results,
information about the goodness of fit, time to fit, and information
about how the fit was set up. Your file will also contain gory details
about how the minimizer ran, but we have cut them off here to save
space.

You should save these files every time you do a fit that you are
likely to care about again later. They are designed to hold all the
information you might need to repeat a calculation or understand how a
fit proceeded at some later point (like say when you are writing a
paper).

.. rubric:: Footnotes

.. [#marray_yaml] 
   
   :class:`.Image` objects and other :class:`.Marray` can be saved as
   yaml files, but they will be large and cannot easily be viewed in a
   text editor like other HoloPy yamls.

   For the curious advanced user, what we actually do is put a yaml
   header with optics and other information, and then encode the array
   of data as a .npy binary (as from np.save) all in the same file.
   This keeps the whole object in a single file, but generates a file
   that is not technically a valid yaml file. HoloPy can load them
   just fine, some tools (unix's more, some editors) will be able to
   show you the text header (and then gibberish for the binary data).







