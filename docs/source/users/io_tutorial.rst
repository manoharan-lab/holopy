.. _io_tutorial:
.. _yaml_ref:

**************************************************
Saving and loading your data, metadata and results
**************************************************

HoloPy can save and load all of its objects using `YAML
<http://www.yaml.org/>`_ files.  These are designed to be both human-
and computer-readable.  

You can easily save and load data and results for future use using yaml
files.  For example, to save a hologram, type ::

  holopy.save('holo.yaml', holo)
    
This will save your hologram, including all metadata associated with it such
as its optics and spacing (pixel size), as a plain text file.  (Opening this 
file with a text editor, however, may be difficult; see note below.)  To
reload a saved hologram, simply type ::

  holopy.load('holo.yaml')
  
You do not have to specify optics or spacing for your hologram in when
loading it from a saved image, as all of this information is already
specified in the yaml file.

yaml files also provide a handy way of dealing with metadata.  For example,
you can save an optics object for future use by simply typing ::

  holopy.save('optics.yaml', optics)

optics.yaml is a plain text file that will look something like the
following (though yours will not contain the explanatory comments
beginning with #):

.. sourcecode:: yaml
  
  !Optics
  wavelen: 0.660    # Wavelength of light (in vacuum) used in creating holograms
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

_[TODO: NEED SOME MORE INFORMATION BEFORE THE NOTE BELOW. ADD EXAMPLES OF SAVING/LOADING DATA OBJECTS]:

.. Note::
   
   Data objects are a special case for yaml output because they 
   will likely contain a large array of data.  They can still be 
   saved, but will generate very large files that may not be 
   easily opened in a text editor like other HoloPy yamls.

   For the curious advanced user, what we actually do is put a yaml
   header with optics and other information, and then encode the array
   of data as a .npy binary (as from np.save) all in the same file.  This
   keeps the whole object in a single file, but generates a file
   that is not quite as easy to work with as other yamls.







