.. _dda_tutorial:

*****************************************
Using DDA (Discrete Dipole Approximation)
*****************************************

If a light scattering model for your scatterer does not exist, you can still visualize how your object scatters light by using a discrete dipole approximation, often referred to as DDA. This technique subdivides an object into little 3D blocks, or 'voxels', and solves Maxwell's equations whilst treating each voxel as a dipole. It can be used for arbitrary shapes, and multiple media. We use the DDA code first developed in University of Amsterdam called ADDA (http://code.google.com/p/a-dda/), and it is now hooked up to HoloPy. To use DDA with HoloPy, you must first install `ADDA <http://code.google.com/p/a-dda/>`_.

Defining the geometry of the scatterer
======================================

To calculate the scattering pattern for an arbitrary object, you first need a function which outputs 'True' if a test coordinate lies within your scatterer, and 'False' if it doesn't.

An example for a sphere is as follows. Basically, we define a radius, and if the distance between the test point and the center of your object is less than the radius, the code will return 'True.' ::

	class Sphere:
	    def __init__(self, center, Radius):
		# store properties as arrays for easy numerical computation
		self.center = np.array(center)
		self.RadiusSq = Radius*Radius
		
	    def isPtIn(self, pt):
		# vector center to pt
		delta = np.array(pt) - self.center
		
		# check if we're within the specified distance from the center
		distSq = np.dot(delta, delta)
		if distSq <= self.RadiusSq:
			return True
		else:
			return False

Creating the scatterer
======================

The next thing to do is create your scattering object by creating a box of test points, and testing the points against your function to create a 3D array of 'True's and 'False's. First, import the following: ::

  import holopy as hp
  import numpy as np
  from holopy.core import ImageSchema, Optics
  from holopy.scattering.scatterer.voxelated import ScattererByFunction, MultidomainScattererByFunction
  from holopy.scattering.theory import DDA

Then, create an appropriate image schema, and use the function ScattererByFunction to make your scatterer. It expects the following arguments::

  ScattererByFunction(test, index,[[lbx,ubx],[lby,uby],[lbz,ubz]], (x,y,z))

where test is the function that you wrote (and should return True or False when tested against a point), index is the refractive index of your object, lb/ub are the lower and upper bounds for the box of test points that will be used to test your function with, and (x,y,z) is the center of the box. Choose the center and the box bounds such that it is the smallest box that will fit your scatterer. This will ensure we're not wasting time by testing points that are definitely not in the scatterer. Last, we give the schema and the scatterer to DDA so it can calculate a hologram. The code should look something like this: ::

  if __name__ == '__main__':
	  o = hp.core.Optics(wavelen=.66, index=1.411, pixel_scale=.1, polarization=(1,0))
	  schema = ImageSchema(shape = 100, spacing = o.pixel_scale, optics = o)
	  x = Sphere(np.array([0,0,0]), .5)
	  s = ScattererByFunction(x.isPtIn, 1.585+0j,[[-1.25,1.25],[-1.25,1.25],[-1.25,1.25]], (5,5,5))
	  holo = DDA.calc_holo(s, schema)

Put both bits of code above into a file and name is 'test.py'. In your iPython shell, run the code and take a look at your sphere: ::

  run test.py
  hp.show(holo)


Examples
========

Janus particle
--------------
The Janus particle is simply a hemisphere with sphere (i.e. a sphere that's been coated with something on one half only).  The two halves have different refractive indices. Instead of using ScattererByFunction which takes a single test and refractive index, we use MultidomainScattererByFunction, which makes a composite scatterer froma list of tests and indices. The syntax is much the same as single-medium scatterers. ::

	import numpy as np

	class HemisphericalShell:
	    def __init__(self, center, normal, innerRadius, outerRadius):
		# store properties as arrays for easy numerical computation
		self.center = np.array(center)
		self.normal = np.array(normal)
		self.innerRadiusSq = innerRadius*innerRadius
		self.outerRadiusSq = outerRadius*outerRadius
		
	    def isPtIn(self, pt):
		# vector center to pt
		delta = np.array(pt) - self.center
		
	       	# check which side of the plane we're on
		if np.dot(delta, self.normal) < 0 : 
			return False
	
		# check if we're within the specified distance from the center
		distSq = np.dot(delta, delta)
		if distSq >= self.innerRadiusSq and distSq <= self.outerRadiusSq:
			return True
		else:
			return False

	class Sphere:
	    def __init__(self, center, Radius):
		# store properties as arrays for easy numerical computation
		self.center = np.array(center)
		self.RadiusSq = Radius*Radius
		
	    def isPtIn(self, pt):
		# vector center to pt
		delta = np.array(pt) - self.center
		
		# check if we're within the specified distance from the center
		distSq = np.dot(delta, delta)
		if distSq <= self.RadiusSq:
			return True
		else:
			return False


	if __name__ == '__main__':
	    from holopy.scattering.scatterer.voxelated import ScattererByFunction, MultidomainScattererByFunction
	    from holopy.scattering.theory import DDA
	    import holopy as hp
	    import numpy as np
	    from holopy.core import ImageSchema, Optics
	    o = hp.core.Optics(wavelen=.66, index=1.411, pixel_scale=.1, polarization=(1,0))
	    x = HemisphericalShell(np.array([0,0,0]), np.array([1,0,0]), .5, .51)
	    y = Sphere(np.array([0,0,0]), .5)
	    s = MultidomainScattererByFunction([x.isPtIn, y.isPtIn], [1.5+0j, 2.5+0j],[[-1.25,1.25],[-1.25,1.25],[-1.25,1.25]], (5,5,5))
	    schema = ImageSchema(shape = 100, spacing = o.pixel_scale, optics = o) 
	    holo = DDA.calc_holo(s, schema)



Saving Results
~~~~~~~~~~~~~~

You will most likely want to save the fit result ::

  holopy.save('result.yaml', result)

This saves all of the information about the fit to a yaml text
file.  These files are reasonably human readable and serve as our archive format for data.  They can be loaded back into python with ::

  loaded_result = holopy.load('result.yaml')
