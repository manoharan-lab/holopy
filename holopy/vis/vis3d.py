from matplotlib import cm
from numpy import arange

# Mayavi moved namespaces in the upgrade to 2.4, this try black will allow using
# either the new or old namespace.
try:
    from mayavi import mlab
except ImportError:
    from enthought.mayavi import mlab

def show_sphere_cluster(s,color):
    # scale factor is 2 because mayavi interprets 4th
    # argument as a diameter, we keep track of radii
    # view is chosen to be looking from the incoming laser's point of view
    if color == 'rainbow':
        for i in arange(0,len(s.x)):
            numberofcolors = max(10,len(s.x))
            mlab.points3d(s.x[i], s.y[i], s.z[i], s.r[i], 
                scale_factor=2.0, resolution=32, 
                color=cm.gist_rainbow(float(i)/numberofcolors)[0:3])
            mlab.view(-90,0,s.z[:].mean())
    else:
        mlab.points3d(s.x, s.y, s.z, s.r, scale_factor=2.0, 
            resolution=32, color=color)
        mlab.view(-90,0,s.z[:].mean())

def volume_contour(d, voxel):
    vol = mlab.pipeline.scalar_field(d)
    vol.spacing = voxel
    contours = mlab.pipeline.contour_surface(vol)
