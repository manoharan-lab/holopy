# Mayavi moved namespaces in the upgrade to 2.4, this try black will allow using
# either the new or old namespace.
try:
    from mayavi import mlab
except ImportError:
    from enthought.mayavi import mlab

def show_sphere_cluster(s,color):
    # I think scale factor needs to be 2 because mayavi probably interprets 4th
    # argument as a diameter, we keep track of radii
    mlab.points3d(s.x, s.y, s.z, s.r, scale_factor=2.0, resolution=32,color=color)

def volume_contour(d, voxel):
    vol = mlab.pipeline.scalar_field(d)
    vol.spacing = voxel
    contours = mlab.pipeline.contour_surface(vol)
