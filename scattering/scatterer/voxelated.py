# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca W. Perry,
# Jerome Fung, and Ryan McGorty
#
# This file is part of Holopy.
#
# Holopy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Holopy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Holopy.  If not, see <http://www.gnu.org/licenses/>.
"""
Generalized scatterers, represented by discretizing space into voxels.  

.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.harvard.edu>
"""
from __future__ import division

import tempfile
import numpy as np
import holopy as hp
from copy import copy
from scatterpy.scatterer import Scatterer, Sphere
from holopy.process.math import rotation_matrix
from scatterpy.scatterer.abstract_scatterer import SingleScatterer

class VoxelatedScatterer(SingleScatterer):
    """
    TODO: need a docstring
    """
    
    def __init__(self, voxels, center, voxels_per_wavelen):
        """
        
        Arguments:
        :param voxels: index of refraction at each voxel
        :type voxels: 3d array of floats

        :param center: vector from the from the imaging origin to the center of
        the scatterer
        :type center: (float, float, float)
        
        """
        self.voxels = voxels
        self.n_bins = 10
        self.center = center
        self.voxels_per_wavelen = voxels_per_wavelen
        
        # leave off the last element of the linspace because nothing will be
        # larger than it, and the first element because we treat things in the
        # smallest bin as background and don't need to write entries for them
        self.bins = np.linspace(self.voxels.min(), self.voxels.max(),
                                self.n_bins)[1:-1] 

        self.binned_voxels = np.zeros(self.voxels.shape,dtype='int')
        # can skip first bin because we initialize to 0
        for i in range(1, len(self.bins)):
            self.binned_voxels[self.voxels > self.bins[i]] = i

        self.bin_indicies = np.zeros_like(self.bins)
        for i in range(len(self.bin_indicies)):
            self.bin_indicies[i] = self.voxels[self.binned_voxels ==
                                                     i].mean()

        empty_bins = np.isnan(self.bin_indicies)
        if empty_bins.any():
            # remove the empty bins, correcting region numbers
            for i, v in reversed(list(enumerate(empty_bins))):
                if empty_bins[i]:
                    self.binned_voxels[self.binned_voxels > i] -= 1

            self.bin_indicies = np.array([b for b in self.bin_indicies if not
                                          np.isnan(b)])
            
    def write_adda_file(self, directory):
        '''
        Writes a representation of this scatterer suitable for adda

        Attributes
        ----------
        directory : the directory to write the file to

        Returns
        -------
        outf : an adda geometry file
        '''

        # We could remove the directory argument to write_adda_file and use the
        # default place.  This is handier for debugging though
        outf = tempfile.NamedTemporaryFile(dir = directory, delete=False) 

        multiindex = self.binned_voxels.max() > 1
        if multiindex:
            outf.write('Nmat={0}\n'.format(len(self.bin_indicies)))
        
        for k, plane in enumerate(self.binned_voxels):
            for j, row in enumerate(plane):
                for i, val in enumerate(row):
                    if val > 0:
                        # val == 0 should nominally correspond to background,
                        # adda will assume that for lines we don't write, so
                        # save some time here and in adda by not writing them
                        line = [str(l) for l in [i, j, k]]
                        if multiindex:
                            line = line + [str(val)]
                        outf.write(' '.join(line)+'\n')

        outf.flush()

        return outf


    @property
    def n(self):
        return [b for b in self.bin_indicies if b != 0]


    @property
    def z(self):
        return self.center[2]

class ScattererByFunction(SingleScatterer):
    def __init__(self, test, n, domain, center):
        self.test = test
        self.n = n
        self.domain = domain
        self.center = center
        self._rotation = None

    def _points(self, spacing):
        if self._rotation is not None:
            rotation_matrix = hp.process.math.rotation_matrix(*self._rotation)
        else:
            rotation_matrix = hp.process.math.rotation_matrix(0, 0, 0)
        for i, x in enumerate(np.arange(self.domain[0][0], self.domain[0][1], spacing)):
            for j, y in enumerate(np.arange(self.domain[1][0], self.domain[1][1], spacing)):
                for k, z in enumerate(np.arange(self.domain[2][0], self.domain[2][1],
                                   spacing)):
                    point = np.dot(rotation_matrix, np.array((x, y, z)))
                    
                    if self.test(point):
                        yield i, j, k

    def rotate(self, alpha, beta, gamma):
        new = copy(self)
        new._rotation = (alpha, beta, gamma)
        return new



class SphereIntersection(VoxelatedScatterer):
    def __init__(self, s1, s2, optics):

        # compute the dipole size
        dpl_target = 10*(s1.n.real/optics.index)
        dpl_d = (2*s1.r)/optics.med_wavelen * dpl_target
        dpl_r = int(np.ceil(dpl_d/2))
        dpl = dpl_r * optics.med_wavelen / s1.r
        # decrease the dipole size slightly to get the discritization to look better
        dpl = dpl*1.01
        dpl_size = optics.med_wavelen/dpl

        x, y, z = np.ogrid[-dpl_r:dpl_r+1,-dpl_r:dpl_r+1,-dpl_r:dpl_r+1]

        x = x*dpl_size + s1.x
        y = y*dpl_size + s1.y
        z = z*dpl_size + s1.z

        def in_sphere(x, y, z, s):
            dsq = ((x-s.x)**2+(y-s.y)**2 + (z-s.z)**2)
            return dsq < s.r**2

        sb = np.logical_and(in_sphere(x, y, z, s1), np.logical_not(in_sphere(x, y, z,
                                                                       s2))) 

        sb = sb * s1.n

        super(SphereIntersection, self).__init__(sb, s1.center, dpl)


def setup_grid(dimension, n, optics):
    """
    Set up a grid for DDA calculations

    Parameters
    ----------
    dimension: float or (float, float, float)
        Size of the box in x, y, z or size of a cube.  Same units as optics
    n: float
        Max or average index of the object
    optics: hp.Optics
        Optics that will be used with this

    Returns
    -------
    grid: np.ogrid 
        x, y, z vectors to construct the grid from
    voxels_per_wavelen: float
        Number of voxels per wavelength
    """

    
    if np.isscalar(dimension):
        dimension = np.array([dimension, dimension, dimension])
    else:
        dimension = np.array(dimension)

    n = abs(n)

    dpl = 10*(n/optics.index)
    dpl_size = optics.med_wavelen/dpl
    axes = dimension / dpl_size
    axes = (np.ceil(axes/2)).astype('int')

    grid = np.ogrid[-axes[0]:axes[0], -axes[1]:axes[1], -axes[2]:axes[2]]

    grid = [x*dpl_size for x in grid]
    
    return grid, dpl

def ellipsoid(grid, axes):
    x, y, z = grid
    a, b, c = axes

    return ((x/a)**2 + (y/b)**2 + (z/c)**2) < 1

class Ellipsiod:
    def __init__(self, a, b, c):
        self.axes = np.array(a, b, c)
    def contains(self, point):
        return (point / self.axes)**2 < 1

    
    
    
        
class Pacman(SphereIntersection):
    def __init__(self, n, r, center, bite_r, bite_offset, bite_beta,
                 bite_gamma, optics):
        """
        Return a 3d rotation matrix
        
        Parameters
        ----------
        s: sphere
            Base sphere to modify
        bite_r:
            radius of the bite to take out
        bite_offset:
            distance of inclusion center from s.center
        bite_beta, bite_gamma: float 
            Rotation of bite from +z
        """

        self.r = r
        self.center = center
        self.bite_r = bite_r
        self.bite_offset = bite_offset
        self.bite_beta = bite_beta
        self.bite_gamma = bite_gamma
        self.optics = optics
        
        s = Sphere(n, r, center=center)
        
        offset = np.dot(rotation_matrix(0, bite_beta, bite_gamma), 
                        [0, 0, bite_offset])

        s2 = Sphere(None, bite_r, center=s.center+offset)

        
        super(Pacman, self).__init__(s, s2, optics)
            
            

    parameter_names_list = ['n.real', 'n.imag', 'r', 'x', 'y', 'z', 'bite_r',
                            'bite_offset', 'bite_beta', 'bite_gamma']

    @property
    def parameter_list(self):
        return np.array([self.n[0].real, self.n[0].imag, self.r, self.center[0],
                         self.center[1], self.center[2], self.bite_r,
                         self.bite_offset, self.bite_beta, self.bite_gamma])

    # not a classmethod because the parameter list does not have enough
    # information to make a new one, need to reference an existing
    # pacman to get optics
    def make_from_parameter_list(self, params):
        n = params[0] + 1.0j * params[1]
        r = params[2]
        center = params[3:6]
        
        return Pacman(n, r, center, *params[6:], optics = self.optics)
