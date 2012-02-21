# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca
# W. Perry, Jerome Fung, and Ryan McGorty
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
Generalized scatterers, represented by discritizing space into voxels.  

.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.harvard.edu>
"""

import tempfile
import numpy as np
from scatterpy.scatterer import Scatterer


class GeneralScatterer(Scatterer):
    """
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
