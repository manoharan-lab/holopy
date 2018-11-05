# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, Anna Wang, Solomon Barkley
#
# This file is part of HoloPy.
#
# HoloPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HoloPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HoloPy.  If not, see <http://www.gnu.org/licenses/>.
"""
Classes for defining models of scattering for fitting

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <jfung@physics.harvard.edu>
"""

from holopy.core.metadata import get_values
from holopy.scattering import calc_holo
from holopy.fitting.errors import fit_warning

def Model(scatterer, calc_func, medium_index=None,
                 illum_wavelen=None, illum_polarization=None, theory='auto',
                 alpha=None, constraints=[]):
    from holopy.inference.model import AlphaModel, ExactModel
    if calc_func is calc_holo:
        fit_warning('hp.inference.AlphaModel')
        if alpha is None:
            alpha = 1.0
        model = AlphaModel(scatterer, None, alpha, medium_index, illum_wavelen,
                        illum_polarization, theory, constraints)
    elif alpha is None:
        fit_warning('hp.inference.ExactModel')
        model = ExactModel(scatterer, calc_func, None, medium_index,
                        illum_wavelen, illum_polarization, theory, constraints)
    else:
        raise ValueError("Cannot interpret alpha for non-hologram scattering \
                            calculations")
    def residual(pars, data):
        return get_values(model.forward(pars, data) - data).flatten()
    setattr(model, '_calc', model.forward)
    setattr(model, 'residual', residual)
    return model

def limit_overlaps(fraction=.1):
    from holopy.inference.model import LimitOverlaps
    return LimitOverlaps(fraction)

