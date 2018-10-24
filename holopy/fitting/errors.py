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
import warnings

class ParameterSpecificationError(Exception):
    pass

class MinimizerConvergenceFailed(Exception):
    def __init__(self, result, details):
        self.result = result
        self.details = details

class InvalidMinimizer(Exception):
    pass

def fit_warning(correct_obj):
        msg = "HoloPy's fitting API is deprecated. \
        Use a {} object instead.".format(correct_obj)
        warnings.warn(msg, UserWarning)
        pass
