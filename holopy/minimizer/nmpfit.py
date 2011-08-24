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
Wrapper around nmpfit to allow it to accept input in our unified format

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>

"""

from holopy.third_party import nmpfit

def minimize(guess, residual, lb=None , ub=None, ftol = 1e-10, xtol = 1e-10,
              gtol = 1e-10, damp = 0, maxiter = 100, quiet = False, err=None):

    def resid_wrapper(p, fjac=None):
        status = 0
        return [status, residual(p)]

    parinfo = []
    for i, par in enumerate(guess):
        parinfo.append({'limited' : [True, True],
                        'limits' : [lb[i], ub[i]],
                        'value' : par})

    fitresult = nmpfit.mpfit(resid_wrapper, parinfo=parinfo, ftol = ftol,
                             xtol = xtol, gtol = gtol, damp = damp,
                             maxiter = maxiter, quiet = quiet)

    print(fitresult.fnorm)

    return fitresult.params
