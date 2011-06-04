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
'''
The nmpfit minizizer specific machinery for fitting.  

Author:
Jerome Fung (fung@physics.harvard.edu)
Tom Dimiduk (tdimiduk@physics.harvard.edu)
'''

import numpy as np
from holopy.third_party import nmpfit
from holopy.utility.errors import NotImplemented

class NmpfitParam(object):
    '''
    Class for a parameter to be fit by nmpfit.
    '''
    def __init__(self, value = None, fixed = False, limits = [None, None],
                 step = 0., maxstep = 0., name = None):
        self.value = value
        self.fixed = fixed
        self.limits = limits
        self.step = step
        self.deriv_side = 0
        self.maxstep = maxstep
        self.name = name
        # tying parameters in nmpfit does not work, so work around
        # ourselves 
        # may need error-checking code b/c tying only makes sense for a few
        # params
        # self.tied = 'param' ties to value of parameter 'param'
        self.fit_value = None # To be set only by a fit!
        self.fit_error = None

    def parinfo_dict(self):
        '''
        Makes PARINFO dictionary needed to pass to nmpfit.
        '''
        pardict = {}

        # minimally needs to have a value
        try:
            pardict['value'] = float(self.value)
        except TypeError:
            print "Error: Parameter " + self.name + " initial value not set!"
            return pardict # check for empty dictionaries

        if self.fixed:
            return {}

        # Limits
        pardict['limited'] = [x is not None for x in self.limits]
        pardict['limits'] = [0, 0]
        for i in range(2):
            if self.limits[i] is not None:
                pardict['limits'][i] = self.limits[i]
        
        if self.name:
            pardict['parname'] = self.name
        if self.step:
            pardict['step'] = self.step
        if self.deriv_side:
            pardict['mpside'] = self.deriv_side
        if self.maxstep:
            pardict['mpmaxstep'] = self.maxstep

        return pardict

    def output_dict(self):
        '''
        Produce dictionary suitable for dumping to yaml file.
        Call this before update_params!
        '''
        out_pdict = {'initial_value' : float(self.value),
                     'fixed' : self.fixed,
                     'tied' : getattr(self, 'tied', None),
                     'deriv_side' : self.deriv_side}
 
        if self.step: # handle both float values and None
            out_pdict['step'] = float(self.step)
        else:
            out_pdict['step'] = self.step
        if self.maxstep:
            out_pdict['maxstep'] = float(self.maxstep)
        else:
            out_pdict['maxstep'] = self.maxstep

        # write fitted param value and error if fitted, None if not
        if (not self.fixed) and (not getattr(self,'tied', None)):
            out_pdict['final_value'] = float(self.fit_value)
            out_pdict['final_error'] = float(self.fit_error)
        else:
            out_pdict['final_value'] = float(self.value)
            # doesn't handle tying, this can't be done w/o knowing
            # param it's tied to
            out_pdict['final_error'] = None
        if self.limits.__class__ == list:
            out_pdict['limits'] = self.limits
        elif self.limits.__class__ == np.ndarray:
            out_pdict['limits'] = self.limits.tolist()
        else:
            raise NotImplemented('Limits cannot be handled in output')
        
        return out_pdict

    def __str__(self):
        return ("NmpfitParam(name: {0.name}, value: {0.value}\n" +
                "            fixed: {0.fixed}, limits: {0.limits}, "+
                "step: {0.step}, maxstep: {0.maxstep},\n" +
                "            deriv_side: {0.deriv_side}\n"
                "            fit_value: {0.fit_value}, fit_error: {0.fit_error})"
                ).format(self)

    
class TiedNmpfitParam(NmpfitParam):
    def __init__(self, tied_par, name):
        # this is a reference to the nmpfit_param object this object
        # is tied to 
        self.tied_par = tied_par
        # tied_nmpfit_param has only two variables, everything else passes
        # through to nmpfit_param
        self.name = name
        self.tied = self.tied_par.name

    # Pass through all parameter requests except for .name or .tied to
    # the tied parameter 
    def __getattr__(self, name):
        return getattr(self.tied_par, name)

    def __str__(self):
        return ("TiedNmpfitParam(name: {0.name}, value: {0.value}, " +
                "tied_to: {0.tied}\n" +
                "                fixed: {0.fixed}, limits: {0.limits}, "+
                "step: {0.step}, maxstep: {0.maxstep},\n" +
                "                deriv_side: {0.deriv_side}\n"
                "                fit_value: {0.fit_value}, "+
                "fit_error: {0.fit_error})"
                ).format(self)
    
def _minimize_nmpfit(residfunct, parinfo, ftol = 1e-10, xtol = 1e-10, 
                     gtol = 1e-10, damp = 0, maxiter = 100, quiet = False):

    fitresult = nmpfit.mpfit(residfunct, parinfo = parinfo, 
                             ftol = ftol, xtol = xtol, gtol = gtol, 
                             damp = damp, maxiter = maxiter, 
                             quiet = False)

    return NmpfitResult(fitresult,ftol, xtol, gtol, damp, maxiter)

class NmpfitResult(object):
    def __init__(self, fitresult, ftol, xtol, gtol, damp, maxiter):
        self.raw_nmpfit_result = fitresult
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.damp = damp
        self.maxiter = maxiter

    def fitter_state_dict(self):
        nmpfit_dict = {'ftol' : self.ftol, 
                       'xtol' : self.xtol,
                       'gtol' : self.gtol,
                       'damp' : self.damp,
                       'maxiter' : self.maxiter,  
                       'status' : self.raw_nmpfit_result.status,
                       'fnorm' : float(self.raw_nmpfit_result.fnorm),
                       'covar' : self.raw_nmpfit_result.covar.tolist(), 
                       'nfev' : self.raw_nmpfit_result.nfev,
                       'niter' : self.raw_nmpfit_result.niter,
                       'params' : self.raw_nmpfit_result.params.tolist(),
                       'perror' : self.raw_nmpfit_result.perror.tolist(),
                       'pnames' : [p.name for p in self.fitted_pars]}
        return nmpfit_dict
