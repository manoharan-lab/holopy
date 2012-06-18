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
Routines for fitting a hologram to an exact solution given an input deck.

.. moduleauthor:: Jerome Fung <jfung@physics.harvard.edu>
.. moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>
.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>

"""
from __future__ import division

import sys
import os
import time
hp_dir = (os.path.split(sys.path[0])[0]).rsplit(os.sep, 1)[0]
sys.path.append(hp_dir)
from scipy import sin, cos, array, pi, sqrt, arcsin, arccos, real, dot
from scatterpy.io import Serializable

import numpy as np

#import minimizers.nmpfit_adapter as minimizer
from scatterpy.errors import (UnrealizableScatterer,
                              InvalidScatterer)

def cost_subtract(holo, calc, selection=None):
    if selection==None:
        return holo - calc
    else:
        return selection*holo - selection*calc
    
def cost_rectified(holo, calc, selection=None):
    if selection==None:
        return abs(holo-1) - abs(calc-1)
    else:
        return abs(selection*holo-1) - abs(selection*calc-1)
        
def fit(holo, initial_guess, theory, minimizer='nmpfit', lower_bound=None,
        upper_bound=None, step = None, tie = None, plot=False, minimizer_params={},
        residual_cost=cost_subtract, selection = None):
    """
    Find a scatterer which best recreates the given holo

    Parameters
    ----------
    holo : :class:`holopy.hologram.Hologram` object
        The hologram to fit to
    initial_guess : (:class:`scatterpy.scatterer.Scatterer`, alpha)
        An initial guess at the scatterer which formed the hologram.  
    theory : :class:`scatterpy.theory.scatteringtheory.ScatteringTheory`
        The scattering theory to use in computing holograms of the scatterer
    minimizer : holopy.minmizer.Minimizer
        The minimizer to use to refine the scatterer to agree with the hologram
    lower_bound, upper_bound : (:class:`scatterpy.scatterer.Scatterer`, alpha)
        The minimum and maximum values which the scatterer can vary
    step: (:class:`scatterpy.scatterer.Scatterer`, alpha)
        Step size for each parameter for the minimizer
    tie: :class:`scatterpy.scatterer.Scatterer`
        Scattering object specifying parameters to tie together, parameters with
        a value of 0 or None are not tied, any parameters with the same value are
        tied together (so specify p1 = 1, p2 = 1 for example to tie parameters
        p1 and p2 together)
    plot : bool
         Whether to show a convergence plot (not available with all fitting
         algorithms)
         

    Notes
    -----
    You must choose a scattering theory which is compatible with the scatterer
    you specified.

    The initial guess fixes the number and type of scatterers, only their
    numerical parameters will be varied to fit the hologram.

    lower_bound and upper_bound should be scatterers of the same character as
    initial_guess.  Each of their parameters is treated individually as a limit
    on the space the fitter will explore.  Thus you can think of the two
    scatterers as describing the lower right and upper left corners of the
    n-dimensional parameter space describing the scatterer.  
    
    """
    time_start = time.time()

    scatterer, alpha = initial_guess

    manager = ParameterManager(initial_guess, lower_bound, upper_bound, step,
                               tie)
    
    if isinstance(theory, type):
        # allow the user to pass the type, we instantiate it here
        theory = theory(imshape = holo.shape, optics = holo.optics)

    
    residual = make_residual(holo, scatterer, theory, manager, residual_cost, selection)

    result, fnorm, status, minimizer_info = minimize(residual, manager,
                                                     minimizer,
                                                     **minimizer_params) 
    

    fit_parameters, fit_alpha = manager.interpret_minimizer_list(result)

    time_stop = time.time()

    # Calculate R^2 here, otherwise FitResult has to know about the theory
    # used to fit.
    fit_scatterer = scatterer.from_parameters(fit_parameters)

    fit_holo = theory.calc_holo(fit_scatterer, fit_alpha)
    Rsquared = 1 - ((holo - fit_holo)**2).sum()/((holo - holo.mean())**2).sum()
    
    return FitResult(fit_scatterer, fit_alpha, 
                     fnorm/(holo.shape[0]*holo.shape[1]), status,
                     time_stop-time_start, minimizer_info, Rsquared)


def make_residual(holo, scatterer, theory, parameter_manager,
                  cost_func=cost_subtract, selection=None):
    """
    Construct a residual function suitable for fitting scatterer to holo using
    theory

    Parameters
    ----------
    holo : :class:`holopy.hologram.Hologram` object
        The hologram to fit to
    theory : :class:`scatterpy.theory.scatteringtheory.ScatteringTheory`
        The scattering theory to use in computing holograms of the scatterer
    initial_guess : :class:`scatterpy.scatterer.Scatterer`
        The scatter which models the hologram
    """
    def residual(p, **keywords):
        parameters, alpha = parameter_manager.interpret_minimizer_list(p)

        # alpha should always be the last parameter, we prune it because the
        # scatterer doesn't want to know about it
        this_scatterer = scatterer.from_parameters(parameters)

        try:
            calculated = theory.calc_holo(this_scatterer, alpha, selection)
        except (UnrealizableScatterer, InvalidScatterer) as e:
            print("Fitter asked for a value which the scattering theory \
thought was unphysical or uncomputable, returning large residual")
            calculated = error = np.ones(holo.shape)*1.e12

        return cost_func(holo, calculated, selection).ravel()

    return residual

def minimize(residual, parameter_manager, algorithm='nmpfit', quiet = False,
             plot = False, ftol = 1e-10, xtol = 1e-10, gtol = 1e-10, damp = 0,
             maxiter = 100, err=None):
    """
    Minmized a function (as defined by residual)

    Parameters
    ----------
    residual : F(parameters) -> ndarray(derivatives)
        The residual function to be minimized
    algorithm : string
        The fitting algorithm to use: valid options are nmpfit, ralg,
        scipy_leastsq, scipy_lbfgsb, scipy_slsqp, or galileo
    guess : ndarray(parameters)
        Initial guess for the fitter.  Must be provided unless you are using a
        global algorithm
    lb, ub : ndarray(parameters)
        Lower and upper bounds on the parameters.  Must be provided if you are
        using a global algorithm
    quiet : bool
        Should the fitting algorithm output its internal feedback information?
    plot : bool
        Should the fitting algorthim show a plot of its convergence (not
        available on all fitters
    ftol, xtol, gtol, damp, maxiter, err: float, float, float, float, int, bool
        nmpfit specific parameters

    Notes
    -----
    All parameters after quiet are specific to specific fitting algorithms, and
    will not be used if you don't select an appropriate fitting algorithm.

    Does on demand importing of fitters since external fitting libraries may not
    be present.  
    """

    openopt_nllsq = ['scipy_leastsq']
    openopt_nlp = ['ralg', 'scipy_lbfgsb', 'scipy_slsqp']
    openopt_global = ['galileo']

    ub = parameter_manager.upper_bound
    lb = parameter_manager.lower_bound
    guess = parameter_manager.initial_guess
    
    if algorithm == 'nmpfit':
        step = parameter_manager.step
        parameter_names = parameter_manager.names(with_scaling=True)
        from holopy.third_party import nmpfit
        def resid_wrapper(p, fjac=None):
            status = 0
            resid = residual(p)
            return [status, residual(p)]

        parinfo = []
        for i, par in enumerate(guess):
            d = {'limited' : [True, True],
                 'limits' : [lb[i], ub[i]],
                 'value' : par}
            if step is not None:
                d['step'] = step[i]
            if parameter_names is not None:
                d['parname'] = parameter_names[i]
            parinfo.append(d)

        fitresult = nmpfit.mpfit(resid_wrapper, parinfo=parinfo, ftol = ftol,
                                 xtol = xtol, gtol = gtol, damp = damp,
                             maxiter = maxiter, quiet = quiet)
        if not quiet:
            print(fitresult.fnorm)
        
        return fitresult.params, fitresult.fnorm, fitresult.status < 4, fitresult

    # Openopt fitters
    openopt_nllsq = ['scipy_leastsq']
    openopt_nlp = ['ralg', 'scipy_lbfgsb', 'scipy_slsqp']
    openopt_global = ['galileo']
    openopt = openopt_nllsq + openopt_nlp + openopt_global
    
    if algorithm in openopt:
        import openopt
        if quiet:
            iprint = 0
        else:
            iprint = 1
        def resid_wrap(p):
            resid = residual(p)
            return np.dot(resid, resid)
        if algorithm in openopt_nlp:
            p = openopt.NLP(resid_wrap, guess, lb=lb, ub=ub, iprint=iprint, plot=plot)
        elif algorithm in openopt_global:
            p = openopt.GLP(resid_wrap, guess, lb=lb, ub=ub, iprint=iprint, plot=plot)
        elif algorithm in openopt_nllsq:
            p = openopt.NLLSP(residual, guess, lb=lb, ub=ub, iprint=iprint, plot=plot)

        r = p.solve(algorithm)
        return r.xf, r.ff, True, r

    else:
        raise MinimizerNotFound(algorithm)

class ParameterManager(object):
    """
    """
    
    def __init__(self, initial_guess, lower_bound=None, upper_bound=None,
                 step=None, tie=None):
        """
        
        Arguments:
        :param initial_guess: 
        :type initial_guess: 
        
        :param lower_bound: 
        :type lower_bound: 
        
        :param upper_bound: 
        :type upper_bound: 
        
        :param tie: 
        :type tie: 
        
        """

        def unpack_bound(b):
            if b is None:
                return None
            return np.append([p[1] for p in b[0].parameters.iteritems()], b[1])

        self._initial_guess = unpack_bound(initial_guess)
        self._names = initial_guess[0].parameters.keys() + ['alpha']
        self._lower_bound = unpack_bound(lower_bound)
        self._upper_bound = unpack_bound(upper_bound)
        self.tie = unpack_bound(tie)
        self._step = unpack_bound(step)

        
        tie_groups = {}
        if self.tie is not None:
            for i, p in enumerate(self.tie):
                if p is not None and p != 0:
                    if tie_groups.has_key(p):
                        tie_groups[p].append(i)
                    else:
                        tie_groups[p] = [i]

        self.tie_groups = list(tie_groups.iteritems())
                

        # check that the initial guess lies within the bounds
        if ((self._initial_guess > self._upper_bound).any() or
            (self._initial_guess < self._lower_bound).any()):
            names = np.array(self.names(prune=False))
            raise GuessOutOfBounds(low=names[self._initial_guess<self._lower_bound],
                                   high=names[self._initial_guess>self._upper_bound])

        self.fixed = (np.array(self._lower_bound) == np.array(self._upper_bound))
        if self.tie is None:
            self.tied = [False for i in range(len(self._initial_guess))]
        else:
            self.tied = np.array([not (p is None or p == 0.0) for p in self.tie])
        self.unusual = np.logical_or(self.fixed, self.tied)

            
                                                              
        self.scale = np.zeros(self._initial_guess.size)
        for i in range(len(self.scale)):
            if self.fixed[i]:
                # fixed parameters never go into the minimizer, so don't bother
                # rescaling them
                self.scale[i] = 1.0
            else:
                self.scale[i] = abs(self._initial_guess[i])
                # if any parameters have an initial value of 0, this way of chosing scale
                # will not work, so instead use one based on the range of allowed values
                if abs(self.scale[i]) < 1e-12:
                    self.scale[i] = (self._upper_bound[i] - self._lower_bound[i])/10


    @property
    def step(self):
        if self._step is None:
            return None
        return self._to_minimizer(self._step)

    @property
    def upper_bound(self):
        return self._to_minimizer(self._upper_bound)

    @property
    def lower_bound(self):
        return self._to_minimizer(self._lower_bound)

    @property
    def initial_guess(self):
        """
        Marshal the parameters for the minimizer, rescaling and removing fixed
        parameters and accounting for tied parameters

        Returns
        -------
        minimizer_list: list
            List of parameters suitable for passing to minimize
        
        """
        return self._to_minimizer(self._initial_guess)

    def _prune(self, p, check=True):
        # figure out the members of each tied group
        tie_values = []
        for group, members in self.tie_groups:
            value = p[members[0]]
            for i in members[1:]:
                if p[i] != value and check:
                    raise TiedParameterValuesNotEqual(self._names[members[0]],
                                                      value,
                                                      self._names[i], p[i])
            tie_values.append(value)
        
        # pull out all the fixed and tied parameters
        pruned = []
        for i, v in enumerate(p):
            if not self.unusual[i]:
                pruned.append(v)

        # now add back in one parameter for each tie group
        return tie_values + pruned
    
    def _to_minimizer(self, p):
        if None not in p:
            p = p/self.scale
        return self._prune(p)
    
    def interpret_minimizer_list(self, minimizer_list):
        """
        Put back in parameters and undo rescalings to get back to the real
        physical form

        Parameters
        ----------
        minimizer_list: list
            List of parameters from the minimizer (should be scaled and
            organized as from self.minimizer_list

        Returns
        -------
        parameters: dict
            dict of parameters suitable for passing to
            scatterer.from_parameters
        alpha: float
            Fitted scaling alpha
        """
        tie_values = minimizer_list[0:len(self.tie_groups)]
        values = minimizer_list[len(self.tie_groups):]

        for i, v in enumerate(self.unusual):
            if v:
                if self.tied[i] != 0 and self.tied[i] is not None:
                    group = 0
                    while i not in self.tie_groups[group][1]:
                        group += 1
                    values = np.insert(values, i, tie_values[group])
                elif self.fixed[i]:
                    values = np.insert(values, i, self._initial_guess[i])
                else:
                    raise ParameterSpecificationError(self.names(prune=False)[i])

        descaled = self.scale * values
        parameters = dict(zip(self._names, descaled))
        alpha = parameters.pop('alpha')
        return parameters, alpha
        return (self.scale * values)

    def names(self, with_scaling=False, prune=True):
        if with_scaling:
            names =  ["{0} (/ {1})".format(self._names[i], self.scale[i]) for i
                      in range(len(self.scale))]
        else:
            names = self._names
        if prune:
            return self._prune(names, check=False)
        else:
            return names

class TiedParameterValuesNotEqual(Exception):
    def __init__(self, p1, v1, p2, v2):
        self.p1 = p1
        self.v1 = v1
        self.p2 = p2
        self.v2 = v2
    def __str__(self):
        return "Parameters: {0} and {2} have their values tied but have \
different values: ({1} and {3}) specified, this is not allowed".format(self.p1,
                                                                      self.v1,
                                                                      self.p2,
                                                                      self.v2)

class FitResult(Serializable):
    def __init__(self, scatterer, alpha, chisq, status, time, 
                 minimizer_info, rsq = None):
        self.scatterer = scatterer
        self.alpha = alpha
        self.chisq = chisq
        self.rsq = rsq
        self.status = status
        self.time = time
        self.minimizer_info = minimizer_info,
    def __getitem__(self, index):
        if index == 0:
            return self.scatterer
        if index == 1:
            return self.alpha
        raise KeyError
    def __repr__(self):
        try:
            return "{s.__class__.__name__}(scatterer={s.scatterer}, \
alpha={s.alpha}, chisq={s.chisq}, rsq={s.rsq}, status={s.status}, \
time={s.time}, minimizer_info={s.minimizer_info})".format(s=self)
        except AttributeError:
            return "{s.__class__.__name__}(scatterer={s.scatterer}, \
alpha={s.alpha}, chisq={s.chisq}, status={s.status}, \
time={s.time}, minimizer_info={s.minimizer_info})".format(s=self)

class FitSetup(Serializable):
    """
    Stores paramaters of how a fit was run
    """
    
    def __init__(self, initial_guess, theory, minimizer, lower_bound,
        upper_bound, step, minimizer_params, residual_cost):
        """
        
        Arguments:
        :param holo: 
        :type holo: 
        
        :param initial_guess: 
        :type initial_guess: 
        
        :param theory: 
        :type theory: 
        
        :param minimizer: 
        :type minimizer: 
        
        :param lower_bound: 
        :type lower_bound: 
        
        :param upper_bound: 
        :type upper_bound: 
        
        :param step: 
        :type step: 
        
        :param minimizer_params: 
        :type minimizer_params: 
        
        :param residual_cost: 
        :type residual_cost: 
        
        """
        self.holo = holo
        self.initial_guess = initial_guess
        self.theory = theory
        self.minimizer = minimizer
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.step = step
        self.minimizer_params = minimizer_params
        self.residual_cost = residual_cost

    
class MinimizerNotFound(Exception):
    def __init__(self, algorithm):
        self.algorthim = algorithm
    def __str__(self):
        return "{0} is not a valid fitting algorithm".format(self.algorthim)

class GuessOutOfBounds(Exception):
    def __init__(self, low, high):
        self.high = high
        self.low = low
    def __str__(self):
        return "Parameters out of range: {0} are below the minimum bounds, and \
{1} are above the maximum bounds".format(self.low, self.high)



def dimer_angles_gap(position1, position2, radius1, radius2):
    """
    Converts the x,y,z position of two beads to the center of
    mass coordinate, their separation distance and the two
    Euler angles.

    Parameters
    ----------
    position1 : three-element tuple or array
        The x,y,z coordinate of the first bead
    position2 : three-element tuple or array
        The x,y,z coordinate of the second bead
    radius1 : float
        First bead's radius
    radius2 : float
        Second bead's radius

    Returns
    -------
    center_of_mass : three-element float array
        The x,y,z coordinate of the center of mass
        of the dimer.
    epsilon_r : float
        The dimensionless separation parameter.
    beta : float
        The Euler angle beta.
    gamma : float
        The Euler angle gamma.

    """
    center_of_mass = 0.5*(position1+position2)
    separation = sqrt(sum((position1 - position2)**2))
    xysep = sqrt(sum((position1[:2] - position2[:2])**2))
    epsilon_r = ((separation - radius2)/radius1)-1.0
    if position2[1] > position1[1]:
        sgn = 1
    else:
        sgn = -1
    gamma = real((180./pi)*arcsin((position2[1]-position1[1])/xysep*sgn))
    if position2[2] > position1[2]:
        sgn = 1
    else:
        sgn = -1
    beta = real((180./pi)*arccos(xysep/separation))*sgn
    return center_of_mass, epsilon_r, beta, gamma
    

def dimer_bead_coords(beta, gamma, gap, x1, x2, k, xcom, ycom, zcom):
    """
    This function takes the position and orientation of the dimer
    defined by the center of mass, the two Euler angles (beta and
    gamma), the dimensionless gap distance parameter between the
    particles, the two size of parameters of each particle, and the
    wave vector, k. It then returns the x,y,z coordinates of each
    bead, their radius and the separation between them. 

    Parameters
    ----------

    beta : float [0,180]
        Euler angle beta. Specified in degrees.
    gamma : float [0,360]
        Euler angle gamma. Specified in degrees.
    gap : float
        Gap parameter which is dimensionless and specified the
        distance between the two individual particles making up
        the dimer.
    x1 : float
        Size parameter of the first particle of the dimer. Size parameter
        is equal to the radius times the wave vector, k.
    x2 : float
        Size parameter of the second particle of the dimer.
    k : float
        Wave vector. 2*pi / wavelength.
    xcom : float
        Center of mass x-coordinate of the dimer
    ycom : float
        Center of mass y-coordinate of the dimer
    zcom : float
        Center of mass z-coordinate of the dimer
    

    Returns
    -------

    position1 : numpy.ndarray of three elements
        x,y,z coordinate of first particle
    position2 : numpy.ndarray of three elements
        x,y,z coordinate of second particle
    radius1 : float
        Radius of first particle.
    radius2 : float
        Radius of second particle.
    center_to_center : float
        Center-to-center separation between the two particles.
        
    """
    radius1 = x1/k
    radius2 = x2/k
    center_to_center = (((1+gap)*x1)+x2)/k

    # Kluge around beta not being modulo anything.
    if beta < 0:
        beta += 180.
    elif beta > 180:
        beta -= 180.

    vect = array([center_to_center, 0, 0])

    trans_1 = array([[cos(gamma*pi/180.), sin(gamma*pi/180.), 0.],
                         [-1.*sin(gamma*pi/180.), cos(gamma*pi/180.), 0.],
                         [0., 0., 1.]])
    trans_2 = array([[cos(beta*pi/180.), 0., -1*sin(beta*pi/180.)],
                         [0., 1., 0.],
                         [sin(beta*pi/180.), 0., cos(beta*pi/180.)]])
    newvect = dot(trans_1, dot(trans_2, vect.transpose()))

    position1 = array([xcom, ycom, zcom])+(0.5*newvect*array([1,1,-1]))
    position2 = array([xcom, ycom, zcom])-(0.5*newvect*array([1,1,-1]))

    return real(position1), real(position2), radius1, radius2, center_to_center


'''
The following permits fit.py to run a fit from the command line, e.g.

python fit.py my_input_deck.yaml

which may be particularly useful for using the grid.
'''

if __name__ == '__main__':
    input_deck_yaml = sys.argv[1]
    fit(input_deck_yaml)
