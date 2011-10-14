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

import sys
import os
hp_dir = (os.path.split(sys.path[0])[0]).rsplit(os.sep, 1)[0]
sys.path.append(hp_dir)
from scipy import sin, cos, array, pi, sqrt, arcsin, arccos, real, dot
from holopy.io import fit_io
from holopy.io.yaml_io import load_yaml
import numpy as np
from holopy.optics import Optics

#import minimizers.nmpfit_adapter as minimizer
from scatterpy.errors import (UnrealizableScatterer, ScattererOverlap,
                              InvalidScattererSphereOverlap, InvalidScatterer)

def cost_subtract(holo, calc):
    return holo - calc

def cost_rectified(holo, calc):
    return abs(holo-1) - abs(calc-1)

class FitResult(object):
    def __init__(self, scatterer, alpha, fnorm, status):
        self.scatterer = scatterer
        self.alpha = alpha
        self.fnorm = fnorm
        self.status = status
    def __getitem__(self, index):
        if index == 0:
            return self.scatterer
        if index == 1:
            return self.alpha
        raise KeyError
    def __repr__(self):
        return "{s.__class__.__name__}(scatterer={s.scatterer}, \
 alpha={s.alpha}, fnorm={s.fnorm}, status={s.status})".format(s=self)


def fit(holo, initial_guess, theory, minimizer='nmpfit', lower_bound=None,
        upper_bound=None, plot=False, minimizer_params={},
        residual_cost=cost_subtract, step = None):
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
    lower_bound, upper_bound : :class:`scatterpy.scatterer.Scatterer`, alpha
        The minimum and maximum values which the scatterer can vary
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

    scatterer, alpha = initial_guess

    scatterer.validate()
    
    def unpack_bound(b):
        return np.append(b[0].parameter_list, b[1])
    if lower_bound:
        lower_bound = unpack_bound(lower_bound)
    if upper_bound:
        upper_bound = unpack_bound(upper_bound)
    if step is not None:
        step = unpack_bound(step)

    names = scatterer.parameter_names_list + ['alpha']


    # check that the initial guess lies within the bounds
    guess_list = unpack_bound(initial_guess)
    if (guess_list > upper_bound).any() or (guess_list < lower_bound).any():
        names = np.array(names)
        raise GuessOutOfBounds(low=names[guess_list<lower_bound],
                               high=names[guess_list>upper_bound])
    
        
    if isinstance(theory, type):
        # allow the user to pass the type, we instantiate it here
        theory = theory(imshape = holo.shape, optics = holo.optics)

    # Rescale parameters so that the minimizer is working with all parameter
    # values ~ 1
    scale = scatterer.parameter_list
    scale = np.append(scale, alpha)
    # since we pick the intial guess as the scaling factor, our initial guess to
    # the minimizer will be all ones

    fixed = []
    for i in range(len(scale)):
        # if the lower_bound == scale == upper_bound the user wants this value
        # fixed, some fittters will not handle this case nicely, so we pull the
        # parameter from the list and add it back at the end.  
        if scale[i] == lower_bound[i] and scale[i] == upper_bound[i]:
            fixed.append(i)

    names = np.delete(names, fixed)

    lower_bound = np.delete(lower_bound/scale, fixed)
    upper_bound = np.delete(upper_bound/scale, fixed)
    if step is not None:
        step = np.delete(step/scale, fixed)
    
    guess = np.ones(len(lower_bound))

    residual = make_residual(holo, scatterer, theory, scale, fixed,
                             residual_cost)

    result, fnorm, status = minimize(residual, minimizer, guess, lower_bound, upper_bound,
                      parameter_names = names, step = step, **minimizer_params)
    
    # put back in the fixed values 
    for v in fixed:
        result = np.insert(result, v, 1.0)

    res = scale*result
        
    return FitResult(scatterer.make_from_parameter_list(res[:-1]), res[-1],
    fnorm, status)


def make_residual(holo, scatterer, theory, scale=1.0, fixed = [],
                  cost_func=cost_subtract):
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
    scale: :class:`numpy.ndarray`
        Factors to rescale each parameter before computing a hologram
    """

    def residual(p, **keywords):
        # put back in the fixed values 
        for v in fixed:
            p = np.insert(p, v, 1.0)
        p = p*scale

        for i, name in enumerate(scatterer.parameter_names_list+['alpha']):
            print('{0}: {1}'.format(name, p[i]))

        error = 1e12*np.ones(holo.size)

        # alpha should always be the last parameter, we prune it because the
        # scatterer doesn't want to know about it
        this_scatterer = scatterer.make_from_parameter_list(p[:-1])
            
        try:
            calculated = theory.calc_holo(this_scatterer, p[-1])
        except (UnrealizableScatterer, InvalidScatterer) as e:
            if isinstance(e, InvalidScattererSphereOverlap):
                print("Hologram computation attempted with overlapping \
spheres, returning large residual")
            else:
                print("Fitter asked for a value which the scattering theory \
thought was unphysical or uncomputable, returning large residual")
            return error

        return cost_func(holo, calculated).ravel()

    return residual

def minimize(residual, algorithm='nmpfit', guess=None, lb=None , ub=None,
             quiet = False, parameter_names = None, plot = False, ftol = 1e-10,
             xtol = 1e-10, gtol = 1e-10, damp = 0, maxiter = 100, err=None, step
             = None):
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

    
    if algorithm == 'nmpfit':
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
        
        return fitresult.params, fitresult.fnorm, fitresult.status

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
        return r.xf

    else:
        raise MinimizerNotFound(algorithm)

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


# Legacy code, figure out what of this should stay
def fit_deck(input_deck):
    '''
    Run a fit described by yaml file input_deck.

    Parameters
    ----------
    input_deck : string (filename)
       yaml file describing the fit you want to run

    Returns
    -------
    Nothing, but outputs two files to directories specified in the input deck

    fit_results.tsv : tsv file
       Fitted physical parameters for each image in the fit series
    fits/imagexxxx_fit.yaml : yaml file
       All parameters of the fit

    Notes
    -----
    Tentative output is a dictionary of dictionaries.
    Output dict has following keys: {'model', 'parameters', 'nmpfit_data',
    'optics', 'io'} whose values could also be dictionaries. This
    seems a little easier (and also more robust) than defining a new
    output_object class and writing PyYAML dumpers, etc to write it
    out. 
    '''
    # read input deck
    deck = fit_io.load_FitInputDeck(input_deck)

    fit_io._setup_output_directory(deck)
 
    # a dict of nmpfit_param objects (all possible)
    parameters = deck._get_fit_parameters()

    outf = fit_io.SimpleFitOutFile(deck)
    
    # the main loop over holograms to be fit 
    for num in range(deck['image_range'][0], 
                     deck['image_range'][1]+1):

        ######################################################################
        # Minimization
        ######################################################################
        
        holo = get_target(deck, num)

        parlist = [parameters[k] for k in deck._get_full_par_ordering()]
        fitted_pars = [par for par in parlist if not (par.fixed or 
                                                      hasattr(par, 'tied'))]
        fixed_pars = [par for par in parlist if par.fixed]
        tied_pars = [par for par in parlist if hasattr(par, 'tied')]

        def forward_holo(values):
            scat_dict = {}
            i = 0
            for par in fitted_pars:
                scat_dict[par.name] = values[i]
                i += 1
            for par in fixed_pars:
                scat_dict[par.name] = par.value
            for par in tied_pars:
                scat_dict[par.name] = scat_dict[par.tied]
            return deck.model._forward_holo(holo.shape, holo.optics, scat_dict)

        fit_result = deck.minimizer._minimize(holo, forward_holo, fitted_pars,
                                              **deck._get_extra_minimizer_params())
        
        # include the list of what was done with each parameter in the fit_result
        fit_result.holo_shape = holo.shape
        fit_result.parlist = parlist
        fit_result.fitted_pars = fitted_pars
        fit_result.fixed_pars = fixed_pars
        fit_result.tied_pars = tied_pars
        
        
        ######################################################################
        # Output
        ######################################################################

        out_param_dict = {}
        for par in parlist:
            out_param_dict[par.name] = par.output_dict()
        for par in tied_pars:
            out_param_dict[par.name] = out_param_dict[par.tied]

        fit_io._output_frame_yaml(deck, fit_result, num)

        outf.write_data_line(out_param_dict, num,
                             fit_result.fit_error,
                             fit_result.fit_status)

        # Update parameters to set initial values for next frame
        parameters = update_params(parameters)

        # Reset parameters if needed
        if deck.get('reset_to_initial'):
            reset_to_initial = deck.get('reset_to_initial')
            for parname in reset_to_initial:
                scaling = deck._param_rescaling_factor(parname)
                parameters[parname].value = deck[parname] * scaling
 
    # Cleanup: close tsv file
    outf.close()

def get_target(deck, number=None):
    """
    Returns the preprocessed hologram that the fitter will try to fit. 

    Use this interactively to view the preprocessed hologram that the
    fitter will compare against.  This is intended for comparison
    against the results of fit.get_initial_guess() to evaluate
    parameter guesses.

    It is also used internally to the fitting routine to generate the
    hologram to fit.

    Parameters
    ----------
    deck : dict
       Input parameters from input deck yaml file
    number : int
       Number of the image to use for this frame.  Defaults to first
       hologram in the fit series.

    Returns
    -------
    target : :class:`holopy.hologram.Hologram`
       Hologram properly preprocessed for fitting

    Notes
    -----
    Background is cached by FitInputDeck, so calling this repeatedly
    does not incur overhead of loading the background each time.

    Examples
    --------
    >>> holopy.fit.get_target('input_deck.yaml', image_number)
    Hologram([[ 1.00600797,  1.00298787,  0.99809726, ...,  0.99809726,
         1.00298787,  1.00600797],
       [ 1.00298039,  0.99806496,  0.9939991 , ...,  0.9939991 ,
         0.99806496,  1.00298039],
       [ 0.99813346,  0.99403441,  0.99316255, ...,  0.99316255,
         0.99403441,  0.99813346],
       ..., 
       [ 0.99813346,  0.99403441,  0.99316255, ...,  0.99316255,
         0.99403441,  0.99813346],
       [ 1.00298039,  0.99806496,  0.9939991 , ...,  0.9939991 ,
         0.99806496,  1.00298039],
       [ 1.00600797,  1.00298787,  0.99809726, ...,  0.99809726,
         1.00298787,  1.00600797]])
    """
    deck = fit_io.load_FitInputDeck(deck)

    return deck._get_image_by_number(number)

def get_initial_guess(deck):
    """
    Calculate the hologram that input_deck specifies as its initial
    guess.

    Intended for interactive use to check how a guess you have specified
    compares to the hologram you are fitting.  

    Parameters
    ----------
    input_deck : string (filename)
       yaml fit input file

    Returns
    -------
    guess_holo : :class:`holopy.hologram.Hologram`
       The computed hologram that input_deck uses as its initial guess

    Examples
    --------
    >>> holopy.fit.get_initial_guess('input_deck.yaml')
    Hologram([[ 1.00600797,  1.00298787,  0.99809726, ...,  0.99809726,
         1.00298787,  1.00600797],
       [ 1.00298039,  0.99806496,  0.9939991 , ...,  0.9939991 ,
         0.99806496,  1.00298039],
       [ 0.99813346,  0.99403441,  0.99316255, ...,  0.99316255,
         0.99403441,  0.99813346],
       ..., 
       [ 0.99813346,  0.99403441,  0.99316255, ...,  0.99316255,
         0.99403441,  0.99813346],
       [ 1.00298039,  0.99806496,  0.9939991 , ...,  0.9939991 ,
         0.99806496,  1.00298039],
       [ 1.00600797,  1.00298787,  0.99809726, ...,  0.99809726,
         1.00298787,  1.00600797]])

    """
    deck = fit_io.load_FitInputDeck(deck)

    parameters = deck._get_fit_parameters()
    h = get_target(deck)
    shape = h.shape

    guess = dict([(p, parameters[p].value) for p in
                  deck._get_full_par_ordering()])

    return deck.model._forward_holo(shape, h.optics, guess)


def get_fit_result(fit_yaml): 
    '''
    Returns the hologram corresponding to the best fit from a fitting
    run. 

    Parameters
    ----------
    fit_yaml : string
        filename and path of yaml file describing the fit results, as
        produced by fit.fit() 

    Returns
    -------
    holo : :class:`holopy.hologram.Hologram` object
        Hologram corresponding to best fit
    '''
    fit = load_yaml(fit_yaml)
    opt = Optics(**fit['optics'])

    model = fit_io._choose_model(fit['model'])
    
    scat_dict = {}
    for param_name, param_out_dict in fit['parameters'].iteritems():
        scat_dict[param_name] = param_out_dict['final_value']

    return model._forward_holo(fit['io']['hologram_shape'], opt,
                                     scat_dict)

    

def _minimize(holo, parlist, model, err=None, extra_fitter_params={}):
    
    fitted_pars = [par for par in parlist if not (par.fixed or 
                                                  hasattr(par, 'tied'))]
    fixed_pars = [par for par in parlist if par.fixed]
    tied_pars = [par for par in parlist if hasattr(par, 'tied')]

    def forward_holo(values):
        scat_dict = {}
        i = 0
        for par in fitted_pars:
            scat_dict[par.name] = values[i]
            i += 1
        for par in fixed_pars:
            scat_dict[par.name] = par.value
        for par in tied_pars:
            scat_dict[par.name] = scat_dict[par.tied]
        return model._forward_holo(holo.shape, holo.optics, scat_dict)

    fitresult = minimizer._minimize(holo, forward_holo, fitted_pars,
                                    **extra_fitter_params)

    # include the list of what was done with each parameter in the fitresult
    fitresult.holo_shape = holo.shape
    fitresult.parlist = parlist
    fitresult.fitted_pars = fitted_pars
    fitresult.fixed_pars = fixed_pars
    fitresult.tied_pars = tied_pars

    return fitresult

def update_params(parameters):
    '''
    Create updated parameter dictionary for the next fit in a series.
    '''
    for name, par in parameters.iteritems():
        if not (par.fixed or hasattr(par,'tied')): # fitted parameter
            parameters[name].value = par.fit_value
        if hasattr(par,'tied'):
            # check if any parameters are tied, and update those too so that 
            # output of all parameters is done correctly
            parameters[name].value = parameters[par.tied].value

    return parameters


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
