# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca W. Perry,
# Jerome Fung, and Ryan McGorty
#
# This file is part of Holopy.
#
# Holopy is free software: you can1
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
from holopy.io import load
from holopy.io.yaml_io import load_yaml
from holopy.hologram import subimage
from holopy.process.enhance import normalize, background, divide
from holopy.process import centerfinder
import numpy as np
from holopy.optics import Optics


import minimizers.nmpfit_adapter as minimizer

def fit(input_deck):
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
    if position2[1]>position1[1]:
        sgn = 1
    else:
        sgn = -1
    gamma = real((180./pi)*arcsin((position2[1]-position1[1])/xysep*sgn))
    if position2[2]>position1[2]:
        sgn = 1
    else:
        sgn = -1
    beta = real((180./pi)*arccos(xysep/separation))*sgn
    return center_of_mass, epsilon_r, beta, gamma
    

def dimer_bead_coords(beta, gamma, gap, x1,x2,k, xcom, ycom, zcom):
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

    vect = array([center_to_center, 0,0])

    trans_1 = array([[cos(gamma*pi/180.), sin(gamma*pi/180.), 0.],
                         [-1.*sin(gamma*pi/180.), cos(gamma*pi/180.), 0.],
                         [0., 0., 1.]])
    trans_2 = array([[cos(beta*pi/180.), 0., -1*sin(beta*pi/180.)],
                         [0., 1., 0.],
                         [sin(beta*pi/180.), 0., cos(beta*pi/180.)]])
    newvect = dot(trans_1,dot(trans_2, vect.transpose()))

    position1 = array([xcom,ycom,zcom])+(0.5*newvect*array([1,1,-1]))
    position2 = array([xcom,ycom,zcom])-(0.5*newvect*array([1,1,-1]))

    return real(position1), real(position2), radius1, radius2, center_to_center


'''
The following permits fit.py to run a fit from the command line, e.g.

python fit.py my_input_deck.yaml

which may be particularly useful for using the grid.
'''

if __name__ == '__main__':
    input_deck_yaml = sys.argv[1]
    fit(input_deck_yaml)
