import inspect

import numpy as np

import scatterpy
from holopy.utility.helpers import _ensure_pair
from holopy.io.yaml_io import Serializable


class FitResult(Serializable):
    def __init__(self, scatterer, alpha):
        self.scatterer = scatterer
        self.alpha = alpha

def fit(model, data, algorithm='nmpfit'):
    minimizer = Minimizer(algorithm)
    result = minimizer.minimize(model.parameters, model.cost_func(data))
    return model.make_scatterer_from_par_values(result)

class Model(object):
    """
    Representation of a model to fit to data

    Parameters
    ----------
    parameters: list(:class:`Parameter`)
        The parameters which can be varied in this model
    theory: :class:`scatterpy.theory.ScatteringTheory`
        The theory that should be used to compute holograms
    scatterer: :class:`scatterpy.scatterer.AbstractScatterer`
        Scatterer to compute holograms of, ignored if make_scatterer is given
    make_scatterer: function(par_values) -> :class:`scatterpy.scatterer.AbstractScatterer`
        Function that returns a scatterer given parameters

    Notes
    -----
    Any arbitrary parameterization can be used by simply specifying a
    make_scatterer function which can turn the parameters into a scatterer
    
    """
    def __init__(self, parameters, theory, scatterer=None, make_scatterer=None):
        self.parameters = parameters
        self.theory = theory
        self.scatterer=scatterer
        self.make_scatterer = make_scatterer
        self.compare = lambda calc, data: (calc-data).ravel()

    def make_scatterer_from_par_values(self, par_values):
        all_pars = {}
        for i, p in enumerate(self.parameters):
            all_pars[p.name] = p.unscale(par_values[i])
        for_scatterer = {}
        for arg in inspect.getargspec(self.make_scatterer).args:
            for_scatterer[arg] = all_pars[arg] 
        return self.make_scatterer(**for_scatterer)
        
    # TODO: add a make_optics function so that you can have parameters
    # affect optics things (fit to beam divergence, lens abberations, ...)

    def alpha(self, par_values):
        for i, par in enumerate(self.parameters):
            if par.name == 'alpha':
                return par.unscale(par_values[i])
        return None
    
    def cost_func(self, data): 
        if not isinstance(self.theory, scatterpy.theory.ScatteringTheory):
            theory = self.theory(data.optics, data.shape)
        else:
            theory = self.theory
            
        def cost(par_values):
            calc = theory.calc_holo(self.make_scatterer_from_par_values(par_values),
                             self.alpha(par_values))
            return self.compare(calc, data)
        return cost

    # TODO: make a user overridabel cost function that gets physical parameters
    # so that the unscaling happens only in one place (and as close to the
    # minimizer as possible).  

    # TODO: Allow a layer on top of theory to do things like moving sphere
        
class Minimizer(object):
    def __init__(self, algorithm='nmpfit'):
        self.algorithm = algorithm

    def minimize(self, parameters, cost_func):
        if self.algorithm == 'nmpfit':
            from holopy.third_party import nmpfit
            nmp_pars = []
            for i, par in enumerate(parameters):

                def resid_wrapper(p, fjac=None):
                    status = 0
                    return [status, cost_func(p)]
    
                d = {'parname': par.name}
                if par.limit is not None:
                    d['limited'] = [par.scale(l) is not None for l in par.limit]
                    d['limits'] = par.scale(np.array(par.limit))
                else:
                    d['limited'] = [False, False]    
                if par.guess is not None:
                    d['value'] = par.scale(par.guess)
                else:
                    raise NeedInitialGuess()
                nmp_pars.append(d)
            fitresult = nmpfit.mpfit(resid_wrapper, parinfo=nmp_pars)
            return fitresult.params

        

class Parameter(object):
    def __init__(self, name = None, guess = None, limit = None, misc = None):
        self.name = name
        self.guess = guess
        self.limit = limit
        self.misc = misc
        if guess is not None:
            self.scale_factor = guess
        else:
            self.scale_factor = np.sqrt(limits[0]*limits[1])

    def scale(self, physical):
        """
        Scales parameters to approximately unity

        Parameters
        ----------
        physical: np.array(dtype=float)

        Returns
        -------
        scaled: np.array(dtype=float)
        """

        return physical / self.scale_factor

    def unscale(self, scaled):
        """
        Inverts scale's transformation

        Parameters
        ----------
        scaled: np.array(dtype=float)

        Returns
        -------
        physical: np.array(dtype=float)
        """
        return scaled * self.scale_factor


class RigidSphereCluster(Model):
    def __init__(self, reference_scatterer, alpha, beta, gamma, x, y, z):
        self.parameters = [alpha, beta, gamma, x, y, z]
        self.theory = scatterpy.theory.Multisphere
        self.reference_scatterer = reference_scatterer

    def make_scatterer(self, par_values):
        unscaled = []
        for i, val in enumerate(par_values):
            unscaled.append(self.parameters[i].unscale(val))
        return self.reference_scatterer.rotated(unscaled[:3]).translated(unscaled[3:6])


# Archiving:
# Model (parameters, theory, cost function, 

################################################################
# Fitseries engine

# default
# Load, normalize, background
# fit
# archive

# provide customization hooks
# prefit - user supplied
# fit
# postfit - user supplied
# archive

# archive to a unique directory name (probably from time and hostname)
