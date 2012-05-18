import scatterpy
    
def fit(model, data, algorithm='nmpfit'):
    result = algorithm.minimize(model.parameters, model.cost_func(data))
    return result

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
        self._make_scatterer=make_scatterer

    def make_scatterer(self, par_values):
        if self._make_scatterer:
            return self._make_scatterer(par_values)
        else:
            physical = {}
            for i, parameter in enumerate(self.parameters):
                physical[parameter.name] = parameter.unscale(par_values[i])
            return self.scatterer.from_parameters(physical)

    # TODO: add a make_optics function so that you can have parameters
    # affect optics things (fit to beam divergence, lens abberations, ...)

    def alpha(self, par_values):
        return self.parameters[-1].unscale(par_values[-1])
    
    def cost_func(self, data): 
        if not isinstance(self.theory, scatterpy.theory.ScatteringTheory):
            theory = self.theory(data.optics, data.shape)
        else:
            theory = self.theory
            
        def cost(par_values):
            calc = theory.calc_holo(self.make_scatterer(par_values[:-1]), self.alpha(par_values))
            return compare(calc, data)
        return cost

    # TODO: make a user overridabel cost function that gets physical parameters
    # so that the unscaling happens only in one place (and as close to the
    # minimizer as possible).  

    # TODO: Allow a layer on top of theory to do things like moving sphere
        
class Minimizer(object):
    def __init__(self):
        pass

    def minimize(self, parameters, cost_func):
        pass
        

class Parameter(object):
    def __init__(self, guess = None, limits = None, name = None, misc = None):
        self.name = name
        self.guess = guess
        self.limits = limits
        self.misc = misc
        if guess is not None:
            self.scale = guess
        else:
            
        self.scale = #compute this

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

        return physical * self.scale

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
        return scaled / self.scale


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
