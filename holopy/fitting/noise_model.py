from model import BaseModel
from parameter import Parameter
from holopy.core.holopy_object import HoloPyObject
from holopy.scattering.theory.scatteringtheory import scattered_field_to_hologram

import numpy as np

class NoiseModel(BaseModel):
    def __init__(self, scatterer, theory, noise_sd):
        super(NoiseModel, self).__init__(scatterer)
        self.theory = theory
        self._use_parameter(noise_sd, 'noise_sd')

    def _use_parameter(self, par, name):
        setattr(self, name, par)
        if isinstance(par, Parameter):
            if par.name is None:
                par.name = name
            self.parameters.append(par)

    def _pack(self, vals):
        return {par.name: val for par, val in zip(self.parameters, vals)}

    def lnprior(self, par_vals):
        return sum([p.lnprob(v) for p, v in zip(self.parameters, par_vals)])

    def lnposterior(self, par_vals, data):
        lnprior = self.lnprior(par_vals)
        # prior is sometimes used to forbid thing like negative radius
        # which will fail if you attempt to compute a hologram of, so
        # don't try to compute likelihood where the prior already
        # forbids you to be
        if lnprior == -np.inf:
            return lnprior
        else:
            return lnprior + self.lnlike(par_vals, data)


class AlphaModel(NoiseModel):
    def __init__(self, scatterer, theory, noise_sd, alpha):
        super(AlphaModel, self).__init__(scatterer, theory, noise_sd)
        self._use_parameter(alpha, 'alpha')

    def lnlike(self, par_vals, data):
        N = data.size
        pars = self._pack(par_vals)
        alpha = pars.pop('alpha', self.alpha)
        noise_sd = pars.pop('noise_sd', self.noise_sd)

        scatterer = self.scatterer.make_from(pars)
        fields = self.theory.calc_field(scatterer, data)
        holo = scattered_field_to_hologram(alpha*fields, data.optics)

        return -N*np.log(noise_sd*np.sqrt(2*np.pi)) - ((holo-data)**2).sum()/(2*noise_sd**2)

class SpeckleModel(NoiseModel):
    def __init__(self, scatterer_priors, theory, noise_sd, a_priors, beta_priors):
        super(SpeckleModel, self).__init__(scatterer_priors, theory, noise_sd)
        if len(a_priors) != len(beta_priors):
            raise TheoryDefinitionError("you must provide the same number of priors for a and beta")
