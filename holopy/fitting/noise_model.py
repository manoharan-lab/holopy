from model import BaseModel
from parameter import Parameter
from holopy.core.holopy_object import HoloPyObject
from holopy.core import Marray
from holopy.core.helpers import dict_without
from holopy.scattering.theory.scatteringtheory import scattered_field_to_hologram
from holopy.scattering.errors import MultisphereExpansionNaN, MultisphereFieldNaN, ConvergenceFailureMultisphere, ScattererDefinitionError

import numpy as np
import pandas as pd
from copy import copy

class NoiseModel(BaseModel):
    def __init__(self, scatterer, theory, noise_sd):
        super(NoiseModel, self).__init__(scatterer)
        self.theory = theory
        self._use_parameter(noise_sd, 'noise_sd')

    def _pack(self, vals):
        return {par.name: val for par, val in zip(self.parameters, vals)}

    def lnprior(self, par_vals):
        if isinstance(par_vals, dict):
            return sum([p.lnprob(par_vals[p.name]) for p in self.parameters])
        else:
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

    def _fields(self, pars, schema):
        scatterer = self.scatterer.make_from(pars)
        try:
            return self.theory.calc_field(scatterer, schema)
        except (MultisphereExpansionNaN, MultisphereFieldNaN, ConvergenceFailureMultisphere, ScattererDefinitionError):
            return -np.inf

    def _lnlike(self, pars, data):
        noise_sd = pars.pop('noise_sd', self.noise_sd)
        holo = self._holo(pars, data)
        N = data.size
        return (-N*np.log(noise_sd*np.sqrt(2*np.pi)) -
                ((holo-data)**2).sum()/(2*noise_sd**2))

    def lnlike(self, par_vals, data):
        return self._lnlike(self._pack(par_vals), data)




class AlphaModel(NoiseModel):
    def __init__(self, scatterer, theory, noise_sd, alpha):
        super(AlphaModel, self).__init__(scatterer, theory, noise_sd)
        self._use_parameter(alpha, 'alpha')

    def _holo(self, pars, schema, alpha=None):
        if alpha is None:
            alpha = self.alpha
        alpha = pars.pop('alpha', alpha)
        fields = self._fields(pars, schema)
        return scattered_field_to_hologram(alpha*fields, schema.optics)


class SpeckleModel(NoiseModel):
    names = 'a', 'beta', 'xi', 'eta', 'd'
    def __init__(self, scatterer, theory, noise_sd, order, a, beta, xi, eta, d):
        super(SpeckleModel, self).__init__(scatterer, theory, noise_sd)
        self.order = order
        # TODO: switch to storing these as an array rather than going back and forth through dictionaries
        for i in range(order):
            for name, par in zip(self.names, (a, beta, xi, eta, d)):
                self._use_parameter(copy(par), "{}_{}".format(name, i))


    def lnlike(self, par_vals, data):
        N = data.size
        raw_pars = self._pack(par_vals)
        # TODO: switch to storing these as an array rather than going back and forth through dictionaries
        def get_pars(i):
            names = [(name, "{}_{}".format(name, i)) for name in self.names]
            return {col: raw_pars.pop(name, getattr(self, name)) for col, name in names}
        pars = pd.DataFrame([get_pars(i) for i in range(self.order)])
        # TODO: this will only work for subsetted data (need to reshape for other data I think)
        a = pars.a.reshape(1, -1)
        beta = pars.beta.reshape(1, -1)
        xi = pars.xi.reshape(1, -1)
        eta = pars.eta.reshape(1, -1)
        d = pars.d.reshape(1, -1)

        noise_sd = raw_pars.pop('noise_sd', self.noise_sd)
        x, y, z = data.positions.xyz().T
        x = x[:,np.newaxis]
        y = y[:,np.newaxis]
        z = z[:,np.newaxis]

        # For now, assume a single polarization along x or y. This is generally what we do, and will save computation time.
        pol = np.nonzero(data.optics.polarization)[0][0]
        scatterer = self.scatterer.make_from(raw_pars)
        fields = self.theory.calc_field(scatterer, data)
        A = (a * np.exp(-2*np.pi * 1j / (data.optics.med_wavelen * (d-z)) * (x*xi + y*eta))).sum(-1)
        holo = Marray(np.abs(1+fields[...,pol] + A)**2, **dict_without(fields._dict, ['dtype', 'components']))

        return -N*np.log(noise_sd*np.sqrt(2*np.pi)) - ((holo-data)**2).sum()/(2*noise_sd**2)
