import re

import numpy as np

from . import matrix

def uniform(par, a, b):
    def logpriorfunc(params):
        return matrix.jnp.where(matrix.jnp.logical_and(params[par] >= a, params[par] <= b),
                                0, -matrix.jnp.inf)

    return logpriorfunc

priordict_standard = {
    '.*_rednoise_log10_A': [-18, -11],
    '.*_rednoise_gamma': [0, 7],
    '.*_red_noise_log10_A': [-18, -11],  # deprecated
    '.*_red_noise_gamma': [0, 7],        # deprecated
    'crn_log10_A': [-18, -11],
    'crn_gamma': [0, 7],
    'gw_(.*_)?log10_A': [-18, -11],
    'gw_(.*_)?gamma': [0, 7],
    'dmgp_log10_A': [-20, -11],
    'dmgp_gamma': [0, 7],
    'crn_log10_rho': [-9, -4],
    'gw_(.*_)?log10_rho': [-9, -4]
}

def makelogprior_uniform(params, priordict={}):
    priordict = {**priordict_standard, **priordict}

    priors = []
    for par in params:
        for parname, range in priordict.items():
            if parname in par:
                priors.append(uniform(par, *range))
                break

    def logprior(params):
        return sum(prior(params) for prior in priors)

    return logprior

def sample_uniform(params, priordict={}, n=1):
    priordict = {**priordict_standard, **priordict}

    sample = {}
    for par in params:
        for parname, range in priordict.items():
            if re.match(parname, par):
                if par.endswith(')'):
                    sample[par] = (np.random.uniform(*range, size=int(par[par.index('(')+1:-1])) if n == 1
                                   else np.random.uniform(*range, size=(n,int(par[par.index('(')+1:-1]))))
                else:
                    sample[par] = (np.random.uniform(*range) if n == 1
                                   else np.random.uniform(*range, size=n))
                break
        else:
            raise KeyError(f"No known prior for {par}.")

    return sample
