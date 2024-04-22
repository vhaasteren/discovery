import functools

from . import matrix

# these versions of ORFs take only one parameter (the angle)
# z = matrix.jnp.dot(pos1, pos2)

def hd_orfa(z):
    omc2 = (1.0 - z) / 2.0
    return 1.5 * omc2 * matrix.jnp.log(omc2) - 0.25 * omc2 + 0.5 + 0.5 * matrix.jnp.allclose(z, 1.0)

def dipole_orfa(z):
    return z + 1.0e-6 * matrix.jnp.allclose(z, 1.0)

def monopole_orfa(z):
    return 1.0 + 1.0e-6 * matrix.jnp.allclose(z, 1.0)


class OS:
    def __init__(self, gbl):
        if gbl.globalgp is None:
            raise ValueError('GlobalLikelihood passed to OS needs a globalgp.')

        self.psls = gbl.psls
        self.globalgp = gbl.globalgp

        self.pos = [matrix.jnparray(p) for p in self.globalgp.pos]
        self.pairs = [(i1, i2) for i1 in range(len(self.pos)) for i2 in range(i1 + 1, len(self.pos))]
        self.angles = [matrix.jnp.dot(self.pos[i], self.pos[j]) for (i,j) in self.pairs]

    @functools.cached_property
    def os_rhosigma(self):
        kernelsolves = [psl.N.make_kernelsolve(psl.y, Tmat) for (psl, Tmat) in zip(self.psls, self.globalgp.Fs)]

        getN = self.globalgp.Phi.getN
        components = self.globalgp.Fs[0].shape[1]    # needed because of makegp_fourier_allpsr
        pairs = self.pairs                           # better not close on self

        def get_rhosigma(params):
            # TO DO: make this work with makegp_fourier_global, or even better with a single psr getN
            sN = matrix.jnp.sqrt(getN(params)[:components])
            ks = [k(params) for k in kernelsolves]

            ts = [matrix.jnp.dot(sN * ks[i][0], sN * ks[j][0]) for (i,j) in pairs]

            ds = [sN[:,matrix.jnp.newaxis] * k[1] * sN[matrix.jnp.newaxis,:] for k in ks]
            bs = [matrix.jnp.trace(ds[i] @ ds[j]) for (i,j) in pairs]

            return (matrix.jnp.array(ts) / matrix.jnp.array(bs),
                    1.0 / matrix.jnp.sqrt(matrix.jnp.array(bs)))

        get_rhosigma.params = sorted(set.union(*[set(k.params) for k in kernelsolves], getN.params))

        return get_rhosigma

    @functools.cached_property
    def os(self):
        # find the parameter that sets the GW amplitude
        gwpar = [par for par in self.globalgp.Phi.params if 'log10_A' in par][0]

        os_rhosigma = self.os_rhosigma    # getos will close on os_rhosigma
        angles = matrix.jnparray(self.angles)

        def get_os(params, orf=hd_orfa):
            rhos, sigmas = os_rhosigma(params)

            gwnorm = 10**(2.0 * params[gwpar])
            rhos, sigmas = gwnorm * rhos, gwnorm * sigmas

            orfs = orf(angles)

            os = matrix.jnp.sum(rhos * orfs / sigmas**2) / matrix.jnp.sum(orfs**2 / sigmas**2)
            os_sigma = 1.0 / matrix.jnp.sqrt(matrix.jnp.sum(orfs**2 / sigmas**2))
            snr = os / os_sigma

            return {'os': os, 'os_sigma': os_sigma, 'snr': snr, 'log10_A': params[gwpar]} # , 'rhos': rhos, 'sigmas': sigmas}

        get_os.params = os_rhosigma.params

        return get_os

    @functools.cached_property
    def scramble(self):
        gwpar = [par for par in self.globalgp.Phi.params if 'log10_A' in par][0]

        os_rhosigma = self.os_rhosigma    # getos will close on os_rhosigma
        pairs = self.pairs

        def get_scramble(params, pos, orf=hd_orfa):
            rhos, sigmas = os_rhosigma(params)

            gwnorm = 10**(2.0 * params[gwpar])
            rhos, sigmas = gwnorm * rhos, gwnorm * sigmas

            angles = matrix.jnparray([matrix.jnp.dot(pos[i], pos[j]) for (i,j) in pairs])
            orfs = orf(angles)

            os = matrix.jnp.sum(rhos * orfs / sigmas**2) / matrix.jnp.sum(orfs**2 / sigmas**2)
            os_sigma = 1.0 / matrix.jnp.sqrt(matrix.jnp.sum(orfs**2 / sigmas**2))
            snr = os / os_sigma

            return {'os': os, 'os_sigma': os_sigma, 'snr': snr, 'log10_A': params[gwpar]} #, 'rhos': rhos, 'sigmas': sigmas}

        get_scramble.params = os_rhosigma.params

        return get_scramble
