import inspect
import typing
from collections.abc import Iterable

import numpy as np
import jax.numpy as jnp

from . import matrix
from . import const

# residuals

def residuals(psr):
    return psr.residuals

# EFAC/EQUAD/ECORR noise

# no backends
def makenoise_measurement_simple(psr, noisedict={}):
    efac = f'{psr.name}_efac'
    log10_t2equad = f'{psr.name}_log10_t2equad'
    params = [efac, log10_t2equad]

    if all(par in noisedict for par in params):
        noise = noisedict[efac]**2 * (psr.toaerrs**2 + 10.0**(2.0 * noisedict[log10_t2equad]))

        return matrix.NoiseMatrix1D_novar(noise)
    else:
        toaerrs = matrix.jnparray(psr.toaerrs)
        def getnoise(params):
            return params[efac]**2 * (toaerrs**2 + 10.0**(2.0 * params[log10_t2equad]))
        getnoise.params = params

        return matrix.NoiseMatrix1D_var(getnoise)

# nanograv backends
def makenoise_measurement(psr, noisedict={}):
    backends = sorted(set(psr.backend_flags))

    efacs = [f'{psr.name}_{backend}_efac' for backend in backends]
    log10_t2equads = [f'{psr.name}_{backend}_log10_t2equad' for backend in backends]
    params = efacs + log10_t2equads

    masks = [(psr.backend_flags == backend) for backend in backends]

    if all(par in noisedict for par in params):
        noise = sum(mask * noisedict[efac]**2 * (psr.toaerrs**2 + 10.0**(2 * noisedict[log10_t2equad]))
                    for mask, efac, log10_t2equad in zip(masks, efacs, log10_t2equads))

        return matrix.NoiseMatrix1D_novar(noise)
    else:
        toaerrs, masks = matrix.jnparray(psr.toaerrs), [matrix.jnparray(mask) for mask in masks]
        def getnoise(params):
            return sum(mask * params[efac]**2 * (toaerrs**2 + 10.0**(2 * params[log10_t2equad]))
                    for mask, efac, log10_t2equad in zip(masks, efacs, log10_t2equads))
        getnoise.params = params

        return matrix.NoiseMatrix1D_var(getnoise)


# ECORR quantization
#
# note the resulting ecorr degrees of freedom are slightly different than in enterprise
# (and of course I forgot about it)

# bins = (psr.toas + 0.5).astype(np.int64)
# uniques, counts = np.unique(bins, return_counts=True)
# Umat = jnp.array(np.vstack([bins == unique for unique, count in zip(uniques, counts) if count > 1]).astype(jnp.float64).T)

def quantize(toas):
    isort = np.argsort(toas)
    bins = np.zeros_like(toas, np.int64)

    b, v = 0, toas.min()
    for j in isort:
        if toas[j] - v > 1.0:
            v = toas[j]
            b = b + 1

        bins[j] = b

    return bins

# no backends
def makegp_ecorr_simple(psr, noisedict={}):
    log10_ecorr = f'{psr.name}_log10_ecorr'
    params = [log10_ecorr]

    bins = quantize(psr.toas)
    Umat = np.vstack([bins == i for i in range(bins.max() + 1)]).T
    ones = np.ones(Umat.shape[1], dtype=np.float64)

    if all(par in noisedict for par in params):
        phi = (10.0**(2.0 * noisedict[log10_ecorr])) * ones

        return matrix.ConstantGP(matrix.NoiseMatrix1D_novar(phi), Umat)
    else:
        ones = matrix.jnparray(ones)
        def getphi(params):
            return (10.0**(2.0 * params[log10_ecorr])) * ones
        getphi.params = Params

        return matrix.VariableGP(matrix.NoiseMatrix1D_var(getphi), Umat)

# nanograv backends
def makegp_ecorr(psr, noisedict={}):
    log10_ecorrs, Umats = [], []

    backends = sorted(set(psr.backend_flags))
    masks = [np.array(psr.backend_flags == backend) for backend in backends]
    for backend, mask in zip(backends, masks):
        log10_ecorrs.append(f'{psr.name}_{backend}_log10_ecorr')

        bins = quantize(psr.toas * mask)
        Umats.append(np.vstack([bins == i for i in range(1, bins.max() + 1)]).T)
    Umatall = np.hstack(Umats)
    params = log10_ecorrs

    pmasks, cnt = [], 0
    for Umat in Umats:
        z = np.zeros(Umatall.shape[1], dtype=np.float64)
        z[cnt:cnt+Umat.shape[1]] = 1.0
        pmasks.append(z)
        cnt = cnt + Umat.shape[1]

    if all(par in noisedict for par in params):
        phi = sum(10.0**(2 * noisedict[log10_ecorr]) * pmask for (log10_ecorr, pmask) in zip(log10_ecorrs, pmasks))

        return matrix.ConstantGP(matrix.NoiseMatrix1D_novar(phi), Umatall)
    else:
        pmasks = [matrix.jnparray(pmask) for pmask in pmasks]
        def getphi(params):
            return sum(10.0**(2 * params[log10_ecorr]) * pmask for (log10_ecorr, pmask) in zip(log10_ecorrs, pmasks))
        getphi.params = params

        return matrix.VariableGP(matrix.NoiseMatrix1D_var(getphi), Umatall)

# timing model

def makegp_improper(psr, fmat, constant=1.0e40, name='improperGP'):
    return matrix.ConstantGP(matrix.NoiseMatrix1D_novar(constant * np.ones(fmat.shape[1])), fmat)

def makegp_timing(psr, constant=1.0e40):
    fmat = np.array(psr.Mmat / np.sqrt(np.sum(psr.Mmat**2, axis=0)), dtype=np.float64)

    return makegp_improper(psr, fmat, constant=constant, name='timingmodel')

# Fourier GP

def getspan(psrs):
    if isinstance(psrs, Iterable):
        return max(psr.toas.max() for psr in psrs) - min(psr.toas.min() for psr in psrs)
    else:
        return psrs.toas.max() - psrs.toas.min()

def fourierbasis(psr, components, T=None):
    if T is None:
        T = getspan(psr)

    f  = np.arange(1, components + 1, dtype=np.float64) / T
    df = np.diff(np.concatenate((np.array([0]), f)))

    fmat = np.zeros((psr.toas.shape[0], 2*components), dtype=np.float64)
    for i in range(components):
        fmat[:, 2*i  ] = np.sin(2.0 * jnp.pi * f[i] * psr.toas)
        fmat[:, 2*i+1] = np.cos(2.0 * jnp.pi * f[i] * psr.toas)

    return np.repeat(f, 2), np.repeat(df, 2), fmat

def dmfourierbasis(psr, components, T=None, fref=1400.0):
    f, df, fmat = fourierbasis(psr, components, T)

    Dm = (fref / psr.freqs)**2

    return f, df, fmat * Dm[:, None]

def makegp_fourier(psr, prior, components, T=None, fourierbasis=fourierbasis, common=[], name='fourierGP'):
    argspec = inspect.getfullargspec(prior)
    argmap = [(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
              (f'({components})' if argspec.annotations.get(arg) == typing.Sequence else '')
              for arg in argspec.args if arg not in ['f', 'df']]

    f, df, fmat = fourierbasis(psr, components, T)

    f, df = matrix.jnparray(f), matrix.jnparray(df)
    def priorfunc(params):
        return prior(f, df, *[params[arg] for arg in argmap])
    priorfunc.params = argmap

    return matrix.VariableGP(matrix.NoiseMatrix1D_var(priorfunc), fmat)

# Global Fourier GP

def makegp_fourier_global(psrs, prior, orf, components, T, fourierbasis=fourierbasis, name='globalFourierGP'):
    argspec = inspect.getfullargspec(prior)
    argmap = [f'{name}_{arg}' + (f'({components})' if argspec.annotations.get(arg) == typing.Sequence else '')
              for arg in argspec.args if arg not in ['f', 'df']]

    fs, dfs, fmats = zip(*[fourierbasis(psr, components, T) for psr in psrs])
    orfmat = np.array([[orf(p1.pos, p2.pos) for p1 in psrs] for p2 in psrs], dtype='d')

    orfs = matrix.jnparray(orfmat)
    f, df = matrix.jnparray(fs[0]), matrix.jnparray(dfs[0])
    def priorfunc(params):
        phidiag = prior(f, df, *[params[arg] for arg in argmap])

        return jnp.block([[jnp.diag(val * phidiag) for val in row] for row in orfs])
    priorfunc.params = argmap

    invorfs, invlogdet = matrix.jnparray(np.linalg.inv(orfmat)), 1.0 / jnp.linalg.slogdet(orfmat)[1]
    def invprior(params):
        invphidiag = 1.0 / prior(f, df, *[params[arg] for arg in argmap])

        return (jnp.block([[jnp.diag(val * invphidiag) for val in row] for row in invorfs]),
                jnp.sum(invlogdet + len(invphidiag) * jnp.log(invphidiag)))
    invprior.params = argmap

    # maybe make this a GlobalVariableGP
    return matrix.GlobalVariableGP(matrix.NoiseMatrix2D_var(priorfunc), fmats, invprior)


# priors: these need to be jax functions

def powerlaw(f, df, log10_A, gamma):
    return (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df

def freespectrum(f, df, log10_rho: typing.Sequence):
    return jnp.repeat(10.0**(2.0 * log10_rho), 2)

# combined red_noise + crn

def makepowerlaw_crn(components):
    if matrix.jnp == jnp:
        def powerlaw_crn(f, df, log10_A, gamma, crn_log10_A, crn_gamma):
            phi = (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df
            phi = phi.at[:2*components].add((10.0**(2.0 * crn_log10_A)) / 12.0 / jnp.pi**2 *
                                            const.fyr ** (crn_gamma - 3.0) * f[:2*components] ** (-crn_gamma) * df[:2*components])
            return phi
    elif matrix.jnp == np:
        def powerlaw_crn(f, df, log10_A, gamma, crn_log10_A, crn_gamma):
            phi = (10.0**(2.0 * log10_A)) / 12.0 / np.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df
            phi[:2*components] += ((10.0**(2.0 * crn_log10_A)) / 12.0 / np.pi**2 *
                                   const.fyr ** (crn_gamma - 3.0) * f[:2*components] ** (-crn_gamma) * df[:2*components])
            return phi

    return powerlaw_crn

# ORFs: OK as numpy functions

def hd_orf(pos1, pos2):
    if np.all(pos1 == pos2):
        return 1.0
    else:
        omc2 = (1.0 - np.dot(pos1, pos2)) / 2.0
        return 1.5 * omc2 * np.log(omc2) - 0.25 * omc2 + 0.5

def monopole_orf(pos1, pos2):
    if np.all(pos1 == pos2):
        # conditioning trick from enterprise
        return 1.0 + 1.0e-6
    else:
        return 1.0

def dipole_orf(pos1, pos2):
    if np.all(pos1 == pos2):
        return 1.0 + 1.0e-6
    else:
        return np.dot(pos1, pos2)

# delay

def makedelay(psr, delay, common=[], name='delay'):
    argspec = inspect.getfullargspec(delay)
    argmap = [(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}')
              for arg in argspec.args]

    def delayfunc(params):
        return delay(*[params[arg] for arg in argmap])
    delayfunc.params = argmap

    return delayfunc
