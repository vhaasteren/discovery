import inspect
import functools

import numpy as np
import scipy as sp
import scipy.constants as sc

import jax
import jax.config
jax.config.update("jax_enable_x64", True)
import jax.numpy
import jax.scipy

# import numba

def config(backend):
    global jnp, jsp, jaxarray, jaxjit, memclean

    if backend == 'jax':
        jnp, jsp = jax.numpy, jax.scipy
        jaxarray = jax.Array
        jaxjit = lambda f: f
        memclean = jax.clear_backends
    elif backend == 'numpy':
        jnp, jsp = np, sp
        jaxarray = np.ndarray
        jaxjit = lambda f: f
        memclean = lambda : None
#    elif backend == 'numba':
#        jnp, jsp = np, sp
#        jaxarray = np.ndarray
#        jaxjit = numba.jit
    else:
        raise ValueError('Supported backends are currently jax and numpy.')

config('jax')

class const:
    c = sc.speed_of_light
    G = sc.gravitational_constant
    h = sc.Planck

    yr = sc.Julian_year
    day = sc.day
    fyr = 1.0 / yr

    AU = sc.astronomical_unit
    ly = sc.light_year
    pc = sc.parsec
    kpc = pc * 1.0e3
    Mpc = pc * 1.0e6
    Gpc = pc * 1.0e9

    GMsun = 1.327124400e20
    Msun = GMsun / G
    Rsun = GMsun / (c**2)
    Tsun = GMsun / (c**3)

    erg = sc.erg

    DM_K = 2.41e-16  # for DM variation design matrix

    # relative angle between the Earth's ecliptic and the galactic equator
    e_ecl = 23.43704 * np.pi / 180.0

    # unit vector pointing in direction of angle between Earth's ecliptic and the galactic equator
    M_ecl = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(e_ecl), -np.sin(e_ecl)], [0.0, np.sin(e_ecl), np.cos(e_ecl)]])

class ConstantMatrix:
    pass

class VariableMatrix:
    pass

# consider passing inv as a 1D object

class NoiseMatrix1D_novar(ConstantMatrix):
    def __init__(self, N):
        self.N = N

        ld = jnp.sum(jnp.log(N))

        # could cache the inverse
        def inv():
            return jnp.diag(1.0 / N), ld

        def solve_1d(y):
            return y / N, ld

        def solve_2d(T):
            return T / N[:, jnp.newaxis], ld

        self.inv = jaxjit(inv)
        self.solve_1d = jaxjit(solve_1d)
        self.solve_2d = jaxjit(solve_2d)

        self.params = []

class NoiseMatrix1D_var(VariableMatrix):
    def __init__(self, getN):
        def N(params):
            return getN(params)

        def inv(params):
            N = getN(params)
            return jnp.diag(1.0 / N), jnp.sum(jnp.log(N))

        def solve_1d(y, params):
            N = getN(params)
            return y / N, jnp.sum(jnp.log(N))

        def solve_2d(T, params):
            N = getN(params)
            return T / N[:, jnp.newaxis], jnp.sum(jnp.log(N))

        def solve_12d(y, T, params):
            N = getN(params)
            return y / N, T / N[:, jnp.newaxis], jnp.sum(jnp.log(N))

        self.N = jaxjit(N)
        self.inv = jaxjit(inv)
        self.solve_1d = jaxjit(solve_1d)
        self.solve_2d = jaxjit(solve_2d)
        self.solve_12d = jaxjit(solve_12d)

        self.params = getN.params

def NoiseMatrix1D(N):
    return NoiseMatrix1D_var(N) if callable(N) else NoiseMatrix1D_novar(N)


class NoiseMatrix2D_novar(ConstantMatrix):
    def __init__(self, N):
        cf = jsp.linalg.cho_factor(N)
        ld = 2.0 * jnp.sum(jnp.log(jnp.diag(cf[0])))

        def solve(y):
            return jsp.linalg.cho_solve(cf, y), ld

        self.solve_1d = self.solve_2d = jaxjit(solve)

        self.params = []

class NoiseMatrix2D_var(VariableMatrix):
    def __init__(self, getN):
        def inv(params):
            N = getN(params)
            return jnp.linalg.inv(N), jnp.linalg.slogdet(N)[1]

        def solve(y, params):
            cf = jsp.linalg.cho_factor(getN(params))
            return jsp.linalg.cho_solve(cf, y), 2.0 * jnp.sum(jnp.log(jnp.diag(cf[0])))

        def solve_12d(y, T, params):
            cf = jsp.linalg.cho_factor(getN(params))
            return jsp.linalg.cho_solve(cf, y), jsp.linalg.cho_solve(cf, T), 2.0 * jnp.sum(jnp.log(jnp.diag(cf[0])))

        self.inv = jaxjit(inv)
        self.solve_1d = self.solve_2d = jaxjit(solve)
        self.solve_12d = jaxjit(solve_12d)

        self.params = getN.params

def NoiseMatrix2D(N):
    return NoiseMatrix2D_var(N) if callable(N) else NoiseMatrix2D_novar(N)


class ShermanMorrison_novar(ConstantMatrix):
    def __init__(self, N, F, P):
        # (N + F P F^T)^-1 = N^-1 - N^-1 F (P^-1 + F^T N^-1 F)^-1 F^T N^-1
        # |N + F P F^T| = |N| |P| |P^-1 + F^T N^-1 F|

        NmF, ldN = N.solve_2d(F)
        FtNmF = F.T @ NmF
        Pinv, ldP = P.inv()
        cf = jsp.linalg.cho_factor(Pinv + FtNmF)
        ld = ldN + ldP + 2.0 * jnp.sum(jnp.log(jnp.diag(cf[0])))

        def solve_1d(y):
            # a = N.solve_1d(y)[0]
            # b1 = NmF.T @ y
            # b2 = jsp.linalg.cho_solve(cf, b1)
            # b3 = NmF @ b2
            # return a - b3, ld

            return N.solve_1d(y)[0] - NmF @ jsp.linalg.cho_solve(cf, NmF.T @ y), ld

        def solve_2d(y):
            return N.solve_2d(y)[0] - NmF @ jsp.linalg.cho_solve(cf, NmF.T @ y), ld

        self.solve_1d = jaxjit(solve_1d)
        self.solve_2d = jaxjit(solve_2d)

        self.params = []

class ShermanMorrison_varN(VariableMatrix):
    def __init__(self, N_var, F, P):
        Pinv, ldP = P.inv()

        def solve_1d(y, params):
            Nmy, NmF, ldN = N_var.solve_12d(y, F, params)
            cf = jsp.linalg.cho_factor(Pinv + F.T @ NmF)

            return (Nmy - NmF @ jsp.linalg.cho_solve(cf, NmF.T @ y),
                    ldN + ldP + 2.0 * jnp.sum(jnp.log(jnp.diag(cf[0]))))

        self.solve_1d = jaxjit(solve_1d)

        self.params = N_var.params


class VariableKernel:
    pass

class ShermanMorrisonKernel_varP(VariableKernel):
    def __init__(self, N, F, P_var):
        self.N, self.F, self.P_var = N, F, P_var

    def make_kernelproduct(self, y):
        NmF, ldN = self.N.solve_2d(self.F)
        FtNmF = self.F.T @ NmF

        ytNmy, _ = y @ self.N.solve_1d(y)
        NmFty = NmF.T @ y

        P_var_inv = self.P_var.inv

        # closes on P_var_inv, FtNmF, NmFty, ytNmy, ldN
        @jaxjit
        def kernelproduct(params):
            Pinv, ldP = P_var_inv.inv(params)
            cf = jsp.linalg.cho_factor(Pinv + FtNmF)
            ytXy = NmFty.T @ jsp.linalg.cho_solve(cf, NmFty)

            return -0.5 * (ytNmy - ytXy) - 0.5 * (ldN + ldP + 2.0 * jnp.sum(jnp.log(jnp.diag(cf[0]))))

        kernelproduct.params = self.P_var.params

        return kernelproduct

    def make_kernelterms(self, y, T):
        # Sigma = (N + F P Ft)
        # Sigma^-1 = Nm - Nm F (P^-1 + Ft Nm F)^-1 Ft Nm
        #
        # yt Sigma^-1 y = yt Nm y - (yt Nm F) C^-1 (Ft Nm y)
        # Tt Sigma^-1 y = Tt Nm y - Tt Nm F C^-1 (Ft Nm y)
        # Tt Sigma^-1 T = Tt Nm T - (Tt Nm F) C^-1 (Ft Nm T)

        Nmy, ldN = self.N.solve_1d(y)
        ytNmy = y @ Nmy
        FtNmy = self.F.T @ Nmy
        TtNmy = T.T @ Nmy

        NmF, _ = self.N.solve_2d(self.F)
        FtNmF = self.F.T @ NmF
        TtNmF = T.T @ NmF

        NmT, _ = self.N.solve_2d(T)
        FtNmT = self.F.T @ NmT
        TtNmT = T.T @ NmT

        P_var_inv = self.P_var.inv

        # closes on P_var, FtNmF, FtNmy, FtMmT, ytNmy, TtNmy, TtNmT, TtNmF
        def kernelterms(params):
            Pinv, ldP = P_var_inv(params)
            cf = jsp.linalg.cho_factor(Pinv + FtNmF)

            sol = jsp.linalg.cho_solve(cf, FtNmy)
            sol2 = jsp.linalg.cho_solve(cf, FtNmT)

            a = -0.5 * (ytNmy - FtNmy.T @ sol) - 0.5 * (ldN + ldP + 2.0 * jnp.sum(jnp.log(jnp.diag(cf[0]))))
            b = TtNmy - TtNmF @ sol
            c = TtNmT - TtNmF @ sol2

            return a, b, c

        kernelterms.params = self.P_var.params

        return kernelterms


class ShermanMorrisonKernel_varP_old(VariableKernel):
    def __init__(self, y, N, F, P_var):
        NmF, ldN = N.solve_2d(F)
        FtNmF = F.T @ NmF

        ytNmy, _ = y @ N.solve_1d(y)
        NmFty = NmF.T @ y

        # closes on P_var, FtNmF, NmFty, ytNmy, ldN
        def kernelproduct(params):
            Pinv, ldP = P_var.inv(params)
            cf = jsp.linalg.cho_factor(Pinv + FtNmF)
            ytXy = NmFty.T @ jsp.linalg.cho_solve(cf, NmFty)

            return -0.5 * (ytNmy - ytXy) - 0.5 * (ldN + ldP + 2.0 * jnp.sum(jnp.log(jnp.diag(cf[0]))))

        self.kernelproduct = jaxjit(kernelproduct)

        self.params = P_var.params

        self.y, self.N, self.F, self.P_var = y, N, F, P_var

    # TO DO: some redundant storage here...
    def make_kernelterms(self, T):
        y, N, F, P_var = self.y, self.N, self.F, self.P_var

        # Sigma = (N + F P Ft)
        # Sigma^-1 = Nm - Nm F (P^-1 + Ft Nm F)^-1 Ft Nm
        #
        # yt Sigma^-1 y = yt Nm y - (yt Nm F) C^-1 (Ft Nm y)
        # Tt Sigma^-1 y = Tt Nm y - Tt Nm F C^-1 (Ft Nm y)
        # Tt Sigma^-1 T = Tt Nm T - (Tt Nm F) C^-1 (Ft Nm T)

        Nmy, ldN = N.solve_1d(y)
        ytNmy = y @ Nmy
        FtNmy = F.T @ Nmy
        TtNmy = T.T @ Nmy

        NmF, _ = N.solve_2d(F)
        FtNmF = F.T @ NmF
        TtNmF = T.T @ NmF

        NmT, _ = N.solve_2d(T)
        FtNmT = F.T @ NmT
        TtNmT = T.T @ NmT

        # closes on P_var, FtNmF, FtNmy, FtMmT, ytNmy, TtNmy, TtNmT, TtNmF
        def kernelterms(params):
            Pinv, ldP = P_var.inv(params)
            cf = jsp.linalg.cho_factor(Pinv + FtNmF)

            sol = jsp.linalg.cho_solve(cf, FtNmy)
            sol2 = jsp.linalg.cho_solve(cf, FtNmT)

            a = -0.5 * (ytNmy - FtNmy.T @ sol) - 0.5 * (ldN + ldP + 2.0 * jnp.sum(jnp.log(jnp.diag(cf[0]))))
            b = TtNmy - TtNmF @ sol
            c = TtNmT - TtNmF @ sol2

            return a, b, c

        return kernelterms


class ShermanMorrison_varP(VariableMatrix):
    def __init__(self, N, F, P_var):
        NmF, ldN = N.solve_2d(F)
        FtNmF = F.T @ NmF

        def solve_1d(y, params):
            Pinv, ldP = P_var.inv(params)
            cf = jsp.linalg.cho_factor(Pinv + FtNmF)

            # a = N.solve_1d(y)[0]
            # b = NmF @ jsp.linalg.cho_solve(cf, NmF.T @ y)
            # return (a - b,
            #         ldN + ldP + 2.0 * jnp.sum(jnp.log(jnp.diag(cf[0]))))

            return (N.solve_1d(y)[0] - NmF @ jsp.linalg.cho_solve(cf, NmF.T @ y),
                    ldN + ldP + 2.0 * jnp.sum(jnp.log(jnp.diag(cf[0]))))

        def solve_12d(y, T, params):
            Pinv, ldP = P_var.inv(params)
            cf = jsp.linalg.cho_factor(Pinv + FtNmF)

            # not a lot gained by calling N.solve_12d...
            return (N.solve_1d(y)[0] - NmF @ jsp.linalg.cho_solve(cf, NmF.T @ y),
                    N.solve_2d(T)[0] - NmF @ jsp.linalg.cho_solve(cf, NmF.T @ T),
                    ldN + ldP + 2.0 * jnp.sum(jnp.log(jnp.diag(cf[0]))))

        self.solve_1d = jaxjit(solve_1d)
        self.solve_12d = jaxjit(solve_12d)

        self.params = P_var.params


class ShermanMorrison_varNP(VariableMatrix):
    def __init__(self, N_var, F, P_var):
        def solve_1d(y, params):
            Nmy, NmF, ldN = N_var.solve_12d(y, F, params)
            Pinv, ldP = P_var.inv(params)
            cf = jsp.linalg.cho_factor(Pinv + F.T @ NmF)

            return (Nmy - NmF @ jsp.linalg.cho_solve(cf, NmF.T @ y),
                    ldN + ldP + 2.0 * jnp.sum(jnp.log(jnp.diag(cf[0]))))

        self.solve_1d = jaxjit(solve_1d)

        self.params = N_var.params + P_var.params

# if y is constant, this can be optimized further by caching all the N.solves
# we should rename the more general version ShermanMorrison_varyP
def ShermanMorrison(N, F, P):
    if not isinstance(F, jaxarray):
        raise TypeError("F must be a JAX array.")

    if   isinstance(N, ConstantMatrix) and isinstance(P, ConstantMatrix):
        return ShermanMorrison_novar(N, F, P)
    elif isinstance(N, ConstantMatrix) and isinstance(P, VariableMatrix):
        return ShermanMorrison_varP(N, F, P)
    elif isinstance(N, VariableMatrix) and isinstance(P, ConstantMatrix):
        return ShermanMorrison_varN(N, F, P)
    elif isinstance(N, VariableMatrix) and isinstance(P, VariableMatrix):
        return ShermanMorrison_varNP(N, F, P)
    else:
        raise TypeError("N and P must be a ConstantMatrix or VariableMatrix")



# power-law GP priors

def powerlaw(f, df, log10_A, gamma):
    return (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df

def freespectrum(f, df, log10_rho):
    return jnp.repeat(10.0**(2.0 * log10_rho), 2)

def fourierbasis(psr, components, T=None):
    if T is None:
        T  = psr.toas.max() - psr.toas.min()

    f  = jnp.arange(1, components + 1, dtype=jnp.float64) / T
    df = jnp.diff(jnp.concatenate((jnp.array([0]), f)))

    fmat = jnp.zeros((psr.toas.shape[0], 2*components), dtype=jnp.float64)
    for i in range(components):
        if jnp == jax.numpy:
            fmat = fmat.at[:, 2*i  ].set(jnp.sin(2.0 * jnp.pi * f[i] * psr.toas))
            fmat = fmat.at[:, 2*i+1].set(jnp.cos(2.0 * jnp.pi * f[i] * psr.toas))
        elif jnp == np:
            fmat[:, 2*i  ] = jnp.sin(2.0 * jnp.pi * f[i] * psr.toas)
            fmat[:, 2*i+1] = jnp.cos(2.0 * jnp.pi * f[i] * psr.toas)
        else:
            raise ValueError("Unsupported backend in fourierbasis")

    return jnp.repeat(f, 2), jnp.repeat(df, 2), fmat # jnp.cos(2.0 * jnp.pi * f[i] * psr.toas) #fmat

# GP factories
class ConstantGP:
    def __init__(self, Phi, F):
        self.Phi, self.F = Phi, F

class VariableGP:
    def __init__(self, Phi, F):
        self.Phi, self.F = Phi, F

# used by PulsarLikelihood
def CompoundGP(gplist):
    if len(gplist) == 1:
        return gplist[0]

    if all(isinstance(gp, ConstantGP) for gp in gplist) and all(isinstance(gp.Phi, NoiseMatrix1D_novar) for gp in gplist):
        F = jnp.hstack([gp.F for gp in gplist])
        Phi = jnp.concatenate([gp.Phi.N for gp in gplist])

        return ConstantGP(NoiseMatrix1D_novar(Phi), F)
    elif all(isinstance(gp, VariableGP) for gp in gplist) and all(isinstance(gp.Phi, NoiseMatrix1D_var) for gp in gplist):
        F = jnp.hstack([gp.F for gp in gplist])

        @jaxjit
        def Phi(params):
            return jnp.concatenate([gp.Phi.N(params) for gp in gplist])

        Phi.params = sorted(set.union(*[set(gp.Phi.params) for gp in gplist]))

        return VariableGP(NoiseMatrix1D_var(Phi), F)

    raise NotImplementedError("Cannot concatenate these types of GPs.")


def makepowerlaw_crn(components):
    if jnp == jax.numpy:
        def powerlaw_crn(f, df, log10_A, gamma, crn_log10_A, crn_gamma):
            phi = (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df
            phi = phi.at[:2*components].add((10.0**(2.0 * crn_log10_A)) / 12.0 / jnp.pi**2 *
                                            const.fyr ** (crn_gamma - 3.0) * f[:2*components] ** (-crn_gamma) * df[:2*components])
            return phi
    elif jnp == np:
        def powerlaw_crn(f, df, log10_A, gamma, crn_log10_A, crn_gamma):
            phi = (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df
            phi[:2*components] += ((10.0**(2.0 * crn_log10_A)) / 12.0 / jnp.pi**2 *
                                   const.fyr ** (crn_gamma - 3.0) * f[:2*components] ** (-crn_gamma) * df[:2*components])
            return phi
    else:
        raise ValueError("Unsupported backend in makepowerlaw_crn")

    return powerlaw_crn

def makegp_fourier(gpname, psr, prior, components, T=None, common=[]):
    f, df, fmat = fourierbasis(psr, components, T)

    argmap = [arg if arg in common else f'{psr.name}_{gpname}_{arg}'
              for arg in inspect.getfullargspec(prior).args
              if arg not in ['f', 'df']]

    prior = jaxjit(prior)

    # @jaxjit
    def priorfunc(params):
        return prior(f, df, *[params[arg] for arg in argmap])

    priorfunc.params = argmap

    return VariableGP(NoiseMatrix1D_var(priorfunc), fmat)

def makegp_improper(gpname, psr, fmat, constant=1.0e40):
    return ConstantGP(NoiseMatrix1D_novar(constant * jnp.ones(fmat.shape[1])), fmat)

def makegp_timing(psr, constant=1.0e40):
    fmat = jnp.array(psr.Mmat / np.sqrt(np.sum(psr.Mmat**2, axis=0)), dtype=jnp.float64)

    return makegp_improper('timing', psr, fmat, constant=constant)

# residuals

def residuals(psr):
    return jnp.array(psr.residuals, dtype=jnp.float64)

def getspan(psrs):
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]

    return np.max(tmax) - np.min(tmin)

# EFAC/EQUAD/ECORR noise

# no backends
def makenoise_measurement_simple(psr, noisedict={}):
    toaerrs = jnp.array(psr.toaerrs)

    efac = f'{psr.name}_efac'
    log10_t2equad = f'{psr.name}_log10_t2equad'

    def getnoise(params):
        return params[efac]**2 * (toaerrs**2 + 10.0**(2.0 * params[log10_t2equad]))

    getnoise.params = [efac, log10_t2equad]

    if all(par in noisedict for par in getnoise.params):
        return NoiseMatrix1D_novar(getnoise(noisedict))
    else:
        return NoiseMatrix1D_var(jaxjit(getnoise))

def makenoise_measurement(psr, noisedict={}):
    toaerrs = jnp.array(psr.toaerrs)

    backends = sorted(set(psr.backend_flags))

    masks = [jnp.array(psr.backend_flags == backend).astype(jnp.float64) for backend in backends]

    efacs = [f'{psr.name}_{backend}_efac' for backend in backends]
    log10_t2equads = [f'{psr.name}_{backend}_log10_t2equad' for backend in backends]

    def getnoise(params):
        return sum(mask * params[efac]**2 * (toaerrs**2 + 10.0**(2 * params[log10_t2equad]))
                   for mask, efac, log10_t2equad in zip(masks, efacs, log10_t2equads))

    getnoise.params = efacs + log10_t2equads

    if all(par in noisedict for par in getnoise.params):
        return NoiseMatrix1D_novar(getnoise(noisedict))
    else:
        return NoiseMatrix1D_var(jaxjit(getnoise))

# ecorr
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
    bins = quantize(psr.toas)
    Umat = jnp.array(np.vstack([bins == i for i in range(bins.max() + 1)]).astype(np.float64).T)

    ones = jnp.ones(Umat.shape[1], dtype=jnp.float64)

    log10_ecorr = f'{psr.name}_log10_ecorr'

    def getphi(params):
        return (10.0**(2.0 * params[log10_ecorr])) * ones

    getphi.params = [log10_ecorr]

    if all(par in noisedict for par in getphi.params):
        return ConstantGP(NoiseMatrix1D_novar(getphi(noisedict)), Umat)
    else:
        return VariableGP(NoiseMatrix1D_var(getphi), Umat)

def makegp_ecorr(psr, noisedict={}):
    backends = sorted(set(psr.backend_flags))
    masks = [np.array(psr.backend_flags == backend) for backend in backends]

    log10_ecorrs, Umats = [], []
    for backend, mask in zip(backends, masks):
        log10_ecorrs.append(f'{psr.name}_{backend}_log10_ecorr')

        bins = quantize(psr.toas * mask)
        Umats.append(np.vstack([bins == i for i in range(1, bins.max() + 1)]).astype(np.float64).T)

    Umatall = jnp.array(np.hstack(Umats))

    pmasks, cnt = [], 0
    for Umat in Umats:
        if jnp == jax.numpy:
            pmasks.append(jnp.zeros(Umatall.shape[1], dtype=jnp.float64).at[cnt:cnt+Umat.shape[1]].set(1.0))
        elif jnp == np:
            z = jnp.zeros(Umatall.shape[1], dtype=jnp.float64)
            z[cnt:cnt+Umat.shape[1]] = 1.0
            pmasks.append(z)
        else:
            raise ValueError("Backend not supported in makegp_corr")

        cnt = cnt + Umat.shape[1]

    def getphi(params):
        return sum(10.0**(2 * params[log10_ecorr]) * pmask for (log10_ecorr, pmask) in zip(log10_ecorrs, pmasks))

    getphi.params = log10_ecorrs

    if all(par in noisedict for par in getphi.params):
        return ConstantGP(NoiseMatrix1D_novar(getphi(noisedict)), Umatall)
    else:
        return VariableGP(NoiseMatrix1D_var(jaxjit(getphi)), Umatall)

# global correlated Fourier GP

def hd_orf(pos1, pos2):
    if np.all(pos1 == pos2):
        return 1.0
    else:
        omc2 = (1.0 - np.dot(pos1, pos2)) / 2.0
        return 1.5 * omc2 * np.log(omc2) - 0.25 * omc2 + 0.5

# note the resulting VariableGP has a big prior, but F is a list of Fmats
def makegp_fourier_global(gpname, psrs, orf, prior, components, T):
    fs, dfs, fmats = zip(*[fourierbasis(psr, components, T) for psr in psrs])
    f, df = fs[0], dfs[0]

    orfs = jnp.array([[orf(p1.pos, p2.pos) for p1 in psrs] for p2 in psrs])

    argmap = [f'{gpname}_{arg}' for arg in inspect.getfullargspec(prior).args
              if arg not in ['f', 'df']]

    @jaxjit
    def priorfunc(params):
        phidiag = prior(f, df, *[params[arg] for arg in argmap])

        return jnp.block([[jnp.diag(val * phidiag) for val in row] for row in orfs])

    priorfunc.params = argmap

    # maybe make this a GlobalVariableGP
    return VariableGP(NoiseMatrix2D_var(priorfunc), fmats)

# likelihood

# TO DO: there should be no need for this... this should be replaced
#        by a more general call to ShermanMorrisonKernel
def NormalLikelihood_varS(y, S):
    if isinstance(S, ConstantMatrix):
        @jaxjit
        def normal(params):
            Sy, ldS = S.solve_1d(y)
            return -0.5 * y @ Sy - 0.5 * ldS
    elif isinstance(S, VariableMatrix):
        @jaxjit
        def normal(params):
            Sy, ldS = S.solve_1d(y, params)
            return -0.5 * y @ Sy - 0.5 * ldS
    elif isinstance(S, VariableKernel):
        normal = S.kernelproduct

    normal.params = S.params

    return normal

def ShermanMorrisonKernel(N, F, P):
    if not isinstance(F, jaxarray):
        raise TypeError("F must be a JAX array.")

    if isinstance(N, ConstantMatrix) and isinstance(P, VariableMatrix):
        return ShermanMorrisonKernel_varP(N, F, P)
    else:
        raise TypeError("N must be a ConstantMatrix and P a VariableMatrix")


class GlobalLikelihood:
    def __init__(self, psls, varGP=None, clean=False):
        self.psls, self.varGP, self.clean = psls, varGP, clean

    @functools.cached_property
    def logL(self):
        psls, varGP = self.psls, self.varGP

        if varGP is None:
            lls = []
            for psl in psls:
                lls.append(psl.logL)

                if self.clean:
                    memclean()

            @jaxjit
            def loglike(params):
                return sum(ll(params) for ll in lls)

            loglike.params = sorted(set.union(*[set(ll.params) for ll in lls]))
        else:
            P_var, Fs = varGP.Phi, varGP.F

            kterms = []
            for psl, F in zip(psls, Fs):
                if isinstance(psl.N, VariableKernel):
                    kterms.append(jax.jit(psl.N.make_kernelterms(psl.y, F)))
                else:
                    kterms.append((psl.y, psl.N))

                    if self.clean:
                        memclean()

            if callable(kterms[0]):
                @jaxjit
                def loglike(params):
                    terms = [kterm(params) for kterm in kterms]

                    p0 = sum([term[0] for term in terms])
                    FtNmy = jnp.concatenate([term[1] for term in terms])

                    Pinv, ldP = P_var.inv(params)
                    cf = jsp.linalg.cho_factor(Pinv + jsp.linalg.block_diag(*[term[2] for term in terms]))

                    return p0 + 0.5 * (FtNmy.T @ jsp.linalg.cho_solve(cf, FtNmy) - ldP - 2.0 * jnp.sum(jnp.log(jnp.diag(cf[0]))))

                loglike.params = sorted(set.union(*[set(kterm.params) for kterm in kterms])) + P_var.params
            else:
                ys, N_vars = zip(*kterms)

                @jaxjit
                def loglike(params):
                    # Nmy, NmF, ldN
                    terms = [N.solve_12d(y, F, params) for N, y, F in zip(N_vars, ys, Fs)]

                    p0 = jnp.sum([-0.5 * y @ term[0] - 0.5 * term[2] for y, term in zip(ys, terms)])

                    NmFy = jnp.concatenate([term[1].T @ y for y, term in zip(ys, terms)])
                    # NmFy = jnp.concatenate([F.T @ term[0] for F, term in zip(Fs, terms)])

                    Pinv, ldP = P_var.inv(params)
                    cf = jsp.linalg.cho_factor(Pinv + jsp.linalg.block_diag(*[F.T @ term[1] for F, term in zip(Fs, terms)]))

                    return p0 + 0.5 * (NmFy @ jsp.linalg.cho_solve(cf, NmFy) - ldP - 2.0 * jnp.sum(jnp.log(jnp.diag(cf[0]))))

                loglike.params = sorted(set.union(*[set(N_var.params) for N_var in N_vars])) + P_var.params

        return loglike


class PulsarLikelihood:
    def __init__(self, args, concat=True, optimize=True):
        y     = [arg for arg in args if isinstance(arg, jaxarray)]
        noise = [arg for arg in args if isinstance(arg, ConstantMatrix) or isinstance(arg, VariableMatrix)]
        cgps  = [arg for arg in args if isinstance(arg, ConstantGP)]
        vgps  = [arg for arg in args if isinstance(arg, VariableGP)]

        if len(y) > 1 or len(noise) > 1:
            raise ValueError("Only one residual vector and one NoiseMatrix allowed.")
        elif len(noise) == 0:
            raise ValueError("I need exactly one NoiseMatrix.")

        y0 = y[0]

        # GPs can be concatenated or nested

        if cgps:
            if concat:
                cgp = CompoundGP(cgps)
                csm = ShermanMorrison(noise[0], cgp.F, cgp.Phi)
            else:
                csm = noise[0]
                for cgp in cgps:
                    csm = ShermanMorrison(csm, cgp.F, cgp.Phi)
        else:
            csm = noise[0]

        if vgps:
            if concat:
                vgp = CompoundGP(vgps)

                # this optimization only possible if y is constant and we concatenate GPs
                if optimize:
                    vsm = ShermanMorrisonKernel(csm, vgp.F, vgp.Phi)
                else:
                    vsm = ShermanMorrison(csm, vgp.F, vgp.Phi)
            else:
                vsm = csm
                for vgp in vgps:
                    vsm = ShermanMorrison(vsm, vgp.F, vgp.Phi)
        else:
            vsm = csm

        self.y, self.N = y0, vsm
        # self.logL = NormalLikelihood_varS(y0, vsm)

    @functools.cached_property
    def logL(self):
        N, y = self.N, self.y

        if isinstance(N, VariableKernel):
            loglike = N.make_kernelproduct(y)
        elif isinstance(N, ConstantMatrix):
            @jaxjit
            def loglike(params):
                Nmy, ldN = N.solve_1d(y)
                return -0.5 * y @ Nmy - 0.5 * ldN
        elif isinstance(S, VariableMatrix):
            @jaxjit
            def loglike(params):
                Nmy, ldN = N.solve_1d(y, params)
                return -0.5 * y @ Nmy - 0.5 * ldN
        else:
            raise ValueError("Cannot make pulsar likelihood.")

        loglike.params = N.params

        return loglike

# psr = psrs[0]
# Tspan = model_utils.get_tspan(psrs[0:1])

# je.config('numpy')

# npta = je.PulsarLikelihood(je.residuals(psr),
#                            je.makenoise_measurement(psr, noisedict),
#                            je.makegp_ecorr(psr, noisedict),
#                            je.makegp_timing(psr),
#                            je.makegp_fourier('red_noise', psr, je.powerlaw, 10),
#                            je.makegp_fourier('red_noise', psr,
#                                              je.makepowerlaw_crn(5), 10, T=Tspan,
#                                              common=['crn_log10_A', 'crn_gamma']),
#                            concat=True)

# pta = getpta([psr], gw=True, inc_ecorr=True, gp_ecorr=False, marginalizing=True)
# p0 = parameter.sample(pta.params)

# pta.get_lnlikelihood(p0)
# npta.logL(p0)

# je.config('jax')

# lpta = je.PulsarLikelihood(je.residuals(psr),
#                            je.makenoise_measurement(psr, noisedict),
#                            je.makegp_ecorr(psr, noisedict),
#                            je.makegp_timing(psr),
#                            je.makegp_fourier('red_noise', psr, je.powerlaw, 10),
#                            je.makegp_fourier('red_noise', psr,
#                                              je.makepowerlaw_crn(5), 10, T=Tspan,
#                                              common=['crn_log10_A', 'crn_gamma']),
#                            concat=True, optimize=True)
# class jpta:
#     pass

# jpta.logL = jax.jit(lpta.logL)

# lpta.logL(p0)
# jpta.logL(p0)