from collections.abc import Sequence

import numpy as np
import scipy as sp

import jax
import jax.numpy
import jax.scipy

def config(**kwargs):
    global jnp, jsp, jnparray, jnpzeros, intarray, jnpkey, jnpsplit, jnpnormal
    global matrix_factor, matrix_solve, matrix_norm

    backend = kwargs.get('backend')

    if backend == 'numpy':
        jnp, jsp = np, sp

        jnparray = lambda a: np.array(a, dtype=np.float64)
        jnpzeros = lambda a: np.zeros(a, dtype=np.float64)
        intarray = lambda a: np.array(a, dtype=np.int64)

        jnpkey    = lambda seed: np.random.default_rng(seed)
        jnpsplit  = lambda gen: (gen, gen)
        jnpnormal = lambda gen, shape: gen.normal(size=shape)
    elif backend == 'jax':
        jnp, jsp = jax.numpy, jax.scipy

        jnparray = lambda a: jnp.array(a, dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32)
        jnpzeros = lambda a: jnp.zeros(a, dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32)
        intarray = lambda a: jnp.array(a, dtype=jnp.int64)

        jnpkey    = lambda seed: jax.random.PRNGKey(seed)
        jnpsplit  = jax.random.split
        jnpnormal = jax.random.normal

    factor = kwargs.get('factor')

    if factor == 'cholesky':
        matrix_factor = jsp.linalg.cho_factor
        matrix_solve  = jsp.linalg.cho_solve
        matrix_norm   = 2.0
    elif factor == 'lu':
        matrix_factor = jsp.linalg.lu_factor
        matrix_solve  = jsp.linalg.lu_solve
        matrix_norm   = 1.0

config(backend='jax', factor='cholesky')


def rngkey(seed):
    return jnpkey(seed)

class ConstantMatrix:
    pass

class VariableMatrix:
    pass

class Kernel:
    pass

class ConstantKernel(Kernel):
    pass

class VariableKernel(Kernel):
    pass

class GP:
    pass

class ConstantGP:
    def __init__(self, Phi, F):
        self.Phi, self.F = Phi, F

class VariableGP:
    def __init__(self, Phi, F):
        self.Phi, self.F = Phi, F

# class ComponentGP:
#     def __init__(self, Phi, F, cfunc):
#         self.Phi, self.F, self.cfunc = Phi, F, cfunc



# note that all factories that return a GlobalVariableGP should define its `index`
# as a dictionary of component vector names to slices within the Fs matrix, which
# is used by GlobalLikelihood.sample_conditional to parse out the vectors

class GlobalVariableGP:
    def __init__(self, Phi, Fs, Phi_inv=None):
        self.Phi, self.Fs, self.Phi_inv = Phi, Fs, Phi_inv

def CompoundGlobalGP(gplist):
    if all(isinstance(gp, GlobalVariableGP) for gp in gplist):
        fmats = [np.hstack(F) for F in zip(*[gp.Fs for gp in gplist])]

        npsr = len(fmats)

        ngps = [gp.Fs[0].shape[1] for gp in gplist]
        allgp = sum(ngps)

        offsets = [0] + list(np.cumsum(ngps))[:-1]

        if all(isinstance(gp.Phi, NoiseMatrix1D_var) for gp in gplist):
            def priorfunc(params):
                ret = jnp.zeros(npsr*allgp, 'd')

                for gp, ngp, offset in zip(gplist, ngps, offsets):
                    phi = gp.Phi.getN(params)

                    # just look at this craziness... need another code path for numpy
                    ret = jax.lax.fori_loop(0, npsr, lambda i, ret:
                            jax.lax.dynamic_update_slice(ret,
                                                         jax.lax.dynamic_slice(phi, (i*ngp,), (ngp,)),
                                                         (i*allgp + offset,)),
                            ret)

                    for i in range(npsr):
                        ret = ret.at[i*allgp+offset:i*allgp+offset+ngp].set(phi[i*ngp:(i+1)*ngp])

                return ret
            priorfunc.params = sorted(set.union(*[set(gp.Phi.params) for gp in gplist]))

            # def invprior(params):
            #     ps, ls = zip(*[gp.Phi_inv(params) for gp in gplist])
            #     return jnp.diag(jnp.concatenate(ps)), sum(ls)
            # invprior.params = sorted(set.union(*[set(gp.Phi_inv.params) for gp in gplist]))

            multigp = GlobalVariableGP(NoiseMatrix1D_var(priorfunc), fmats, None) # invprior
        else:
            def priorfunc(params):
                ret = jnp.zeros((npsr*allgp, npsr*allgp), 'd')

                for gp, ngp, offset in zip(gplist, ngps, offsets):
                    phi = jnp.diag(gp.Phi.getN(params)) if isinstance(gp.Phi, NoiseMatrix1D_var) else gp.Phi.getN(params)

                    ret = jax.lax.fori_loop(0, npsr, lambda i, ret:
                            jax.lax.fori_loop(0, npsr, lambda j, ret:
                                jax.lax.dynamic_update_slice(ret,
                                                             jax.lax.dynamic_slice(phi, (i*ngp, j*ngp), (ngp, ngp)),
                                                             (i*allgp + offset, j*allgp + offset)),
                                ret),
                            ret)

                return ret

            priorfunc.params = sorted(set.union(*[set(gp.Phi.params) for gp in gplist]))

            phiinvs = [gp.Phi_inv or gp.Phi.make_inv() for gp in gplist]

            def invprior(params):
                ret = jnp.zeros((npsr*allgp, npsr*allgp), 'd')
                ps, ls = zip(*[phiinv(params) for phiinv in phiinvs])

                for p, ngp, offset in zip(ps, ngps, offsets):
                    phiinv = jnp.diag(p) if p.ndim == 1 else p

                    ret = jax.lax.fori_loop(0, npsr, lambda i, ret:
                            jax.lax.fori_loop(0, npsr, lambda j, ret:
                                jax.lax.dynamic_update_slice(ret,
                                                             jax.lax.dynamic_slice(phiinv, (i*ngp, j*ngp), (ngp, ngp)),
                                                             (i*allgp + offset, j*allgp + offset)),
                                ret),
                            ret)

                return ret, sum(ls)

            invprior.params = sorted(set.union(*[set(gp.Phi.params) for gp in gplist]))

            multigp = GlobalVariableGP(NoiseMatrix2D_var(priorfunc), fmats, invprior)
    else:
        raise NotImplementedError("Cannot concatenate these types of GlobalGPs.")

    index, cnt = {}, 0
    for vars in zip(*[gp.index.items() for gp in gplist]):
        for var, sli in vars:
            width = sli.stop - sli.start
            index[var] = slice(cnt, cnt + width)
            cnt = cnt + width
    multigp.index = index

    return multigp

# concatenate GPs

def VectorCompoundGP(gplist):
    if gplist is None or not isinstance(gplist, Sequence):
        return gplist
    elif len(gplist) == 1:
        return gplist[0]

    if all(isinstance(gp, (VariableGP, GlobalVariableGP)) for gp in gplist):
        # each gp.F is a tuple of F matrices, one for each pulsar
        # globalgp has gp.Fs instead
        F = [np.hstack(Fs) for Fs in zip(*[gp.F if hasattr(gp, 'F') else gp.Fs for gp in gplist])]

        if all(isinstance(gp.Phi, VectorNoiseMatrix1D_var) for gp in gplist):
            def Phi(params):
                return jnp.hstack([gp.Phi.getN(params) for gp in gplist])
            Phi.params = sorted(set.union(*[set(gp.Phi.params) for gp in gplist]))

            multigp = VariableGP(VectorNoiseMatrix1D_var(Phi), F)
        elif all(isinstance(gp.Phi, (VectorNoiseMatrix1D_var, NoiseMatrix2D_var)) for gp in gplist):
            cvarslist = [list(gp.index) for gp in gplist]
            pinvlist = [gp.Phi.make_inv() for gp in gplist]

            def priorfunc(params):
                ret = 0.0

                for cvars, pinv in zip(cvarslist, pinvlist):
                    Pm, ldP = pinv(params)

                    if Pm.shape[0] == Pm.shape[1]: # 2D prior
                        c = jnp.concatenate([params[cvar] for cvar in cvars])
                        ret = ret - 0.5 * c @ Pm @ c - 0.5 * ldP
                    else: # batched prior
                        c = jnp.array([params[cvar] for cvar in cvars])
                        ret = ret - 0.5 * jnp.sum(c * Pm * c) - 0.5 * jnp.sum(ldP)

                return ret

            priorfunc.params = sorted(set.union(*[set(gp.Phi.params) for gp in gplist]))

            multigp = VariableGP(None, F)
            multigp.prior = priorfunc
        else:
            raise TypeError('VectorCompoundGP works only with VectorNoiseMatrix1D_var so far.')
    else:
        raise TypeError('VectorCompoundGP works only with VariableGPs so far.')

    multigp.index = [dict(g) for g in zip(*[gp.index.items() for gp in gplist])]

    return multigp

def CompoundGP(gplist):
    if len(gplist) == 1:
        return gplist[0]

    if all(isinstance(gp, ConstantGP) for gp in gplist):
        if all(isinstance(gp.Phi, NoiseMatrix1D_novar) for gp in gplist):
            F = np.hstack([gp.F for gp in gplist])
            PhiN = np.concatenate([gp.Phi.N for gp in gplist])

            multigp = ConstantGP(NoiseMatrix1D_novar(PhiN), F)
        elif all(isinstance(gp.Phi, (NoiseMatrix1D_novar, NoiseMatrix2D_novar)) for gp in gplist):
            F = np.hstack([gp.F for gp in gplist])
            PhiN = jsp.linalg.block_diag(*[np.diag(gp.Phi.N) if isinstance(gp.Phi, NoiseMatrix1D_novar)
                                                             else gp.Phi.N
                                          for gp in gplist])

            multigp = ConstantGP(NoiseMatrix2D_novar(PhiN), F)
    elif all(isinstance(gp, VariableGP) for gp in gplist):
        if any(callable(gp.F) for gp in gplist):
            def F(params):
                return jnp.hstack([gp.F(params) if callable(gp.F) else gp.F for gp in gplist])
            F.params = sum((gp.F.params if callable(gp.F) else [] for gp in gplist), [])
        else:
            F = np.hstack([gp.F for gp in gplist])

        if all(isinstance(gp.Phi, NoiseMatrix1D_var) for gp in gplist):
            def Phi(params):
                return jnp.concatenate([gp.Phi.getN(params) for gp in gplist])
            Phi.params = sorted(set.union(*[set(gp.Phi.params) for gp in gplist]))

            multigp = VariableGP(NoiseMatrix1D_var(Phi), F)
        elif all(isinstance(gp.Phi, (NoiseMatrix1D_var, NoiseMatrix2D_var)) for gp in gplist):
            def Phi(params):
                # Phis = [Phifunc(params) for Phifunc in Phifuncs]
                # return jsp.linalg.block_diag(*Phis)
                return jsp.linalg.block_diag(*[jnp.diag(gp.Phi.getN(params)) if isinstance(gp.Phi, NoiseMatrix1D_var)
                                                                             else gp.Phi.getN(params)
                                               for gp in gplist])
            Phi.params = sorted(set.union(*[set(gp.Phi.params) for gp in gplist]))

            multigp = VariableGP(NoiseMatrix2D_var(Phi), F)
    else:
        raise NotImplementedError("Cannot concatenate these types of GPs.")

    if all(hasattr(gp, 'index') for gp in gplist):
        index, cnt = {}, 0
        for vars in zip(*[gp.index.items() for gp in gplist]):
            for var, sli in vars:
                width = sli.stop - sli.start
                index[var] = slice(cnt, cnt + width)
                cnt = cnt + width
        multigp.index = index

    return multigp

# sum delays

def CompoundDelay(delaylist):
    if len(delaylist) == 1:
        return delaylist[0]

    def delayfunc(params):
        return sum(delay(params) for delay in delaylist)
    delayfunc.params = sorted(set.union(*[set(delay.params) for delay in delaylist]))

    return delayfunc

# consider passing inv as a 1D object

class NoiseMatrix1D_novar(ConstantKernel):
    def __init__(self, N):
        self.N = N
        self.ld = np.sum(jnp.log(N))

        self.params = []

    def make_kernelproduct(self, y):
        product = -0.5 * jnp.sum(y**2 / self.N) - 0.5 * jnp.sum(jnp.log(self.N))

        def kernelproduct(params={}):
            return product
        kernelproduct.params = []

        return kernelproduct

    def inv(self):
        return np.diag(1.0 / self.N), self.ld

    def make_sqrt(self):
        sN = jnp.sqrt(self.N)

        def sqrt(params={}):
            return sN
        sqrt.params = []

        return sqrt

    def solve_1d(self, y):
        return y / self.N, self.ld

    def make_solve_1d(self):
        N, ld = jnparray(self.N), jnparray(self.ld)

        # closes on N, ld
        def solve_1d(y):
            return y / N, ld

        return solve_1d

    def solve_2d(self, T):
        return T / self.N[:, np.newaxis], self.ld

    def make_solve_2d(self):
        N, ld = jnparray(self.N[:, np.newaxis]), jnparray(self.ld)

        def solve_2d(T):
            return T / N, ld

        return solve_2d

    def make_sample(self):
        N12 = jnparray(np.sqrt(self.N))

        def sample(key):
            key, subkey = jnpsplit(key)
            return key, jnpnormal(subkey, N12.shape) * N12

        sample.params = []

        return sample


class NoiseMatrix1D_var(VariableKernel):
    def __init__(self, getN):
        self.getN = getN
        self.params = getN.params
        self.Phi_inv = None

    def make_kernelproduct(self, y):
        y2, getN = jnparray(y**2), self.getN

        # closes on y2, getN
        def kernelproduct(params):
            N = getN(params)
            return -0.5 * jnp.sum(y2 / N) - 0.5 * jnp.sum(jnp.log(N))

        kernelproduct.params = getN.params

        return kernelproduct

    def make_inv(self):
        getN = self.getN

        # closes on getN
        def inv(params):
            N = getN(params)
            return jnp.diag(1.0 / N), jnp.sum(jnp.log(N))
        inv.params = getN.params

        return inv

    def make_sqrt(self):
        getN = self.getN

        def sqrt(params):
            return jnp.sqrt(getN(params))
        sqrt.params = getN.params

        return sqrt

    def make_sample(self):
        getN = self.getN

        # closes on getN
        def sample(key, params):
            N12 = jnp.sqrt(getN(params))

            key, subkey = jnpsplit(key)
            return key, jnpnormal(subkey, N12.shape) * N12

        sample.params = getN.params

        return sample

    def solve_2d(self, params, F):
        inv = self.make_inv()
        myinv, ld = inv(params)
        return myinv @ F, ld

    def solve_1d(self, params, y):
        inv = self.make_inv()
        myinv, ld = inv(params)
        return myinv @ y, ld


class NoiseMatrix2D_novar(ConstantKernel):
    def __init__(self, N):
        self.N = N

        self.invN = jnp.linalg.inv(N)
        self.ld = jnp.linalg.slogdet(N)[1]

    def inv(self):
        return self.invN, self.ld


class NoiseMatrix2D_var(VariableKernel):
    def __init__(self, getN):
        self.getN = getN
        self.params = getN.params

    def make_inv(self):
        getN = self.getN

        # closes on getN
        def inv(params):
            N = getN(params)
            return jnp.linalg.inv(N), jnp.linalg.slogdet(N)[1]
        inv.params = getN.params

        return inv

    def make_sample(self):
        getN = self.getN

        def sample(key, params):
            N = getN(params)

            key, subkey = jnpsplit(key)
            return key, jnp.dot(jsp.linalg.cholesky(N, lower=True),
                                jnpnormal(subkey, (N.shape[0],)))

        sample.params = getN.params

        return sample


class VectorNoiseMatrix1D_var(VariableKernel):
    def __init__(self, getN):
        self.getN = getN
        self.params = getN.params

    def make_inv(self):
        getN = self.getN
        def inv(params):
            N = getN(params)

            return 1.0 / N, jnp.sum(jnp.log(N), axis=1)

            # n, m = N.shape
            # i1, i2 = jnp.diag_indices(m, ndim=2) # it's hard to vectorize numpy.diag!
            # return jnpzeros((n, m, m)).at[:,i1,i2].set(1.0 / N), jnp.sum(jnp.log(N), axis=1)
        inv.params = getN.params

        return inv


# now obsolete
# def ComponentKernel(N, F, P, c):
#     if isinstance(N, ConstantKernel) and isinstance(P, VariableKernel):
#         return ComponentKernel_varP(N, F, P, c)
#     else:
#         raise TypeError("N must be a ConstantKernel and P a VariableKernel")


def ShermanMorrisonKernel(N, F, P):
    if isinstance(N, ConstantKernel) and isinstance(P, ConstantKernel):
        return ShermanMorrisonKernel_novar(N, F, P)
    elif isinstance(N, ConstantKernel) and isinstance(P, VariableKernel):
        if not callable(F):
            return ShermanMorrisonKernel_varP(N, F, P)
        else:
            return ShermanMorrisonKernel_varFP(N, F, P)

    elif isinstance(N, VariableKernel) and isinstance(P, ConstantKernel):
        return ShermanMorrisonKernel_varN(N, F, P)
    elif isinstance(N, VariableKernel) and isinstance(P, VariableKernel):
        return ShermanMorrisonKernel_varNP(N, F, P)
    else:
        raise TypeError("N must be a ConstantKernel and P a VariableKernel")


# this is obsolete since the functionality was moved to logL_gpcomponent
# class ComponentKernel_varP:
#     def __init__(self, N, F, P_var, c):
#         self.N, self.F, self.P_var, self.c = N, F, P_var, c

#     def make_kernelproduct(self, y):
#         # -0.5 yt Nm y + yt Nm F a - 0.5 ct Ft Nm F c - 0.5 log |2 pi N| - 0.5 cT Pm c - 0.5 log |2 pi P|

#         NmF, ldN = self.N.solve_2d(self.F)
#         FtNmF = self.F.T @ NmF
#         NmFty = NmF.T @ y

#         Nmy, ldN = self.N.solve_1d(y)
#         ytNmy = y @ Nmy

#         ytNmy, NmFty, FtNmF = jnparray(ytNmy), jnparray(NmFty), jnparray(FtNmF)
#         P_inv = self.P_var.make_inv()
#         cfunc = self.c

#         def kernelproduct(params):
#             c = cfunc(params)
#             Pm, ldP = P_inv(params) # note Pm is 2D

#             return (-0.5 * ytNmy + c @ NmFty - 0.5 * c @ (FtNmF @ c)
#                     -0.5 * ldN - 0.5 * c @ (Pm @ c) - 0.5 * ldP)    # assuming Pm is 1D
#         kernelproduct.params = sorted(P_inv.params + cfunc.params)

#         return kernelproduct


class ShermanMorrisonKernel_novar(ConstantKernel):
    def __init__(self, N, F, P):
        # (N + F P F^T)^-1 = N^-1 - N^-1 F (P^-1 + F^T N^-1 F)^-1 F^T N^-1
        # |N + F P F^T| = |N| |P| |P^-1 + F^T N^-1 F|

        self.N, self.F, self.P = N, F, P
        self.NmF, ldN = N.solve_2d(F)
        FtNmF = F.T @ self.NmF
        Pinv, ldP = P.inv()
        self.cf = matrix_factor(Pinv + FtNmF)
        self.ld = ldN + ldP + matrix_norm * jnp.sum(jnp.log(np.diag(self.cf[0])))

        self.params = []

    def make_sample(self):
        N_sample = self.N.make_sample()
        P_sample = self.P.make_sample()
        F = jnparray(self.F)

        def sample(key):
            key, n = N_sample(key)
            key, c = P_sample(key)

            return key, n + jnp.dot(F, c)

        sample.params = []

        return sample

    def make_kernelproduct(self, y):
        Nmy = self.N.solve_1d(y)[0] - self.NmF @ matrix_solve(self.cf, self.NmF.T @ y)
        product = -0.5 * y @ Nmy - 0.5 * self.ld

        # closes on product
        def kernelproduct(params={}):
            return product
        kernelproduct.params = []

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

        sol = sp.linalg.cho_solve(self.cf, FtNmy)
        sol2 = sp.linalg.cho_solve(self.cf, FtNmT)

        a = -0.5 * (ytNmy - FtNmy.T @ sol) - 0.5 * self.ld
        b = jnparray(TtNmy - TtNmF @ sol)
        c = jnparray(TtNmT - TtNmF @ sol2)

        def kernelterms(params={}):
            return a, b, c
        kernelterms.params = []

        return kernelterms

    def make_kernelsolve(self, y, T):
        # Tt Sigma y = Tt (N + F P Ft) y
        # Tt Sigma^-1 y = Tt (Nm - Nm F (P^-1 + Ft Nm F)^-1 Ft Nm) y
        #               = Tt Nm y - Tt Nm F (P^-1 + Ft Nm F)^-1 Ft Nm y
        # Tt Sigma^-1 T = Tt Nm T - Tt Nm F (P^-1 + Ft Nm F)^-1 Ft Nm T

        Nmy, _ = self.N.solve_1d(y)
        FtNmy  = self.F.T @ Nmy

        NmF, _ = self.N.solve_2d(self.F)
        FtNmF = self.F.T @ NmF

        if callable(T):
            Nmy, Nmf = jnparray(Nmy), jnparray(NmF)
            N_solve_2d = self.N.make_solve_2d()
            cf = (jnparray(self.cf[0]), self.cf[1])
            F, FtNmy, FtNmF = jnparray(self.F), jnparray(FtNmy), jnparray(FtNmF)

            def kernelsolve(params):
                Tmat = T(params)

                TtNmy  = Tmat.T @ Nmy
                TtNmF  = Tmat.T @ NmF

                NmT, _ = N_solve_2d(Tmat)
                FtNmT  = F.T @ NmT
                TtNmT  = Tmat.T @ NmT

                TtSy = TtNmy - TtNmF @ jsp.linalg.cho_solve(cf, FtNmy)
                TtST = TtNmT - TtNmF @ jsp.linalg.cho_solve(cf, FtNmT)

                return TtSy, TtST

            kernelsolve.params = T.params
        else:
            TtNmy  = T.T @ Nmy
            TtNmF  = T.T @ NmF

            NmT, _ = self.N.solve_2d(T)
            FtNmT  = self.F.T @ NmT
            TtNmT  = T.T @ NmT

            TtSy = jnparray(TtNmy - TtNmF @ sp.linalg.cho_solve(self.cf, FtNmy))
            TtST = jnparray(TtNmT - TtNmF @ sp.linalg.cho_solve(self.cf, FtNmT))

            # closes on TtSy and TtST
            def kernelsolve(params={}):
                return TtSy, TtST

            kernelsolve.params = []

        return kernelsolve

    def solve_1d(self, y):
        return self.N.solve_1d(y)[0] - self.NmF @ sp.linalg.cho_solve(self.cf, self.NmF.T @ y), self.ld

    def make_solve_1d(self):
        N_solve_1d = self.N.make_solve_1d()
        NmF = jnparray(self.NmF)
        cf = (jnparray(self.cf[0]), self.cf[1])
        ld = jnp.array(self.ld)

        # closes on N_solve_1d, NmF, cf, ld
        def solve_1d(y):
            return N_solve_1d(y)[0] - NmF @ jsp.linalg.cho_solve(cf, NmF.T @ y), ld

        return solve_1d

    def solve_2d(self, y):
        return self.N.solve_2d(y)[0] - self.NmF @ sp.linalg.cho_solve(self.cf, self.NmF.T @ y), self.ld

    def make_solve_2d(self):
        N_solve_2d = self.N.make_solve_2d()
        NmF = jnparray(self.NmF)
        cf = (jnparray(self.cf[0]), self.cf[1])
        ld = jnp.array(self.ld)

        def solve_2d(F):
            return N_solve_2d(F)[0] - NmF @ jsp.linalg.cho_solve(cf, NmF.T @ F), self.ld

        return solve_2d



# this is the terminal class for a standard model 2A; the only parameters come in through P_var
# it defines make_kernelproduct() which is effectively the model 2A likelihood for fixed y
#        and make_kernelterms() which is used in the multi-pulsar likelihood
# it could also handle variable y via a make_kernel(self) method
# or a separate ShermanMorrisonKernel_vary_varP class

# now the make_* methods return a function that closes on jax arrays (either CPU or GPU)
# and also (in this case) on self.P_var.inv. If P_var is NoiseMatrix1D_var, then inv will
# call N = getN(params), which is supposed to return a jax array. But getN could be,
# for instance, priorfunc, as returned by makegp_fourier, which needs to be jaxed.

# in short, it may be hard to separate np preparations and jax cached variables
# unless anything that is constant is numpy, and anything that will take parameters
# is either jax from the start

class ShermanMorrisonKernel_varFP(VariableKernel):
    def __init__(self, N, F_var, P_var):
        self.N, self.F, self.P_var = N, F_var, P_var

    def make_kernelproduct(self, y):
        Nmy, _  = self.N.solve_1d(y)
        ytNmy = y @ Nmy

        ytNmy = jnparray(ytNmy)
        F_var, N_solve_2d = self.F, self.N.make_solve_2d()
        P_var_inv = self.P_var.make_inv()

        def kernelproduct(params):
            F = F_var(params)
            NmF, ldN = N_solve_2d(F)
            FtNmF = F.T @ NmF
            NmFty = NmF.T @ y

            Pinv, ldP = P_var_inv(params)
            cf = matrix_factor(Pinv + FtNmF)
            ytXy = NmFty.T @ matrix_solve(cf, NmFty)

            return -0.5 * (ytNmy - ytXy) - 0.5 * (ldN + ldP + matrix_norm * jnp.sum(jnp.log(jnp.diag(cf[0]))))

        kernelproduct.params = F_var.params + P_var_inv.params

        return kernelproduct

class ShermanMorrisonKernel_varNP(VariableKernel):
    def __init__(self, N_var, F, P_var):
        self.N_var, self.F, self.P_var = N_var, F, P_var

    def make_kernelproduct(self, y):

        F = self.F
        P_var_inv = self.P_var.make_inv()
        # N_var_inv = self.N_var.make_inv()
        N_var = self.N_var

        def kernelproduct(params):
            # Ninv, ldN = N_var_inv.inv(params)
            N_solve_1d = N_var.make_solve_1d()
            Nmy, ldN = N_solve_1d(params, y)
            ytNmy = y @ Nmy



            # ytNmy = jnparray(ytNmy)
            # NmF, ldN = N_var.solve_2d(params, F)

            N_solve_2d = N_var.make_solve_2d()
            NmF, ldN = N_solve_2d(params, F)

            FtNmF = F.T @ NmF
            NmFty = NmF.T @ y

            Pinv, ldP = P_var_inv(params)
            cf = matrix_factor(Pinv + FtNmF)
            ytXy = NmFty.T @ matrix_solve(cf, NmFty)
            return -0.5 * (ytNmy - ytXy) - 0.5 * (ldN + ldP + matrix_norm * jnp.sum(jnp.log(jnp.diag(cf[0]))))
        kernelproduct.params = N_var.params + P_var_inv.params

        return kernelproduct

    # probably correct, but not needed yet
    #
    # def make_kernelsolve(self, y, T):
    #     Nmy, _ = self.N.solve_1d(y) if y.ndim == 1 else self.N.solve_2d(y)
    #     TtNmy  = T.T @ Nmy

    #     NmT, _ = self.N.solve_2d(T)
    #     TtNmT  = T.T @ NmT

    #     TtNmy, TtNmT = jnparray(TtNmy), jnparray(TtNmT)
    #     F_var, N_solve_2d = self.F_var, self.N.make_solve_2d()
    #     P_var_inv = self.P_var.make_inv()

    #     def kernelsolve(params):
    #         F = F_var(params)
    #         FtNmy  = F.T @ Nmy
    #         FtNmT  = F.T @ NmT

    #         NmF, _ = N_solve_2d(F)
    #         FtNmF = F.T @ NmF
    #         TtNmF = T.T @ NmF

    #         Pinv, _ = P_var_inv(params)
    #         cf = jsp.linalg.cho_factor(Pinv + FtNmF)

    #         TtSy = TtNmy - TtNmF @ jsp.linalg.cho_solve(cf, FtNmy)
    #         TtST = TtNmT - TtNmF @ jsp.linalg.cho_solve(cf, FtNmT)

    #         return TtSy, TtST

    #     kernelsolve.params = F_var.params + P_var_inv.params

    #     return kernelsolve

class ShermanMorrisonKernel_varP(VariableKernel):
    def __init__(self, N, F, P_var):
        self.N, self.F, self.P_var = N, F, P_var

    def make_sample(self):
        N_sample = self.N.make_sample()
        P_sample = self.P_var.make_sample()
        F = jnparray(self.F)

        def sample(key, params):
            key, n = N_sample(key)
            key, c = P_sample(key, params)

            return key, n + jnp.dot(F, c)

        sample.params = P_sample.params

        return sample

    def make_kernel(self, y, delay):
        NmF, ldN = self.N.solve_2d(self.F)
        FtNmF = self.F.T @ NmF

        y = jnparray(y)
        NmF, FtNmF, ldN = jnparray(NmF), jnparray(FtNmF), jnparray(ldN)
        P_var_inv = self.P_var.make_inv()
        N_solve_1d = self.N.make_solve_1d()

        # closes on y, delay, P_var_inv, NmF, FtNmF, ldN
        def kernel(params):
            yp = y - delay(params)

            Nmy, _ = N_solve_1d(yp)
            ytNmy = yp @ Nmy
            NmFty = NmF.T @ yp

            Pinv, ldP = P_var_inv(params)
            cf = matrix_factor(Pinv + FtNmF)
            ytXy = NmFty.T @ matrix_solve(cf, NmFty)

            return -0.5 * (ytNmy - ytXy) - 0.5 * (ldN + ldP + matrix_norm * jnp.sum(jnp.log(jnp.diag(cf[0]))))

        kernel.params = delay.params + P_var_inv.params

        return kernel

    # makes a function that returns the tuple
    # - Tt (Phi + Ft Nm F)^-1 y
    # - Tt (Phi + Ft Nm F)^-1 T
    # with fixed y and T

    def make_kernelsolve(self, y, T):
        # Tt Sigma y = Tt (N + F P Ft) y
        # Tt Sigma^-1 y = Tt (Nm - Nm F (P^-1 + Ft Nm F)^-1 Ft Nm) y
        #               = Tt Nm y - Tt Nm F (P^-1 + Ft Nm F)^-1 Ft Nm y
        # Tt Sigma^-1 T = Tt Nm T - Tt Nm F (P^-1 + Ft Nm F)^-1 Ft Nm T

        Nmy, _ = self.N.solve_1d(y) if y.ndim == 1 else self.N.solve_2d(y)
        FtNmy  = self.F.T @ Nmy
        TtNmy  = T.T @ Nmy

        NmT, _ = self.N.solve_2d(T)
        FtNmT  = self.F.T @ NmT
        TtNmT  = T.T @ NmT

        NmF, _ = self.N.solve_2d(self.F)
        FtNmF = self.F.T @ NmF
        TtNmF = T.T @ NmF

        FtNmF, TtNmy = jnparray(FtNmF), jnparray(TtNmy)
        FtNmT, TtNmT = jnparray(FtNmT), jnparray(TtNmT)
        TtNmF, FtNmy = jnparray(TtNmF), jnparray(FtNmy)
        P_var_inv = self.P_var.make_inv()

        def kernelsolve(params):
            Pinv, _ = P_var_inv(params)
            cf = matrix_factor(Pinv + FtNmF)

            TtSy = TtNmy - TtNmF @ matrix_solve(cf, FtNmy)
            TtST = TtNmT - TtNmF @ matrix_solve(cf, FtNmT)

            return TtSy, TtST

        kernelsolve.params = P_var_inv.params

        return kernelsolve

    def make_kernelproduct(self, y):
        NmF, ldN = self.N.solve_2d(self.F)
        FtNmF = self.F.T @ NmF

        Nmy, _  = self.N.solve_1d(y)
        ytNmy = y @ Nmy
        NmFty = NmF.T @ y

        FtNmF, NmFty, ytNmy, ldN = jnparray(FtNmF), jnparray(NmFty), jnparray(ytNmy), jnparray(ldN)
        P_var_inv = self.P_var.make_inv()

        # closes on P_var_inv, FtNmF, NmFty, ytNmy, ldN
        def kernelproduct(params):
            Pinv, ldP = P_var_inv(params)
            cf = matrix_factor(Pinv + FtNmF)
            ytXy = NmFty.T @ matrix_solve(cf, NmFty)

            return -0.5 * (ytNmy - ytXy) - 0.5 * (ldN + ldP + matrix_norm * jnp.sum(jnp.log(jnp.diag(cf[0]))))

        kernelproduct.params = P_var_inv.params

        return kernelproduct

    def make_kernelproduct_gpcomponent(self, y):
        # -0.5 yt Nm y + yt Nm F a - 0.5 ct Ft Nm F c - 0.5 log |2 pi N| - 0.5 cT Pm c - 0.5 log |2 pi P|

        NmF, ldN = self.N.solve_2d(self.F)
        FtNmF = self.F.T @ NmF
        NmFty = NmF.T @ y

        Nmy, ldN = self.N.solve_1d(y)
        ytNmy = y @ Nmy

        ytNmy, NmFty, FtNmF = jnparray(ytNmy), jnparray(NmFty), jnparray(FtNmF)
        P_inv = self.P_var.make_inv()

        cvars = list(self.index.keys())

        def kernelproduct(params):
            c = jnp.concatenate([params[cvar] for cvar in cvars])
            Pm, ldP = P_inv(params)

            return (-0.5 * ytNmy + c @ NmFty - 0.5 * c @ (FtNmF @ c)
                    -0.5 * ldN - 0.5 * c @ (Pm @ c) - 0.5 * ldP)    # note Pm is 2D
        kernelproduct.params = sorted(P_inv.params + cvars)

        return kernelproduct

    # makes a function that returns the tuple
    # * -0.5 (yt Nm y - yt Nm F (Phi + Ft Nm F)^-1 y - 0.5 (logdet N + logdet Phi + logdet S)
    # * Tt (Phi + Ft Nm F)^-1 y
    # * Tt (Phi + Ft Nm F)^-1 T
    # with fixed y and T

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

        P_var_inv = self.P_var.make_inv()

        FtNmF, FtNmy, FtNmT, ytNmy = jnparray(FtNmF), jnparray(FtNmy), jnparray(FtNmT), jnparray(ytNmy)
        TtNmy, TtNmT, TtNmF = jnparray(TtNmy), jnparray(TtNmT), jnparray(TtNmF)
        # closes on P_var_inv, FtNmF, FtNmy, FtMmT, ytNmy, TtNmy, TtNmT, TtNmF
        def kernelterms(params):
            Pinv, ldP = P_var_inv(params)
            cf = matrix_factor(Pinv + FtNmF)

            sol = matrix_solve(cf, FtNmy)
            sol2 = matrix_solve(cf, FtNmT)

            a = -0.5 * (ytNmy - FtNmy.T @ sol) - 0.5 * (ldN + ldP + matrix_norm * jnp.sum(jnp.log(jnp.diag(cf[0]))))
            b = TtNmy - TtNmF @ sol
            c = TtNmT - TtNmF @ sol2

            return a, b, c

        kernelterms.params = self.P_var.params

        return kernelterms


class ShermanMorrisonKernel_varN(VariableKernel):
    def __init__(self, N_var, F, P):
        # (N + F P F^T)^-1 = N^-1 - N^-1 F (P^-1 + F^T N^-1 F)^-1 F^T N^-1
        # |N + F P F^T| = |N| |P| |P^-1 + F^T N^-1 F|

        self.N_var, self.F, self.P = N_var, F, P
        # self.NmF, ldN = N_var.solve_2d(F)
        # FtNmF = F.T @ self.NmF
        Pinv, ldP = P.inv()
        # self.cf = matrix_factor(Pinv + FtNmF)
        # self.ld = ldN + ldP + matrix_norm * np.sum(np.log(np.diag(self.cf[0])))

        self.params = self.N_var.params

    def make_sample(self, params):
        P_sample = self.P.make_sample()
        F = jnparray(self.F)
        N_sample = self.N_var.make_sample()
        def sample(key):
            # N = N_var(params)
            key, n = N_sample(key, params)
            key, c = P_sample(key)

            return key, n + jnp.dot(F, c)

        sample.params = N_sample.params

        return sample

    def make_kernelproduct(self, y):

        Pinv, ldP = self.P.inv()
        F = self.F
        N_var = self.N_var
        # closes on product
        def kernelproduct(params={}):
            # get N
            NmF, ldN = N_var.solve_2d(params, self.F)
            FtNmF = F.T @ NmF
            cf = matrix_factor(Pinv + FtNmF)
            Nmy = N_var.solve_1d(params, y)[0] - NmF @ matrix_solve(cf, NmF.T @ y)
            ld = ldN + ldP + matrix_norm * np.sum(jnp.log(jnp.diag(cf[0])))
            product = -0.5 * y @ Nmy - 0.5 * ld
            return product
        kernelproduct.params = N_var.params

        return kernelproduct

    def make_kernelterms(self, y, T):
        # Sigma = (N + F P Ft)
        # Sigma^-1 = Nm - Nm F (P^-1 + Ft Nm F)^-1 Ft Nm
        #
        # yt Sigma^-1 y = yt Nm y - (yt Nm F) C^-1 (Ft Nm y)
        # Tt Sigma^-1 y = Tt Nm y - Tt Nm F C^-1 (Ft Nm y)
        # Tt Sigma^-1 T = Tt Nm T - (Tt Nm F) C^-1 (Ft Nm T)

        # Nmy, ldN = self.N.solve_1d(y)
        # ytNmy = y @ Nmy
        # FtNmy = self.F.T @ Nmy
        # TtNmy = T.T @ Nmy

        # NmF, _ = self.N.solve_2d(self.F)
        # FtNmF = self.F.T @ NmF
        # TtNmF = T.T @ NmF

        # NmT, _ = self.N.solve_2d(T)
        # FtNmT = self.F.T @ NmT
        # TtNmT = T.T @ NmT

        # sol = sp.linalg.cho_solve(self.cf, FtNmy)
        # sol2 = sp.linalg.cho_solve(self.cf, FtNmT)

        # a = -0.5 * (ytNmy - FtNmy.T @ sol) - 0.5 * self.ld
        # b = jnparray(TtNmy - TtNmF @ sol)
        # c = jnparray(TtNmT - TtNmF @ sol2)
        N_var = self.N_var
        F = self.F
        Pinv, ldP = self.P.inv()
        def kernelterms(params={}):
            Nmy, ldN = N_var.solve_1d(params, y)
            ytNmy = y @ Nmy
            FtNmy = F.T @ Nmy
            TtNmy = T.T @ Nmy

            NmF, _ = N_var.solve_2d(params, F)
            FtNmF = F.T @ NmF
            TtNmF = T.T @ NmF

            NmT, _ = N_var.solve_2d(params, T)
            FtNmT = F.T @ NmT
            TtNmT = T.T @ NmT
            cf = matrix_factor(Pinv + FtNmF)
            cf = (jnparray(cf[0]), cf[1])
            sol = sp.linalg.cho_solve(cf, FtNmy)
            sol2 = sp.linalg.cho_solve(cf, FtNmT)
            ld = ldN + ldP + matrix_norm * np.sum(jnp.log(jnp.diag(cf[0])))
            a = -0.5 * (ytNmy - FtNmy.T @ sol) - 0.5 * ld
            b = jnparray(TtNmy - TtNmF @ sol)
            c = jnparray(TtNmT - TtNmF @ sol2)
            return a, b, c
        kernelterms.params = N_var.params

        return kernelterms

    def make_kernelsolve(self, y, T):
        # Tt Sigma y = Tt (N + F P Ft) y
        # Tt Sigma^-1 y = Tt (Nm - Nm F (P^-1 + Ft Nm F)^-1 Ft Nm) y
        #               = Tt Nm y - Tt Nm F (P^-1 + Ft Nm F)^-1 Ft Nm y
        # Tt Sigma^-1 T = Tt Nm T - Tt Nm F (P^-1 + Ft Nm F)^-1 Ft Nm T
        N_var = self.N_var
        F = self.F
        # Nmy, _ = self.N.solve_1d(y)
        # FtNmy  = self.F.T @ Nmy

        # NmF, _ = self.N.solve_2d(self.F)
        # FtNmF = self.F.T @ NmF
        Pinv, ldP = self.P.inv()
        if callable(T):
            # Nmy, NmF = jnparray(Nmy), jnparray(NmF)
            # N_solve_2d = self.N.make_solve_2d()
            # cf = (jnparray(self.cf[0]), self.cf[1])
            # F, FtNmy, FtNmF = jnparray(self.F), jnparray(FtNmy), jnparray(FtNmF)
            F = jnparray(self.F)

            def kernelsolve(params):
                Nmy, ldN = N_var.solve_1d(params, y)
                FtNmy = F.T @ Nmy
                NmF, _ = N_var.solve_2d(params, F)
                FtNmF = F.T @ NmF
                cf = matrix_factor(Pinv + FtNmF)
                cf = (jnparray(cf[0]), cf[1])
                Tmat = T(params)

                TtNmy  = Tmat.T @ Nmy
                TtNmF  = Tmat.T @ NmF

                NmT, _ = N_var.solve_2d(params, Tmat)
                FtNmT  = F.T @ NmT
                TtNmT  = Tmat.T @ NmT

                TtSy = TtNmy - TtNmF @ jsp.linalg.cho_solve(cf, FtNmy)
                TtST = TtNmT - TtNmF @ jsp.linalg.cho_solve(cf, FtNmT)

                return TtSy, TtST

            kernelsolve.params = N_var.params + T.params
        else:


            # closes on TtSy and TtST
            def kernelsolve(params={}):
                Nmy, ldN = N_var.solve_1d(params, y)
                FtNmy = F.T @ Nmy
                NmF, _ = N_var.solve_2d(params, F)
                TtNmy  = T.T @ Nmy
                TtNmF  = T.T @ NmF

                NmT, _ = N_var.solve_2d(params, T)
                FtNmT  = self.F.T @ NmT
                TtNmT  = T.T @ NmT

                TtSy = jnparray(TtNmy - TtNmF @ sp.linalg.cho_solve(self.cf, FtNmy))
                TtST = jnparray(TtNmT - TtNmF @ sp.linalg.cho_solve(self.cf, FtNmT))
                return TtSy, TtST

            kernelsolve.params = N_var.params

        return kernelsolve

    def solve_1d(self, params, y):
        N_var_inv = self.N_var.make_inv()
        Ninv, ldN = N_var_inv(params)
        NmF = Ninv @ self.F
        NmF = NmF
        NmFty = NmF.T @ y
        Pinv, ldP = self.P.inv()
        cf = matrix_factor(Pinv + self.F.T @ NmF)
        # cf = (jnparray(cf[0]), cf[1])
        ld = ldN + ldP + matrix_norm * np.sum(jnp.log(jnp.diag(cf[0])))
        return Ninv @ y - NmF @ sp.linalg.cho_solve(cf, NmFty), ld

    def make_solve_1d(self):


        N_var_inv = self.N_var.make_inv()
        F = self.F
        P = self.P
        # closes on N_solve_1d, NmF, cf, ld
        def solve_1d(params, y):
            Ninv, ldN = N_var_inv(params)
            NmF = Ninv @ self.F
            Pinv, ldP = P.inv()
            cf = matrix_factor(Pinv + F.T @ NmF)
            print('==hi==')
            print(cf[1])
            print(np.sum(np.log(np.diag(cf[0]))))
            ld = ldN + ldP + matrix_norm * np.sum(jnp.log(jnp.diag(cf[0])))

            # ld = ldN + ldP + matrix_norm * jnp.linalg.slogdet(cf[0])[1]
            NmF = jnparray(NmF)
            NmFty = jnparray(NmF.T @ y)
            cf = (jnparray(cf[0]), cf[1])
            ld = jnp.array(ld)
            return Ninv @ y - NmF @ jsp.linalg.cho_solve(cf, NmFty), ld

        return solve_1d

    def solve_2d(self, params, Fr):
        N_var_inv = self.N_var.make_inv()
        Ninv, ldN = N_var_inv(params)
        NmFl = Ninv @ self.F
        NmFl = jnparray(NmFl)
        Pinv, ldP = self.P.inv()
        cf = matrix_factor(Pinv + self.F.T @ NmFl)
        cf = (jnparray(cf[0]), cf[1])
        NmFltFr = jnparray(NmFl.T @ Fr)
        ld = ldN + ldP + matrix_norm * np.sum(jnp.log(jnp.diag(cf[0])))
        # ld = ldN + ldP + matrix_norm * jnp.linalg.slogdet(cf[0])[1]
        return Ninv @ Fr - NmFl @ sp.linalg.cho_solve(cf, NmFltFr), ld

    def make_solve_2d(self):
        N_var_inv = self.N_var.make_inv()
        Pinv, ldP = self.P.inv()
        Fl = self.F
        def solve_2d(params, Fr):
            Ninv, ldN = N_var_inv(params)
            NmFl = Ninv @ Fl
            cf = matrix_factor(Pinv + Fl.T @ NmFl)
            NmFl = jnparray(NmFl)
            NmFltFr = jnparray(NmFl.T @ Fr)
            cf = (jnparray(cf[0]), cf[1])
            ld = ldN + ldP + matrix_norm * jnp.sum(jnp.log(jnp.diag(cf[0])))
            # ld = ldN + ldP + matrix_norm * jnp.linalg.slogdet(cf[0])[1]
            ld = jnp.array(ld)

            return Ninv @ Fr - NmFl @ jsp.linalg.cho_solve(cf, NmFltFr), ld
        return solve_2d


class VectorShermanMorrisonKernel_varP(VariableKernel):
    def __init__(self, Ns, Fs, P_var):
        self.Ns, self.Fs, self.P_var = Ns, Fs, P_var

    def make_kernelproduct(self, ys):
        NmFs, ldNs = zip(*[N.solve_2d(F) for N, F in zip(self.Ns, self.Fs)])
        FtNmFs = [F.T @ NmF for F, NmF in zip(self.Fs, NmFs)]

        Nmys, _  = zip(*[N.solve_1d(y) for N, y in zip(self.Ns, ys)])
        ytNmys = [y @ Nmy for y, Nmy in zip(ys, Nmys)]
        NmFtys = [NmF.T @ y for NmF, y in zip(NmFs, ys)]

        P_var_inv = self.P_var.make_inv()
        FtNmF, NmFty = jnparray(FtNmFs), jnparray(NmFtys)
        ytNmy, ldN = float(sum(ytNmys)), float(sum(ldNs))

        def kernelproduct(params):
            Pinv, ldP = P_var_inv(params)            # Pinv.shape = FtNmF.shape = [npsr, ngp, ngp]

            i1, i2 = jnp.diag_indices(Pinv.shape[1], ndim=2)
            cf = matrix_factor(FtNmF.at[:,i1,i2].add(Pinv))

            # cf = jsp.linalg.cho_factor(Pinv + FtNmF)

            ytXy = jnp.sum(NmFty * matrix_solve(cf, NmFty)) # was NmFty.T @ jsp.linalg.cho_solve(...)

            i1, i2 = jnp.diag_indices(cf[0].shape[1], ndim=2) # it's hard to vectorize numpy.diag!
            return -0.5 * (ytNmy - ytXy) - 0.5 * (ldN + jnp.sum(ldP) + matrix_norm * jnp.sum(jnp.log(cf[0][:,i1,i2])))

        kernelproduct.params = P_var_inv.params

        return kernelproduct

    def make_kernelproduct_gpcomponent(self, ys):
        # -0.5 yt Nm y + yt Nm F a - 0.5 ct Ft Nm F c - 0.5 log |2 pi N| - 0.5 cT Pm c - 0.5 log |2 pi P|

        NmFs, ldNs = zip(*[N.solve_2d(F) for N, F in zip(self.Ns, self.Fs)])
        FtNmFs = [F.T @ NmF for F, NmF in zip(self.Fs, NmFs)]

        Nmys, _  = zip(*[N.solve_1d(y) for N, y in zip(self.Ns, ys)])
        ytNmys = [y @ Nmy for y, Nmy in zip(ys, Nmys)]
        NmFtys = [NmF.T @ y for NmF, y in zip(NmFs, ys)]

        FtNmF, NmFty = jnparray(FtNmFs), jnparray(NmFtys)
        ytNmy, ldN = float(sum(ytNmys)), float(sum(ldNs))
        cvarsall = self.index if isinstance(self.index, list) else [self.index]

        if hasattr(self, 'prior'):
            P_var_prior = self.prior

            def kernelproduct(params):
                c = jnp.array([jnp.concatenate([params[cvar] for cvar in cvars]) for cvars in cvarsall])
                logpr = P_var_prior(params)

                return (-0.5 * ytNmy + jnp.sum(c * NmFty) - 0.5 * jnp.einsum('ij,ijk,ik', c, FtNmF, c)
                        -0.5 * ldN - logpr) # note Pm is 1D

            kernelproduct.params = sorted(P_var_prior.params + sum([list(cvars) for cvars in cvarsall], []))
        else:
            P_var_inv = self.P_var.make_inv()

            def kernelproduct(params):
                c = jnp.array([jnp.concatenate([params[cvar] for cvar in cvars]) for cvars in cvarsall])
                Pm, ldP = P_var_inv(params)

                return (-0.5 * ytNmy + jnp.sum(c * NmFty) - 0.5 * jnp.einsum('ij,ijk,ik', c, FtNmF, c)
                        -0.5 * ldN - 0.5 * jnp.sum(c * Pm * c) - 0.5 * jnp.sum(ldP)) # note Pm is 1D

            kernelproduct.params = sorted(P_var_inv.params + sum([list(cvars) for cvars in cvarsall], []))

        return kernelproduct

    def make_kernelterms(self, ys, Ts):
        # Sigma = (N + F P Ft)
        # Sigma^-1 = Nm - Nm F (P^-1 + Ft Nm F)^-1 Ft Nm
        #
        # yt Sigma^-1 y = yt Nm y - (yt Nm F) C^-1 (Ft Nm y)
        # Tt Sigma^-1 y = Tt Nm y - Tt Nm F C^-1 (Ft Nm y)
        # Tt Sigma^-1 T = Tt Nm T - (Tt Nm F) C^-1 (Ft Nm T)

        Nmys, ldNs = zip(*[N.solve_1d(y) for N, y in zip(self.Ns, ys)])
        ytNmys = [y @ Nmy for y, Nmy in zip(ys, Nmys)]
        FtNmys = [F.T @ Nmy for F, Nmy in zip(self.Fs, Nmys)]
        TtNmys = [T.T @ Nmy for T, Nmy in zip(Ts, Nmys)]

        NmFs, _ = zip(*[N.solve_2d(F) for N, F in zip(self.Ns, self.Fs)])
        FtNmFs = [F.T @ NmF for F, NmF in zip(self.Fs, NmFs)]
        TtNmFs = [T.T @ NmF for T, NmF in zip(Ts, NmFs)]

        NmTs, _ = zip(*[N.solve_2d(T) for N, T in zip(self.Ns, Ts)])
        FtNmTs = [F.T @ NmT for F, NmT in zip(self.Fs, NmTs)]
        TtNmTs = [T.T @ NmT for T, NmT in zip(Ts, NmTs)]

        P_var_inv = self.P_var.make_inv()

        FtNmF, FtNmy, FtNmT = jnparray(FtNmFs), jnparray(FtNmys), jnparray(FtNmTs)
        ldN, ytNmy = float(sum(ldNs)), float(sum(ytNmys))
        TtNmy, TtNmT, TtNmF = jnparray(TtNmys), jnparray(TtNmTs), jnparray(TtNmFs)
        def kernelterms(params):
            Pinv, ldP = P_var_inv(params)

            i1, i2 = jnp.diag_indices(Pinv.shape[1], ndim=2)
            cf = matrix_factor(FtNmF.at[:,i1,i2].add(Pinv))

            sol = matrix_solve(cf, FtNmy)
            sol2 = matrix_solve(cf, FtNmT)

            i1, i2 = jnp.diag_indices(cf[0].shape[1], ndim=2)
            a = -0.5 * (ytNmy - jnp.sum(FtNmy * sol)) - 0.5 * (ldN + jnp.sum(ldP) + matrix_norm * jnp.sum(jnp.log(cf[0][:,i1,i2])))
            b = TtNmy - jnp.sum(TtNmF * sol[:, jnp.newaxis, :], axis=2)
            c = TtNmT - TtNmF @ sol2 # fine as is!

            return a, b, c

        kernelterms.params = self.P_var.params

        return kernelterms