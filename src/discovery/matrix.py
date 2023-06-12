import numpy as np
import scipy as sp

import jax
import jax.numpy
import jax.scipy

def config(backend):
    global jnp, jsp, jnparray

    if backend == 'numpy':
        jnp, jsp = np, sp
        jnparray = lambda a: np.array(a, dtype=np.float64)
    elif backend == 'jax':
        jnp, jsp = jax.numpy, jax.scipy
        jnparray = lambda a: jnp.array(a, dtype=jnp.float64)

config('jax')

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

class GlobalVariableGP:
    def __init__(self, Phi, Fs):
        self.Phi, self.Fs = Phi, Fs

# concatenate GPs
def CompoundGP(gplist):
    if len(gplist) == 1:
        return gplist[0]

    if all(isinstance(gp, ConstantGP) for gp in gplist) and all(isinstance(gp.Phi, NoiseMatrix1D_novar) for gp in gplist):
        F = np.hstack([gp.F for gp in gplist])
        PhiN = np.concatenate([gp.Phi.N for gp in gplist])

        return ConstantGP(NoiseMatrix1D_novar(PhiN), F)
    elif all(isinstance(gp, VariableGP) for gp in gplist) and all(isinstance(gp.Phi, NoiseMatrix1D_var) for gp in gplist):
        F = np.hstack([gp.F for gp in gplist])

        def Phi(params):
            return jnp.concatenate([gp.Phi.getN(params) for gp in gplist])
        Phi.params = sorted(set.union(*[set(gp.Phi.params) for gp in gplist]))

        return VariableGP(NoiseMatrix1D_var(Phi), F)
    else:
        raise NotImplementedError("Cannot concatenate these types of GPs.")

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
        self.ld = np.sum(np.log(N))

        self.params = []

    def make_kernelproduct(self, y):
        product = -0.5 * jnp.sum(y**2 / self.N) - 0.5 * jnp.sum(jnp.log(self.N))

        def kernelproduct(params={}):
            return product

        kernelproduct.params = []

        return kernelproduct

    def inv(self):
        return np.diag(1.0 / self.N), self.ld

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


class NoiseMatrix1D_var(VariableKernel):
    def __init__(self, getN):
        self.getN = getN
        self.params = getN.params

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


def ShermanMorrisonKernel(N, F, P):
    if not isinstance(F, np.ndarray):
        raise TypeError("F must be a numpy array.")

    if isinstance(N, ConstantKernel) and isinstance(P, ConstantKernel):
        return ShermanMorrisonKernel_novar(N, F, P)
    elif isinstance(N, ConstantKernel) and isinstance(P, VariableKernel):
        return ShermanMorrisonKernel_varP(N, F, P)
    else:
        raise TypeError("N must be a ConstantMatrix and P a VariableMatrix")

# the first argument of ShermanMorrisonKernel will be

# VariableKernel should have methods
# - make_kernelproduct
# - make_kernelterms

class ShermanMorrisonKernel_novar(ConstantKernel):
    def __init__(self, N, F, P):
        # (N + F P F^T)^-1 = N^-1 - N^-1 F (P^-1 + F^T N^-1 F)^-1 F^T N^-1
        # |N + F P F^T| = |N| |P| |P^-1 + F^T N^-1 F|

        self.N = N
        self.NmF, ldN = N.solve_2d(F)
        FtNmF = F.T @ self.NmF
        Pinv, ldP = P.inv()
        self.cf = sp.linalg.cho_factor(Pinv + FtNmF)
        self.ld = ldN + ldP + 2.0 * np.sum(np.log(np.diag(self.cf[0])))

        self.params = []

    def make_kernelproduct(self, y):
        Nmy = self.N.solve_1d(y)[0] - self.NmF @ jsp.linalg.cho_solve(self.cf, self.NmF.T @ y)
        product = -0.5 * y @ Nmy - 0.5 * self.ld

        # closes on product
        def kernelproduct(params={}):
            return product

        kernelproduct.params = []

        return kernelproduct

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

class ShermanMorrisonKernel_varP(VariableKernel):
    def __init__(self, N, F, P_var):
        self.N, self.F, self.P_var = N, F, P_var

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
            cf = jsp.linalg.cho_factor(Pinv + FtNmF)
            ytXy = NmFty.T @ jsp.linalg.cho_solve(cf, NmFty)

            return -0.5 * (ytNmy - ytXy) - 0.5 * (ldN + ldP + 2.0 * jnp.sum(jnp.log(jnp.diag(cf[0]))))

        kernel.params = delay.params + P_var_inv.params

        return kernel

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
            cf = jsp.linalg.cho_factor(Pinv + FtNmF)
            ytXy = NmFty.T @ jsp.linalg.cho_solve(cf, NmFty)

            return -0.5 * (ytNmy - ytXy) - 0.5 * (ldN + ldP + 2.0 * jnp.sum(jnp.log(jnp.diag(cf[0]))))

        kernelproduct.params = P_var_inv.params

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

        P_var_inv = self.P_var.make_inv()

        FtNmF, FtNmy, FtNmT, ytNmy = jnparray(FtNmF), jnparray(FtNmy), jnparray(FtNmT), jnparray(ytNmy)
        TtNmy, TtNmT, TtNmF = jnparray(TtNmy), jnparray(TtNmT), jnparray(TtNmF)
        # closes on P_var_inv, FtNmF, FtNmy, FtMmT, ytNmy, TtNmy, TtNmT, TtNmF
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