from collections.abc import Sequence
import functools
import inspect

import numpy as np
import scipy as sp

try:
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    import jax.tree_util
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import mlx.core as mx
    import mlx.nn as mx_nn
    import mlx.core.random as mx_random
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

def mlx_block_diag(*arrs):
    """Create a block diagonal matrix from provided arrays for MLX backend."""
    if not arrs:
        return mx.array([])
    
    shapes = [a.shape for a in arrs]
    out_shape = (sum(s[0] for s in shapes), sum(s[1] for s in shapes))
    
    out = mx.zeros(out_shape, dtype=arrs[0].dtype)
    r, c = 0, 0
    
    for arr in arrs:
        nr, nc = arr.shape
        out[r:r+nr, c:c+nc] = arr
        r += nr
        c += nc
    
    return out


def backend_diag_indices(n, ndim=2):
    """Backend-agnostic diagonal indices generator"""
    backend = globals().get('_current_backend', 'numpy')
    
    if backend == 'jax':
        return jnp.diag_indices(n, ndim=ndim)
    elif backend == 'mlx':
        if ndim == 2:
            return (mx.arange(n), mx.arange(n))
        else:
            return tuple(mx.arange(n) for _ in range(ndim))
    else:
        return np.diag_indices(n, ndim=ndim)


def _convert_to_backend_array_old(x, backend):
    """Convert input to backend's array type"""
    if backend == 'numpy':
        return np.asarray(x)
    elif backend == 'jax':
        return jnp.asarray(x)
    elif backend == 'mlx':
        if isinstance(x, (np.ndarray, list, tuple)):
            return mx.array(x)
        elif isinstance(x, mx.array):
            return x
        else:
            return mx.array(x)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _convert_to_backend_array(x, backend):
    """Recursively convert input to backend's array type, handling nested structures"""
    if x is None:
        return None
        
    if backend == 'numpy':
        return np.asarray(x)
    elif backend == 'jax':
        return jnp.asarray(x)
    elif backend == 'mlx':
        if isinstance(x, mx.array):
            return x
        elif isinstance(x, (list, tuple)):
            if all(isinstance(e, (np.ndarray, mx.array)) for e in x):
                converted = [_convert_to_backend_array(e, backend) for e in x]
                return mx.stack(converted)
            else:
                return mx.array(x)
        elif isinstance(x, np.ndarray):
            return mx.array(x.tolist()) if x.dtype.kind in 'fiu' else mx.array(x)
        else:
            try:
                return mx.array(x)
            except:
                raise TypeError(f"Cannot convert {type(x)} to MLX array")
    raise ValueError(f"Unknown backend: {backend}")

def _wrap_array_func(func):
    """Decorator to automatically convert inputs to backend arrays"""
    def wrapped(*args, **kwargs):
        backend = globals().get('_current_backend', 'numpy')

        converted_args = [_convert_to_backend_array(arg, backend) for arg in args]
        converted_kwargs = {k: _convert_to_backend_array(v, backend) for k, v in kwargs.items()}

        return func(*converted_args, **converted_kwargs)
    return wrapped

def backend_vmap(func, in_axes=0):
    """Backend-agnostic vectorization wrapper"""
    # TODO: This can probably be optimized a lot for MLX
    backend = globals().get('_current_backend', 'numpy')

    if backend == 'jax':
        return jax.vmap(func, in_axes=in_axes)
    elif backend == 'mlx':
        def mlx_vmapped(*args):
            if isinstance(in_axes, int):
                axis_flags = [in_axes] * len(args)
            else:
                axis_flags = in_axes

            vec_dims = [i for i, ax in enumerate(axis_flags) if ax is not None]
            if not vec_dims:
                return func(*args)

            batch_size = args[vec_dims[0]].shape[axis_flags[vec_dims[0]]]

            results = []
            for i in range(batch_size):
                batch_args = []
                for arg, ax in zip(args, axis_flags):
                    if ax is None:
                        batch_args.append(arg)
                    else:
                        batch_args.append(mx.take(arg, mx.array([i]), axis=ax))
                results.append(func(*batch_args))
            return mx.stack(results)
        return mlx_vmapped
    else:
        def numpy_vmapped(*args):
            return np.vectorize(func, signature='()->()')(*args)
        return numpy_vmapped

def config(**kwargs):
    global jnp, jsp, jnparray, jnpzeros, intarray, jnpkey, jnpsplit, jnpnormal
    global matrix_factor, matrix_solve, matrix_norm, partial, _current_backend

    np.logdet = lambda a: np.sum(np.log(np.abs(a)))
    if JAX_AVAILABLE:
        jax.numpy.logdet = lambda a: jax.numpy.sum(jax.numpy.log(jax.numpy.abs(a)))

    np.make2d = lambda a: a if a.ndim == 2 else np.diag(a)
    if JAX_AVAILABLE:
        jax.numpy.make2d = lambda a: a if a.ndim == 2 else jax.numpy.diag(a)

    backend = kwargs.get('backend')
    _current_backend = backend

    if backend == 'numpy':
        jnp, jsp = np, sp

        jnparray = lambda a: np.array(a, dtype=np.float64)
        jnpzeros = lambda a: np.zeros(a, dtype=np.float64)
        intarray = lambda a: np.array(a, dtype=np.int64)

        jnpkey    = lambda seed: np.random.default_rng(seed)
        jnpsplit  = lambda gen: (gen, gen)
        jnpnormal = lambda gen, shape: gen.normal(size=shape)

        partial = functools.partial
    elif backend == 'jax' and JAX_AVAILABLE:
        jnp, jsp = jax.numpy, jax.scipy

        jnparray = lambda a: jnp.array(a, dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32)
        jnpzeros = lambda a: jnp.zeros(a, dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32)
        intarray = lambda a: jnp.array(a, dtype=jnp.int64)

        jnpkey    = lambda seed: jax.random.PRNGKey(seed)
        jnpsplit  = jax.random.split
        jnpnormal = jax.random.normal

        partial = jax.tree_util.Partial
    elif backend == 'mlx' and MLX_AVAILABLE:
        jnp, jsp = mx, mx

        mx.diag = _wrap_array_func(mx.diag)

        def mlx_jnparray(a):
            if isinstance(a, (list, tuple)) and any(isinstance(x, np.ndarray) for x in a):
                return mx.stack([_convert_to_backend_array(x, backend) for x in a])
            return _convert_to_backend_array(a, backend)

        jnparray = mlx_jnparray
        jnp.diag_indices = backend_diag_indices
        jnpzeros = lambda a: mx.zeros(a, dtype=mx.float32)
        intarray = lambda a: mx.array(a, dtype=mx.int64)

        jnpkey    = lambda seed: mx_random.key(seed)
        jnpsplit  = mx_random.split
        jnpnormal = mx_random.normal

        jsp.linalg.block_diag = mlx_block_diag
        partial = functools.partial
    else:
        raise ValueError(f"Backend {backend} not available or not supported")

    factor = kwargs.get('factor')

    if factor == 'cholesky':
        if backend == 'mlx':
            matrix_factor = lambda a: (mx.linalg.cholesky(a), False)
            matrix_solve = lambda cf, b: mx.linalg.cholesky_solve(cf[0], b)
        else:
            matrix_factor = jsp.linalg.cho_factor
            matrix_solve = jsp.linalg.cho_solve
        matrix_norm = 2.0
    elif factor == 'lu':
        if backend == 'mlx':
            raise NotImplementedError("LU factorization not yet implemented in MLX")
        else:
            matrix_factor = jsp.linalg.lu_factor
            matrix_solve = jsp.linalg.lu_solve
        matrix_norm = 1.0

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
        # globalgp has gp.Fs instead, which maybe is not ideal
        F = [np.hstack(Fs) for Fs in zip(*[gp.F if hasattr(gp, 'F') else gp.Fs for gp in gplist])]

        if all(isinstance(gp.Phi, VectorNoiseMatrix1D_var) for gp in gplist):
            def Phi(params):
                return jnp.hstack([gp.Phi.getN(params) for gp in gplist])
            Phi.params = sorted(set.union(*[set(gp.Phi.params) for gp in gplist]))

            multigp = VariableGP(VectorNoiseMatrix1D_var(Phi), F)
        elif all(isinstance(gp.Phi, (VectorNoiseMatrix1D_var, NoiseMatrix2D_var)) for gp in gplist):
            cvarslist = [list(gp.index) for gp in gplist]
            # pinvlist = [gp.Phi.make_inv() for gp in gplist]
            psolvelist = [gp.Phi.make_solve_1d() for gp in gplist]

            def priorfunc(params):
                ret = 0.0
                for cvars, psolve in zip(cvarslist, psolvelist):
                    if getattr(psolve, 'vector', False):
                        c = jnp.array([params[cvar] for cvar in cvars])
                        Pmc, ldP = psolve(params, c)
                        ret = ret - 0.5 * jnp.sum(c * Pmc) - 0.5 * jnp.sum(ldP)
                    else:
                        c = jnp.concatenate([params[cvar] for cvar in cvars])
                        Pmc, ldP = psolve(params, c)
                        ret = ret - 0.5 * c @ Pmc - 0.5 * ldP

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

def CompoundDelay(residuals, delays):
    def delayfunc(params):
        return residuals - sum(delay(params) for delay in delays)
    delayfunc.params = sorted(set.union(*[set(delay.params) for delay in delays]))

    return delayfunc


# dispatch to NoiseMatrix1D or NoiseMatrix2D based on annotation

def NoiseMatrix12D_var(getN):
    if getN.type == jax.Array:
        return NoiseMatrix2D_var(getN)
    else:
        return NoiseMatrix1D_var(getN)

# consider passing inv as a 1D object

class NoiseMatrix1D_novar(ConstantKernel):
    def __init__(self, N):
        self.N = N
        self.ld = np.logdet(N)

        self.params = []

    def make_kernelproduct(self, y):
        if callable(y):
            y_var = y
            N, ld = jnparray(self.N), np.logdet(self.N)

            def kernelproduct(params):
                yp = y_var(params)

                return -0.5 * jnp.sum(yp**2 / N) - 0.5 * ld
            kernelproduct.params = sorted(set(y.params))
        else:
            product = -0.5 * np.sum(y**2 / self.N) - 0.5 * np.logdet(self.N)

            def kernelproduct(params):
                return product
            kernelproduct.params = []

        return kernelproduct

    def inv(self):
        return jnp.diag(1.0 / self.N), self.ld

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
            return -0.5 * jnp.sum(y2 / N) - 0.5 * jnp.logdet(N)

        kernelproduct.params = getN.params

        return kernelproduct

    def inv(self, params):
        N = self.getN(params)
        return np.diag(1.0 / N), np.logdet(N)

    def make_inv(self):
        getN = self.getN

        # closes on getN
        def inv(params):
            N = getN(params)
            return jnp.diag(1.0 / N), jnp.logdet(N)
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

    def solve_1d(self, params, y):
        N = self.getN(params)

        return y / N, np.logdet(N)

    def solve_2d(self, params, F):
        N = self.getN(params)

        return F / N[:, np.newaxis], np.logdet(N)

    def make_solve_1d(self):
        getN = self.getN

        def solve_1d(params, y):
            N = getN(params)

            return y / N, jnp.logdet(N)

        return solve_1d

    def make_solve_2d(self):
        getN = self.getN

        def solve_2d(params, T):
            N = getN(params)

            return T / N[:, jnp.newaxis], jnp.logdet(N)

        return solve_2d


class NoiseMatrix2D_novar(ConstantKernel):
    def __init__(self, N):
        self.N = N

        self.invN = np.linalg.inv(N)
        self.ld = np.linalg.slogdet(N)[1]

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
#            N = getN(params)
#            return jnp.linalg.inv(N), jnp.linalg.slogdet(N)[1]

            cf = matrix_factor(getN(params))
            inv = matrix_solve(cf, jnp.eye(cf[0].shape[0]))
            ld = matrix_norm * jnp.logdet(jnp.diag(cf[0]))

            return inv, ld
        inv.params = getN.params

        return inv

    def make_solve_1d(self):
        getN = self.getN

        def solve_1d(params, y):
            N = getN(params)

            cf = matrix_factor(N)
            return matrix_solve(cf, y), matrix_norm * jnp.logdet(jnp.diag(cf[0]))
        solve_1d.params = getN.params

        return solve_1d

    def make_sample(self):
        getN = self.getN

        def sample(key, params):
            N = getN(params)

            key, subkey = jnpsplit(key)
            return key, jnp.dot(jsp.linalg.cholesky(N, lower=True),
                                jnpnormal(subkey, (N.shape[0],)))

        sample.params = getN.params

        return sample


def VectorNoiseMatrix12D_var(getN):
    if getN.type == jax.Array:
        return VectorNoiseMatrix2D_var(getN)
    else:
        return VectorNoiseMatrix1D_var(getN)


class VectorNoiseMatrix1D_var(VariableKernel):
    def __init__(self, getN):
        self.getN = getN
        self.params = getN.params

    def make_solve_1d(self):
        getN = self.getN

        def solve_1d(params, y):
            N = getN(params)

            return y / N, jnp.sum(jnp.log(jnp.abs(N)), axis=1)
        solve_1d.params = getN.params
        solve_1d.vector = True

        return solve_1d

    def make_inv(self):
        getN = self.getN

        def inv(params):
            N = getN(params)

            return 1.0 / N, jnp.sum(jnp.log(jnp.abs(N)), axis=1)

            # n, m = N.shape
            # i1, i2 = jnp.diag_indices(m, ndim=2) # it's hard to vectorize numpy.diag!
            # return jnpzeros((n, m, m)).at[:,i1,i2].set(1.0 / N), jnp.sum(jnp.log(N), axis=1)
        inv.params = getN.params
        inv.vector = True

        return inv


class VectorNoiseMatrix2D_var(VariableKernel):
    def __init__(self, getN):
        self.getN = getN
        self.params = getN.params

    def make_inv(self):
        getN = self.getN

        def inv(params):
            N = getN(params)

            # simple inversion
            # if method == 'inv':
            #    return jnp.linalg.inv(N), jnp.linalg.slogdet(N)[1]

            cf = matrix_factor(N)
            inv = matrix_solve(cf, jnp.repeat(jnp.eye(cf[0].shape[1])[jnp.newaxis, :, :],
                                              repeats=cf[0].shape[0], axis=0))

            i1, i2 = jnp.diag_indices(cf[0].shape[1], ndim=2)
            ld = matrix_norm * jnp.sum(jnp.log(jnp.abs(cf[0][:, i1, i2])), axis=1)

            return inv, ld
        inv.params = getN.params
        inv.vector = True

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
        # should be able to handle also _varNFP
        return ShermanMorrisonKernel_varNP(N, F, P)
    else:
        raise TypeError("N and P must be ConstantKernel or VariableKernel instances")


class ShermanMorrisonKernel_novar(ConstantKernel):
    def __init__(self, N, F, P):
        # (N + F P F^T)^-1 = N^-1 - N^-1 F (P^-1 + F^T N^-1 F)^-1 F^T N^-1
        # |N + F P F^T| = |N| |P| |P^-1 + F^T N^-1 F|

        self.N, self.F, self.P = N, F, P
        self.NmF, ldN = N.solve_2d(F)
        FtNmF = F.T @ self.NmF

        Pinv, ldP = P.inv()
        self.cf = sp.linalg.cho_factor(Pinv + FtNmF)
        self.ld = ldN + ldP + np.logdet(np.diag(self.cf[0]))

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
        if callable(y):
            y_var, N_solve_1d = y, self.N.make_solve_1d()
            NmF, cf, ld = jnparray(self.NmF), (jnparray(self.cf[0]), self.cf[1]), self.ld

            def kernelproduct(params):
                yp = y_var(params)

                Nmy = N_solve_1d(yp)[0] - self.NmF @ jsp.linalg.cho_solve(cf, NmF.T @ yp)

                return -0.5 * yp @ Nmy - 0.5 * ld
            kernelproduct.params = sorted(set(y.params))
        else:
            Nmy = self.N.solve_1d(y)[0] - self.NmF @ sp.linalg.cho_solve(self.cf, self.NmF.T @ y)
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
        # returns
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

        y, ytNmy = jnparray(y), jnparray(ytNmy)
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

            return -0.5 * (ytNmy - ytXy) - 0.5 * (ldN + ldP + matrix_norm * jnp.logdet(jnp.diag(cf[0])))

        kernelproduct.params = F_var.params + P_var_inv.params

        return kernelproduct


class ShermanMorrisonKernel_varNP(VariableKernel):
    def __init__(self, N_var, F, P_var):
        self.N, self.F, self.P_var = N_var, F, P_var

    def make_kernelproduct(self, y):
        N_solve_1d = self.N.make_solve_1d()
        N_solve_2d = self.N.make_solve_2d()

        P_var_inv = self.P_var.make_inv()

        y = jnparray(y)

        # avoid a separate ShermanMorrisonKernel_varNFP
        if callable(self.F):
            Ffunc = self.F
        else:
            Fmat = jnparray(self.F)
            Ffunc = lambda params: Fmat

        def kernelproduct(params):
            Nmy, ldN = N_solve_1d(params, y)
            ytNmy = y @ Nmy

            Fmat = Ffunc(params)

            NmF, ldN = N_solve_2d(params, Fmat)
            FtNmF = Fmat.T @ NmF
            NmFty = NmF.T @ y

            Pinv, ldP = P_var_inv(params)
            cf = matrix_factor(Pinv + FtNmF)
            ytXy = NmFty.T @ matrix_solve(cf, NmFty)

            return -0.5 * (ytNmy - ytXy) - 0.5 * (ldN + ldP + matrix_norm * jnp.logdet(jnp.diag(cf[0])))
        kernelproduct.params = sorted(self.N.params + P_var_inv.params)

        return kernelproduct

    def make_kernelproduct_gpcomponent(self, y):
        # -0.5 yt Nm y + yt Nm F a - 0.5 ct Ft Nm F c - 0.5 log |2 pi N| - 0.5 cT Pm c - 0.5 log |2 pi P|

        N_solve_1d = self.N.make_solve_1d()
        N_solve_2d = self.N.make_solve_2d()

        P_solve = self.P_var.make_solve_1d()

        cvars = list(self.index.keys())

        y, Fmat = jnparray(y), jnparray(self.F)

        def kernelproduct(params):
            c = jnp.concatenate([params[cvar] for cvar in cvars])

            NmF, ldN = N_solve_2d(params, Fmat)
            FtNmF = Fmat.T @ NmF
            NmFty = NmF.T @ y

            Nmy, ldN = N_solve_1d(params, y)
            ytNmy = y @ Nmy

            Pmc, ldP = P_solve(params, c)

            return (-0.5 * ytNmy + c @ NmFty - 0.5 * c @ (FtNmF @ c)
                    -0.5 * ldN - 0.5 * c @ Pmc - 0.5 * ldP)    # c @ Pmc was c @ (Pm @ c)

        kernelproduct.params = sorted(self.N.params + self.P_var.params + cvars)

        return kernelproduct

    def make_kernelsolve(self, y, T):
        N_solve_1d = self.N.make_solve_1d()
        N_solve_2d = self.N.make_solve_2d()

        P_var_inv = self.P_var.make_inv()

        y, Fmat = jnparray(y), jnparray(self.F)

        def kernelsolve(params):
            Nmy, _ = N_solve_1d(params, y) if y.ndim == 1 else N_solve_2d(params, y)
            TtNmy = T.T @ Nmy
            FtNmy = Fmat.T @ Nmy

            NmT, _ = N_solve_2d(params, T)
            FtNmT = Fmat.T @ NmT
            TtNmT = T.T @ NmT

            NmF, _ = N_solve_2d(params, Fmat)
            FtNmF = Fmat.T @ NmF
            TtNmF = T.T @ NmF

            Pinv, _ = P_var_inv(params)
            cf = jsp.linalg.cho_factor(Pinv + FtNmF)

            TtSy = TtNmy - TtNmF @ jsp.linalg.cho_solve(cf, FtNmy)
            TtST = TtNmT - TtNmF @ jsp.linalg.cho_solve(cf, FtNmT)

            return TtSy, TtST

        kernelsolve.params = sorted(self.N.params + P_var_inv.params)

        return kernelsolve


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

    def make_kernelproduct_vary(self, y):
        NmF, ldN = self.N.solve_2d(self.F)
        FtNmF = self.F.T @ NmF

        y_var = y
        NmF, FtNmF, ldN = jnparray(NmF), jnparray(FtNmF), jnparray(ldN)
        P_var_inv = self.P_var.make_inv()
        N_solve_1d = self.N.make_solve_1d()

        # closes on y, delay, P_var_inv, NmF, FtNmF, ldN
        def kernel(params):
            yp = y_var(params)

            Nmy, _ = N_solve_1d(yp)
            ytNmy = yp @ Nmy
            NmFty = NmF.T @ yp

            Pinv, ldP = P_var_inv(params)
            cf = matrix_factor(Pinv + FtNmF)
            ytXy = NmFty.T @ matrix_solve(cf, NmFty)

            return -0.5 * (ytNmy - ytXy) - 0.5 * (ldN + ldP + matrix_norm * jnp.logdet(jnp.diag(cf[0])))
        kernel.params = y.params + P_var_inv.params

        return kernel

    def make_kernelproduct(self, y):
        if callable(y):
            return self.make_kernelproduct_vary(y)

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

            # direct inv
            # ytXy = NmFty.T @ jnp.linalg.inv(Pinv + FtNmF) @ NmFty

            # SVD solution
            # U, S, VT = jnp.linalg.svd(Pinv + FtNmF)
            # ytXy = NmFty.T @ VT.T @ np.diag(1/S) @ U.T @ NmFty
            # matrix_norm, cf = 1.0, (np.diag(S), None)

            return -0.5 * (ytNmy - ytXy) - 0.5 * (ldN + ldP + matrix_norm * jnp.logdet(jnp.diag(cf[0])))

        kernelproduct.params = P_var_inv.params

        return kernelproduct

    def make_kernelproduct_gpcomponent(self, y, transform=None):
        # -0.5 yt Nm y + yt Nm F a - 0.5 ct Ft Nm F c - 0.5 log |2 pi N| - 0.5 cT Pm c - 0.5 log |2 pi P|

        NmF, ldN = self.N.solve_2d(self.F)
        FtNmF = self.F.T @ NmF
        NmFty = NmF.T @ y

        Nmy, ldN = self.N.solve_1d(y)
        ytNmy = y @ Nmy

        ytNmy, NmFty, FtNmF = jnparray(ytNmy), jnparray(NmFty), jnparray(FtNmF)
        P_solve = self.P_var.make_solve_1d()
        cvars = self.index

        def kernelproduct(params):
            c = jnp.concatenate([params[cvar] for cvar in cvars])

            if transform is not None:
                c, ldL = transform(params, c)
            else:
                ldL = 0.0

            Pmc, ldP = P_solve(params, c)

            ret = (-0.5 * ytNmy + c @ NmFty - 0.5 * c @ (FtNmF @ c)
                   -0.5 * ldN - 0.5 * c @ Pmc - 0.5 * ldP + ldL)    # c @ Pmc was c @ (Pm @ c)
            return (ret, c) if transform is not None else ret
        kernelproduct.params = sorted(set(self.P_var.params + list(cvars) + ([] if transform is None else transform.params)))

        return kernelproduct

    # makes a function that returns the tuple
    # * -0.5 (yt Nm y - yt Nm F (Phi + Ft Nm F)^-1 y - 0.5 (logdet N + logdet Phi + logdet S)
    # * Tt (Phi + Ft Nm F)^-1 y
    # * Tt (Phi + Ft Nm F)^-1 T
    # with fixed y and T

    def make_kernelterms_vary(self, y, T):
        NmF, _ = self.N.solve_2d(self.F)
        FtNmF = self.F.T @ NmF
        TtNmF = T.T @ NmF

        NmT, _ = self.N.solve_2d(T)
        FtNmT = self.F.T @ NmT
        TtNmT = T.T @ NmT

        y_var = y
        P_var_inv = self.P_var.make_inv()
        Ft, Tt = jnparray(self.F.T), jnparray(T.T)
        FtNmF, FtNmT = jnparray(FtNmF), jnparray(FtNmT)
        TtNmT, TtNmF = jnparray(TtNmT), jnparray(TtNmF)
        N_solve_1d = self.N.make_solve_1d()

        # closes on P_var_inv, FtNmF, FtNmy, FtMmT, ytNmy, TtNmy, TtNmT, TtNmF
        def kernelterms(params):
            yp = y_var(params)

            Nmy, ldN = N_solve_1d(yp)
            ytNmy = yp @ Nmy
            FtNmy = Ft @ Nmy
            TtNmy = Tt @ Nmy

            Pinv, ldP = P_var_inv(params)
            cf = matrix_factor(Pinv + FtNmF)

            sol = matrix_solve(cf, FtNmy)
            sol2 = matrix_solve(cf, FtNmT)

            a = -0.5 * (ytNmy - FtNmy.T @ sol) - 0.5 * (ldN + ldP + matrix_norm * jnp.logdet(jnp.diag(cf[0])))
            b = TtNmy - TtNmF @ sol
            c = TtNmT - TtNmF @ sol2

            return a, b, c

        kernelterms.params = sorted(set(self.P_var.params + y.params))

        return kernelterms

    def make_kernelterms(self, y, T):
        # Sigma = (N + F P Ft)
        # Sigma^-1 = Nm - Nm F (P^-1 + Ft Nm F)^-1 Ft Nm
        #
        # yt Sigma^-1 y = yt Nm y - (yt Nm F) C^-1 (Ft Nm y)
        # Tt Sigma^-1 y = Tt Nm y - Tt Nm F C^-1 (Ft Nm y)
        # Tt Sigma^-1 T = Tt Nm T - (Tt Nm F) C^-1 (Ft Nm T)

        if callable(y):
            return self.make_kernelterms_vary(y, T)

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

            a = -0.5 * (ytNmy - FtNmy.T @ sol) - 0.5 * (ldN + ldP + matrix_norm * jnp.logdet(jnp.diag(cf[0])))
            b = TtNmy - TtNmF @ sol
            c = TtNmT - TtNmF @ sol2

            return a, b, c

        kernelterms.params = self.P_var.params

        return kernelterms


class ShermanMorrisonKernel_varN(VariableKernel):
    def __init__(self, N_var, F, P):
        self.N_var, self.F, self.P = N_var, F, P
        self.Pinv, self.ldP = P.inv()
        self.params = N_var.params

    def make_sample(self, params):
        N_sample = self.N_var.make_sample()
        P_sample = self.P.make_sample()
        F = jnparray(self.F)

        def sample(key, params):
            key, n = N_sample(key, params)
            key, c = P_sample(key)

            return key, n + jnp.dot(F, c)

        sample.params = N_sample.params

        return sample

    def make_kernelproduct(self, y):
        N_solve_1d = self.N_var.make_solve_1d()
        N_solve_2d = self.N_var.make_solve_2d()

        y = jnparray(y)
        F, Pinv, ldP = jnparray(self.F), jnparray(self.Pinv), jnparray(self.ldP)

        def kernelproduct(params={}):
            NmF, ldN = N_solve_2d(params, F)
            FtNmF = F.T @ NmF

            cf = matrix_factor(Pinv + FtNmF)
            Nmy = N_solve_1d(params, y)[0] - NmF @ matrix_solve(cf, NmF.T @ y)
            ld = ldN + ldP + matrix_norm * jnp.logdet(jnp.diag(cf[0]))

            return -0.5 * y @ Nmy - 0.5 * ld
        kernelproduct.params = self.N_var.params

        return kernelproduct

    def make_kernelterms(self, y, T):
        N_solve_1d = self.N_var.make_solve_1d()
        N_solve_2d = self.N_var.make_solve_2d()

        y, T = jnparray(y), jnparray(T)
        F, Pinv, ldP = jnparray(self.F), jnparray(self.Pinv), jnparray(self.ldP)

        def kernelterms(params={}):
            Nmy, ldN = N_solve_1d(params, y)
            ytNmy = y @ Nmy
            FtNmy = F.T @ Nmy
            TtNmy = T.T @ Nmy

            NmF, _ = N_solve_2d(params, F)
            NmT, _ = N_solve_2d(params, T)

            FtNmF = F.T @ NmF
            TtNmF = T.T @ NmF

            FtNmT = F.T @ NmT
            TtNmT = T.T @ NmT

            cf = matrix_factor(Pinv + FtNmF)

            sol = matrix_solve(cf, FtNmy)
            sol2 = matrix_solve(cf, FtNmT)

            ld = ldN + ldP + matrix_norm * jnp.logdet(jnp.diag(cf[0]))

            a = -0.5 * (ytNmy - FtNmy.T @ sol) - 0.5 * ld
            b = TtNmy - TtNmF @ sol
            c = TtNmT - TtNmF @ sol2

            return a, b, c
        kernelterms.params = self.N_var.params

        return kernelterms

    def make_kernelsolve(self, y, T):
        N_solve_1d = self.N_var.make_solve_1d()
        N_solve_2d = self.N_var.make_solve_2d()

        y = jnparray(y)
        F, Pinv, ldP = jnparray(self.F), jnparray(self.Pinv), jnparray(self.ldP)

        if callable(T):
            def kernelsolve(params):
                Nmy, ldN = N_solve_1d(params, y)
                FtNmy = F.T @ Nmy

                NmF, _ = N_solve_2d(params, F)
                FtNmF = F.T @ NmF

                Tmat = T(params)
                TtNmy  = Tmat.T @ Nmy
                TtNmF  = Tmat.T @ NmF

                NmT, _ = N_solve_2d(params, Tmat)
                FtNmT  = F.T @ NmT
                TtNmT  = Tmat.T @ NmT

                cf = matrix_factor(Pinv + FtNmF)

                TtSy = TtNmy - TtNmF @ matrix_solve(cf, FtNmy)
                TtST = TtNmT - TtNmF @ matrix_solve(cf, FtNmT)

                return TtSy, TtST
            kernelsolve.params = self.N_var.params + T.params
        else:
            Tmat = jnparray(T)

            def kernelsolve(params={}):
                Nmy, ldN = N_solve_1d(params, y)
                FtNmy = F.T @ Nmy

                NmF, _ = N_solve_2d(params, F)
                FtNmF = F.T @ NmF

                TtNmy  = Tmat.T @ Nmy
                TtNmF  = Tmat.T @ NmF

                NmT, _ = N_solve_2d(params, T)
                FtNmT  = F.T @ NmT
                TtNmT  = Tmat.T @ NmT

                cf = matrix_factor(Pinv + FtNmF)

                TtSy = TtNmy - TtNmF @ matrix_solve(cf, FtNmy)
                TtST = TtNmT - TtNmF @ matrix_solve(cf, FtNmT)

                return TtSy, TtST
            kernelsolve.params = self.N_var.params

        return kernelsolve

    def solve_1d(self, params, y):
        NmF, ldN = self.N_var.solve_2d(params, self.F)
        NmFty = NmF.T @ y

        cf = sp.linalg.cho_factor(self.Pinv + self.F.T @ NmF)
        ld = ldN + self.ldP + jnp.logdet(np.diag(cf[0]))

        return self.N_var.solve_1d(params, y)[0] - NmF @ sp.linalg.cho_solve(cf, NmFty), ld

    def make_solve_1d(self):
        N_solve_1d = self.N_var.make_solve_1d()
        N_solve_2d = self.N_var.make_solve_2d()

        F, Pinv, ldP = jnparray(self.F), jnparray(self.Pinv), jnparray(self.ldP)

        def solve_1d(params, y):
            NmF, ldN = N_solve_2d(params, F)
            NmFty = NmF.T @ y

            cf = matrix_factor(Pinv + F.T @ NmF)
            ld = ldN + ldP + matrix_norm * jnp.logdet(jnp.diag(cf[0]))

            return N_solve_1d(params, y)[0] - NmF @ matrix_solve(cf, NmFty), ld

        return solve_1d

    def solve_2d(self, params, Fr):
        NmFl, ldN = self.N_var.solve_2d(params, self.F)
        NmFltFr = NmFl.T @ Fr

        cf = sp.linalg.cho_factor(self.Pinv + self.F.T @ NmFl)
        ld = ldN + self.ldP + np.logdet(np.diag(cf[0]))

        return self.N_var.solve_2d(params, Fr)[0] - NmFl @ sp.linalg.cho_solve(cf, NmFltFr), ld

    def make_solve_2d(self):
        N_solve_2d = self.N_var.make_solve_2d()

        N_var_inv = self.N_var.make_inv()
        Pinv, ldP = self.P.inv()

        Fl, Pinv, ldP = jnparray(self.F), jnparray(self.Pinv), jnparray(self.ldP)

        def solve_2d(params, Fr):
            NmFl, ldN = N_solve_2d(params, Fl)
            NmFltFr = NmFl.T @ Fr

            cf = matrix_factor(Pinv + Fl.T @ NmFl)
            ld = ldN + ldP + matrix_norm * jnp.logdet(jnp.diag(cf[0]))

            return N_solve_2d(params, Fr)[0] - NmFl @ matrix_solve(cf, NmFltFr), ld

        return solve_2d


class VectorShermanMorrisonKernel_varP(VariableKernel):
    def __init__(self, Ns, Fs, P_var):
        self.Ns, self.Fs, self.P_var = Ns, Fs, P_var

    def make_kernelproduct_vary(self, ys):
        NmFs, ldNs = zip(*[N.solve_2d(F) for N, F in zip(self.Ns, self.Fs)])
        FtNmFs = [F.T @ NmF for F, NmF in zip(self.Fs, NmFs)]

        # tricky... when defining multiple closures like this, one must take care
        # not to refer to variables that change values in the enclosing
        # there may be a better way to do this

        yprods = []
        for y, N, NmF in zip(ys, self.Ns, NmFs):
            if callable(y):
                def yprod(params, y_var = y, N_solve_1d = N.make_solve_1d(), NmFt = jnparray(NmF.T)):
                    yp = y_var(params)
                    Nmy, _ = N_solve_1d(yp)
                    return jnparray(yp @ Nmy), jnparray(NmFt @ yp)
                yprod.params = y.params
            else:
                Nmy, _ = N.solve_1d(y)

                def yprod(params, ytNmy = jnparray(y @ Nmy), NmFty = jnparray(NmF.T @ y)):
                    return ytNmy, NmFty
                yprod.params = []

            yprods.append(yprod)

        P_var_inv = self.P_var.make_inv()
        FtNmF, ldN = jnparray(FtNmFs), float(sum(ldNs))

        def kernelproduct(params):
            Pinv, ldP = P_var_inv(params)

            yprodvals = [yprod(params) for yprod in yprods]

            ytNmy = sum([ypv[0] for ypv in yprodvals])
            NmFty = jnparray([ypv[1] for ypv in yprodvals])

            if Pinv.ndim == 2:
                # Pinv is diagonal
                i1, i2 = jnp.diag_indices(Pinv.shape[1], ndim=2)
                cf = matrix_factor(FtNmF.at[:,i1,i2].add(Pinv))
            else:
                cf = matrix_factor(FtNmF + Pinv)

            ytXy = jnp.sum(NmFty * matrix_solve(cf, NmFty)) # was NmFty.T @ jsp.linalg.cho_solve(...)

            i1, i2 = jnp.diag_indices(cf[0].shape[1], ndim=2) # it's hard to vectorize numpy.diag!
            return -0.5 * (ytNmy - ytXy) - 0.5 * (ldN + jnp.sum(ldP) + matrix_norm * jnp.logdet(cf[0][:,i1,i2]))

        kernelproduct.params = sorted(set(P_var_inv.params + sum([yp.params for yp in yprods], [])))

        return kernelproduct

    def make_kernelproduct(self, ys):
        if any(callable(y) for y in ys):
            return self.make_kernelproduct_vary(ys)

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

            if Pinv.ndim == 2:
                # Pinv is diagonal
                i1, i2 = jnp.diag_indices(Pinv.shape[1], ndim=2)
                cf = matrix_factor(FtNmF.at[:,i1,i2].add(Pinv))
            else:
                cf = matrix_factor(FtNmF + Pinv)

            ytXy = jnp.sum(NmFty * matrix_solve(cf, NmFty)) # was NmFty.T @ jsp.linalg.cho_solve(...)

            i1, i2 = jnp.diag_indices(cf[0].shape[1], ndim=2) # it's hard to vectorize numpy.diag!
            return -0.5 * (ytNmy - ytXy) - 0.5 * (ldN + jnp.sum(ldP) + matrix_norm * jnp.logdet(cf[0][:,i1,i2]))

        kernelproduct.params = P_var_inv.params

        return kernelproduct

    def make_kernelproduct_gpcomponent(self, ys, transform=None):
        # -0.5 yt Nm y + yt Nm F a - 0.5 ct Ft Nm F c - 0.5 log |2 pi N| - 0.5 cT Pm c - 0.5 log |2 pi P|

        NmFs, ldNs = zip(*[N.solve_2d(F) for N, F in zip(self.Ns, self.Fs)])
        FtNmFs = [F.T @ NmF for F, NmF in zip(self.Fs, NmFs)]

        Nmys, _  = zip(*[N.solve_1d(y) for N, y in zip(self.Ns, ys)])
        ytNmys = [y @ Nmy for y, Nmy in zip(ys, Nmys)]
        NmFtys = [NmF.T @ y for NmF, y in zip(NmFs, ys)]

        FtNmF, NmFty = jnparray(FtNmFs), jnparray(NmFtys)
        ytNmy, ldN = float(sum(ytNmys)), float(sum(ldNs))

        if isinstance(self.index, list):
            cvarsall = self.index
        else:
            cvarsall = [{par: sl} for par, sl in self.index.items()]

        # cvarsall is a list over pulsars; each cvars is a dict over GPs
        # make an npsr x nbasis array from a dictionary of parameter vectors
        def fold(params):
            return jnp.array([jnp.concatenate([params[cvar] for cvar in cvars]) for cvars in cvarsall])

        # make a dictionary back from the array
        def unfold(c):
            cv, cnt = c.flatten(), 0
            return {cvar: cv[cnt:(cnt := cnt + sl.stop - sl.start)]
                    for cvars in cvarsall for cvar, sl in cvars.items()}

        if hasattr(self, 'prior'):
            P_var_prior = self.prior

            def kernelproduct(params):
                c = fold(params)

                if transform is not None:
                    c, ldL = transform(params, c)
                    params = {**params, **unfold(c)}
                else:
                    ldL = 0.0

                logpr = P_var_prior(params)

                ret = (-0.5 * ytNmy + jnp.sum(c * NmFty) - 0.5 * jnp.einsum('ij,ijk,ik', c, FtNmF, c)
                       -0.5 * ldN - logpr + ldL)
                return (ret, c) if transform is not None else ret

            kernelproduct.params = sorted(set(P_var_prior.params +
                                              sum([list(cvars) for cvars in cvarsall], []) +
                                              ([] if transform is None else transform.params)))
        else:
            P_var_inv = self.P_var.make_inv()

            def kernelproduct(params):
                c = fold(params)

                if transform is not None:
                    c, ldL = transform(params, c)
                else:
                    ldL = 0.0

                # P_var_inv does not use the coefficients
                Pm, ldP = P_var_inv(params)

                ret = (-0.5 * ytNmy + jnp.sum(c * NmFty) - 0.5 * jnp.einsum('ij,ijk,ik', c, FtNmF, c)
                       -0.5 * ldN - 0.5 * jnp.sum(c * Pm * c) - 0.5 * jnp.sum(ldP) + ldL) # note Pm is 1D
                return (ret, c) if transform is not None else ret

            kernelproduct.params = sorted(set(P_var_inv.params +
                                              sum([list(cvars) for cvars in cvarsall], []) +
                                              ([] if transform is None else transform.params)))

        return kernelproduct

    def make_kernelterms_vary(self, ys, Ts):
        NmFs, ldNs = zip(*[N.solve_2d(F) for N, F in zip(self.Ns, self.Fs)])
        FtNmFs = [F.T @ NmF for F, NmF in zip(self.Fs, NmFs)]
        TtNmFs = [T.T @ NmF for T, NmF in zip(Ts, NmFs)]

        NmTs, _ = zip(*[N.solve_2d(T) for N, T in zip(self.Ns, Ts)])
        FtNmTs = [F.T @ NmT for F, NmT in zip(self.Fs, NmTs)]
        TtNmTs = [T.T @ NmT for T, NmT in zip(Ts, NmTs)]

        yprods = []
        for y, N, F, T in zip(ys, self.Ns, self.Fs, Ts):
            if callable(y):
                def yprod(params, y_var = y, N_solve_1d = N.make_solve_1d(), Ft = jnparray(F.T), Tt = jnparray(T.T)):
                    yp = y_var(params)
                    Nmy, _ = N_solve_1d(yp)
                    return jnparray(yp @ Nmy), jnparray(Ft @ Nmy), jnparray(Tt @ Nmy)
                yprod.params = y.params
            else:
                Nmy, _ = N.solve_1d(y)

                def yprod(params, ytNmy = jnparray(y @ Nmy), FtNmy = jnparray(F.T @ Nmy), TtNmy = jnparray(T.T @ Nmy)):
                    return ytNmy, FtNmy, TtNmy
                yprod.params = []

            yprods.append(yprod)

        P_var_inv = self.P_var.make_inv()
        FtNmF, FtNmT = jnparray(FtNmFs), jnparray(FtNmTs)
        TtNmT, TtNmF = jnparray(TtNmTs), jnparray(TtNmFs)
        ldN = float(sum(ldNs))

        def kernelterms(params):
            Pinv, ldP = P_var_inv(params)

            if Pinv.ndim == 2:
                i1, i2 = jnp.diag_indices(Pinv.shape[1], ndim=2)
                cf = matrix_factor(FtNmF.at[:,i1,i2].add(Pinv))
            else:
                cf = matrix_factor(FtNmF + Pinv)

            yprodvals = [yprod(params) for yprod in yprods]

            ytNmy = sum([ypv[0] for ypv in yprodvals])
            FtNmy = jnparray([ypv[1] for ypv in yprodvals])
            TtNmy = jnparray([ypv[2] for ypv in yprodvals])

            sol = matrix_solve(cf, FtNmy)
            sol2 = matrix_solve(cf, FtNmT)

            i1, i2 = jnp.diag_indices(cf[0].shape[1], ndim=2)
            a = -0.5 * (ytNmy - jnp.sum(FtNmy * sol)) - 0.5 * (ldN + jnp.sum(ldP) + matrix_norm * jnp.logdet(cf[0][:,i1,i2]))
            b = TtNmy - jnp.sum(TtNmF * sol[:, jnp.newaxis, :], axis=2)
            c = TtNmT - TtNmF @ sol2 # fine as is!

            return a, b, c
        kernelterms.params = sorted(set(P_var_inv.params + sum([yp.params for yp in yprods], [])))

        return kernelterms

    def make_kernelterms(self, ys, Ts):
        # Sigma = (N + F P Ft)
        # Sigma^-1 = Nm - Nm F (P^-1 + Ft Nm F)^-1 Ft Nm
        #
        # yt Sigma^-1 y = yt Nm y - (yt Nm F) C^-1 (Ft Nm y)
        # Tt Sigma^-1 y = Tt Nm y - Tt Nm F C^-1 (Ft Nm y)
        # Tt Sigma^-1 T = Tt Nm T - (Tt Nm F) C^-1 (Ft Nm T)

        if any(callable(y) for y in ys):
            return self.make_kernelterms_vary(ys, Ts)

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

            if Pinv.ndim == 2:
                i1, i2 = jnp.diag_indices(Pinv.shape[1], ndim=2)
                cf = matrix_factor(FtNmF.at[:,i1,i2].add(Pinv))
            else:
                cf = matrix_factor(FtNmF + Pinv)

            sol = matrix_solve(cf, FtNmy)
            sol2 = matrix_solve(cf, FtNmT)

            i1, i2 = jnp.diag_indices(cf[0].shape[1], ndim=2)
            a = -0.5 * (ytNmy - jnp.sum(FtNmy * sol)) - 0.5 * (ldN + jnp.sum(ldP) + matrix_norm * jnp.logdet(cf[0][:,i1,i2]))
            b = TtNmy - jnp.sum(TtNmF * sol[:, jnp.newaxis, :], axis=2)
            c = TtNmT - TtNmF @ sol2 # fine as is!

            return a, b, c

        kernelterms.params = self.P_var.params

        return kernelterms
