"""Metamath-backed PulsarLikelihood / GlobalLikelihood / ArrayLikelihood.

Parallel to `likelihood.py`. Selected by `ds.config(backend='metamath')`,
which rebinds the top-level `ds.PulsarLikelihood` (etc.) to the classes
here. Initially a verbatim copy of `likelihood.py`; matrix.* references
will be migrated to their `metamath` equivalents class-by-class. The
parity suite under `tests/metamatrix/` is the oracle for each swap.
"""
import functools

import numpy as np
import jax

# Numerical primitives (jnp/jsp/jnparray/jnpsplit/jnpnormal/matrix_factor/...)
# and the cglogL Tier-3 helpers (cgsolve, make_logdet_estimator) are reached
# through `utils` (`kh.X`) so that `config(backend=..., factor=...)` continues to
# drive numpy-vs-jax, precision, and cholesky-vs-LU even in the metamath path.
# The GP/Kernel marker types (`ConstantGP`, `VariableGP`, `Kernel`, ...) also
# live in `utils`. Combining a list of globalgps uses `signals.CompoundGlobalGP`,
# which is backend-agnostic (builds metamath kernels in metamath mode).
from . import signals
from . import metamatrix
from . import metamath
from . import utils as kh
from . import summary

# Kernel
#   ConstantKernel
#       define solve_1d, perhaps solve_2d (operate on numpy)
#   VariableKernel
#       define make_solve_1d, perhaps make_solve_2d (return functions that operate on jax)
#
#   all define make_kernelproduct (make_kernelterms)
#
# GP
#   ConstantGP
#       consists of a ConstantKernel and a numpy matrix
#   VariableGP
#       consists of a VariableKernel and a numpy matrix
#
# WoodburyKernel can return a ConstantKernel or a VariableKernel

# npta = je.PulsarLikelihood(je.residuals(psr),
#                            je.makenoise_measurement(psr, noisedict),
#                            je.makegp_ecorr(psr, noisedict),
#                            je.makegp_timing(psr),
#                            je.makegp_fourier('red_noise', psr, je.powerlaw, 10),
#                            je.makegp_fourier('red_noise', psr,
#                                              je.makepowerlaw_crn(5), 10, T=Tspan,
#                                              common=['crn_log10_A', 'crn_gamma']),
#                            concat=True)

def ffunc(graph):
    if callable(graph):
        return graph

    func = metamatrix.func(graph)

    def outfunc(params):
        return func(params=params)
    outfunc.params = func.params

    if hasattr(func, 'graph'):
        outfunc.graph = func.graph

    return outfunc


class PulsarLikelihood(summary.SummaryMixin):
    """Single-pulsar likelihood — metamath-native composition.

    Kernel/GP composition (Woodbury chaining, compound GPs, delays) goes
    through `metamath.*` directly. The GP marker types (`utils.ConstantGP`,
    `utils.VariableGP`) are used for isinstance dispatch because `signals.py`
    factories tag their outputs with them; the actual underlying kernels are
    metamath instances because `ds.config(kernels='metamath')` sets the
    `_kernels` factory mode.
    """
    def __init__(self, args, concat=True, marginalize_all_but_last=None):
        # retain the original components so the model can describe itself
        # (see discovery.summary); the math path uses only y, delay, N below.
        # `concat` is kept too so the kernel-tree view knows whether GPs were
        # fused into one Woodbury layer or chained.
        self.signals = list(args)
        self.concat = concat

        y     = [arg for arg in args if isinstance(arg, np.ndarray) or isinstance(arg, jax.Array)]
        delay = [arg for arg in args if callable(arg)]
        noise = [arg for arg in args if isinstance(arg, kh.Kernel)]
        cgps  = [arg for arg in args if isinstance(arg, kh.ConstantGP)]
        vgps  = [arg for arg in args if isinstance(arg, kh.VariableGP)]

        if len(y) == 0 and len(delay) == 0:
            raise ValueError("I need exactly one residual vector or one or more delay functions.")
        if len(y) > 1 or len(noise) > 1:
            raise ValueError("Only one residual vector and one noise Kernel allowed.")
        elif len(noise) == 0:
            raise ValueError("I need exactly one noise Kernel.")

        if len(y) == 0:
            y = [0.0]

        noise, y = noise[0], y[0]

        if cgps:
            # Timing-model projection (float32-safe, ADR 0004): any GP marked
            # `project=True` (via makegp_timing(..., project=True)) is projected
            # OUT rather than given a 1e40 prior; the remaining cgps stay ordinary
            # Woodbury blocks. Off by default → byte-identical to the branch below.
            proj_cgps = [g for g in cgps if getattr(g, 'project', False)]
            keep_cgps = [g for g in cgps if not getattr(g, 'project', False)]

            if proj_cgps:
                if not keep_cgps:
                    raise NotImplementedError(
                        "timing-model projection currently requires at least one "
                        "non-projected GP (e.g. ECORR) as the kept Woodbury block.")
                # M = timing basis (concatenate if several projected GPs).
                Mbases = [g.F for g in proj_cgps]
                M = (Mbases[0] if len(Mbases) == 1
                     else jax.numpy.concatenate([jax.numpy.asarray(b) for b in Mbases], axis=1))
                kc = (metamath.CompoundGP(keep_cgps)
                      if (len(keep_cgps) > 1 and concat) else keep_cgps[0])
                csm = metamath.WoodburyProjKernel(noise, M, kc.F, kc.Phi)
            elif len(cgps) > 1 and concat:
                cgp = metamath.CompoundGP(cgps)
                csm = metamath.WoodburyKernel(noise, cgp.F, cgp.Phi)
            else:
                csm = noise
                for cgp in cgps:
                    csm = metamath.WoodburyKernel(csm, cgp.F, cgp.Phi)
        else:
            csm = noise

        if vgps:
            for vgp in vgps:
                if hasattr(vgp, 'gpname') and vgp.gpname == 'gw':
                    self.gw = vgp

            # The chained (concat=False) construction below overwrites `.index`
            # per iteration, so only the LAST variable GP keeps sampled
            # coefficients; the rest are silently marginalized. Make that
            # explicit rather than accidental (D2).
            if len(vgps) > 1 and not concat and marginalize_all_but_last is not True:
                shadowed = [getattr(g, 'gpname', '<unnamed>') for g in vgps[:-1]]
                last = getattr(vgps[-1], 'gpname', '<unnamed>')
                raise ValueError(
                    f"PulsarLikelihood(concat=False) with multiple variable GPs "
                    f"analytically marginalizes all but the LAST one: only "
                    f"'{last}' keeps sampled coefficients; {shadowed} are shadowed. "
                    f"Pass marginalize_all_but_last=True to confirm this, or use "
                    f"concat=True to sample all coefficient blocks.")

            if len(vgps) > 1 and concat:
                vgp = metamath.CompoundGP(vgps)
                vsm = metamath.WoodburyKernel(csm, vgp.F, vgp.Phi)
                vsm.index = getattr(vgp, 'index', None)
                vsm.mean = getattr(vgp, 'mean', None)
            else:
                vsm = csm
                for vgp in vgps:
                    vsm = metamath.WoodburyKernel(vsm, vgp.F, vgp.Phi)
                    vsm.index = getattr(vgp, 'index', None)
                    vsm.mean = getattr(vgp, 'mean', None)
        else:
            vsm = csm

        if len(delay) > 0:
            y = metamath.CompoundDelay(y, delay)

        self.y, self.delay, self.N = y, delay, vsm

        for gp in cgps + vgps:
            if hasattr(gp, 'name'):
                self.name = gp.name

    def __setattr__(self, name, value):
        if name == 'residuals' and 'logL' in self.__dict__:
            self.y = value

            if len(self.delay) > 0:
                self.y = metamath.CompoundDelay(self.y, self.delay)

            del self.logL
        else:
            self.__dict__[name] = value

    @functools.cached_property
    def sample_conditional(self):
        cond = self.conditional
        index = self.N.index

        def sample_cond(key, params):
            mu, cf = cond(params)

            key, subkey = kh.jnpsplit(key)
            c = mu + kh.jsp.linalg.solve_triangular(
                cf[0].T, kh.jnpnormal(subkey, mu.shape), lower=False)

            return key, {par: c[sli] for par, sli in index.items()}

        sample_cond.params = cond.params

        return sample_cond

    @functools.cached_property
    def conditional(self):
        # metamath Woodbury kernels always expose `make_conditional` as a
        # graph; the matrix.py-specific fallback (make_kernelsolve_simple /
        # P_var.make_inv) lives in `likelihood.py` and is not needed here.
        if hasattr(self.N, 'make_conditional'):
            return ffunc(self.N.make_conditional(self.y))

        if self.delay:
            raise NotImplementedError(
                "No PulsarLikelihood.conditional with delays so far.")
        raise NotImplementedError(
            "metamath kernel does not expose make_conditional.")

    @functools.cached_property
    def clogL(self):
        if hasattr(self.N, 'make_coefficientproduct'):
            return ffunc(self.N.make_coefficientproduct(self.y))

        if self.delay:
            raise NotImplementedError('No PulsarLikelihood.clogL with delays so far.')
        else:
            # ffunc is a no-op for callables, so this is safe for any return
            # type and makes the property's contract uniform: always a
            # `(params) -> value` callable carrying `.params` (D19).
            return ffunc(self.N.make_kernelproduct_gpcomponent(self.y))

    @functools.cached_property
    def logL(self):
        return ffunc(self.N.make_kernelproduct(self.y))


    @functools.cached_property
    def sample(self):
        if callable(self.y):
            noiseonly = self.N.make_sample()
            delays = self.delay

            def make_sample(key, params):
                key, noise = noiseonly(key, params)
                return key, noise + sum(delay(params) for delay in delays)
            make_sample.params = sorted(set(noiseonly.params + sum([delay.params for delay in delays], [])))

            return make_sample

        return self.N.make_sample()


class GlobalLikelihood(summary.SummaryMixin):
    def __init__(self, psls, globalgp=None):
        self.psls = psls
        self.globalgp = signals.CompoundGlobalGP(globalgp) if isinstance(globalgp, list) else globalgp

    # allow replacement of residuals
    def __setattr__(self, name, value):
        if name == 'residuals':
            for psl, y in zip(self.psls, value):
                psl.y = y

            for p in ['os', 'os_rhosigma', 'logL', 'sample_conditional', 'conditional']:
                if p in self.__dict__:
                    delattr(self, p)
        else:
            self.__dict__[name] = value

    @functools.cached_property
    def sample(self):
        if self.globalgp is None:
            sls = [psl.sample for psl in self.psls]
            if len(sls) == 0:
                raise ValueError('No PulsarLikelihoods in GlobalLikelihood: ' +
                    'if you provided them using a generator, it may have been consumed already. ' +
                    'In that case you can use a list.')

            def sampler(key, params):
                ys = []
                for sl in sls:
                    key, y = sl(key, params)
                    ys.append(y)

                return key, ys

            sampler.params = sorted(set.union(*[set(sl.params) for sl in sls]))
        else:
            sls = [psl.sample for psl in self.psls]
            if len(sls) == 0:
                raise ValueError('No PulsarLikelihoods in GlobalLikelihood: ' +
                    'if you provided them using a generator, it may have been consumed already. ' +
                    'In that case you can use a list.')

            Phi_sample = self.globalgp.Phi.make_sample()

            Fs = [kh.jnparray(F) for F in self.globalgp.Fs]

            i0, slcs = 0, []
            for F in self.globalgp.Fs:
                slcs.append(slice(i0, i0 + F.shape[1]))
                i0 = i0 + F.shape[1]

            def sampler(key, params):
                key, c = Phi_sample(key, params)

                ys = []
                for sl, F, slc in zip(sls, Fs, slcs):
                    key, y = sl(key, params)
                    ys.append(y + kh.jnp.dot(F, c[slc]))

                # ys = [key, _ := sl(key, params) + kh.jnp.dot(F, c[slc]) for sl, F, slc in zip(sls, Fs, slcs)]
                return key, ys

            sampler.params = sorted(set.union(*[set(sl.params) for sl in sls])) + Phi_sample.params

        return sampler

    @functools.cached_property
    def logL(self):
        if self.globalgp is None:
            logls = [psl.logL for psl in self.psls]
            if len(logls) == 0:
                raise ValueError('No PulsarLikelihoods in GlobalLikelihood: ' +
                    'if you provided them using a generator, it may have been consumed already. ' +
                    'In that case you can use a list.')

            def loglike(params):
                return sum(logl(params) for logl in logls)

            loglike.params = sorted(set.union(*[set(logl.params) for logl in logls]))
        else:
            if isinstance(self.globalgp.Phi, metamath.NoiseMatrix):
                Ns, self.ys = zip(*[(psl.N, psl.y) for psl in self.psls])
                self.globalgp.Phi.inv = getattr(self.globalgp, 'Phi_inv', None)
                self.gsm = metamath.GlobalWoodburyKernel(Ns, self.globalgp.Fs, self.globalgp.Phi)

                loglike = ffunc(self.gsm.make_kernelproduct(self.ys))
            else:
                P_var_inv = self.globalgp.Phi_inv or self.globalgp.Phi.make_inv()
                kterms = [psl.N.make_kernelterms(psl.y, Fmat) for psl, Fmat in zip(self.psls, self.globalgp.Fs)]

                if len(kterms) == 0:
                    raise ValueError('No PulsarLikelihoods in GlobalLikelihood: ' +
                        'if you provided them using a generator, it may have been consumed already. ' +
                        'In that case you can use a list.')

                # npsr = len(self.globalgp.Fs)
                # ngp = self.globalgp.Fs[0].shape[1]

                kmeans = getattr(self.globalgp, 'means', None)

                def loglike(params):
                    terms = [kterm(params) for kterm in kterms]

                    p0 = sum([term[0] for term in terms])
                    FtNmy = kh.jnp.concatenate([term[1] for term in terms])

                    Pinv, ldP = P_var_inv(params)

                    # for i, term in enumerate(terms):
                    #     Pinv = Pinv.at[i*ngp:(i+1)*ngp,i*ngp:(i+1)*ngp].add(term[2])
                    # cf = kh.jsp.linalg.cho_factor(Pinv)

                    # this seems a bit slower than the .at/.set scheme in plogL below
                    FtNmF = kh.jsp.linalg.block_diag(*[term[2] for term in terms])
                    cf = kh.jsp.linalg.cho_factor(Pinv + FtNmF)

                    logp = p0 + 0.5 * (FtNmy.T @ kh.jsp.linalg.cho_solve(cf, FtNmy) - ldP - 2.0 * kh.jnp.sum(kh.jnp.log(kh.jnp.diag(cf[0]))))

                    if kmeans is not None:
                        # -0.5 a0t.FtNmF.a0 + 0.5 a0t.FtNmF.Sm.FtNmF.a0 + a0t.FtNmy - a0t.FtNmF.Sm.FtNmy
                        # -0.5 (a0t.FtNmF).a0 + (FtNmy)t.a0 + 0.5 (a0t.FtNmF).Sm.FtNmF.a0 - (FtNmy)t.Sm.FtNmF.a0
                        # -0.5 (a0t.FtNmF).(a0 - Sm.FtNmF.a0) + (FtNmy)t.(a0 - Sm.FtNmF.a0)

                        a0 = kmeans(params)
                        FtNmFa0 = FtNmF @ a0
                        logp = logp - (0.5 * FtNmFa0.T - FtNmy.T) @ (a0 - kh.jsp.linalg.cho_solve(cf, FtNmFa0))

                    return logp

                params_kterms = list(set.union(*[set(kterm.params) for kterm in kterms]))
                params_kmeans = kmeans.params if kmeans is not None else []
                loglike.params = sorted(params_kterms + params_kmeans + P_var_inv.params)

        return loglike

    # MPI parallel likelihood
    @functools.cached_property
    def plogL(self):
        import mpi4py
        import mpi4jax

        mpicomm = mpi4py.MPI.COMM_WORLD
        jaxcomm = mpicomm.Clone()

        size = mpicomm.Get_size()
        rank = mpicomm.Get_rank()

        if self.globalgp is None:
            logls = [psl.logL for psl in self.psls]

            def loglike(params):
                slogl = sum(logl(params) for logl in logls)
                slogl, tk = mpi4jax.allreduce(slogl, mpi4py.MPI.SUM, comm=jaxcomm)
                return slogl

            local_list = sorted(set.union(*[set(logl.params) for logl in logls]))
            loglike.params = sorted(set([p for l in mpicomm.allgather(local_list) for p in l]))
        else:
            # handle the case where there are more matrices in self.globalgp than likelihoods
            Fmats = {name: Fmat for name, Fmat in zip(self.globalgp.name, self.globalgp.Fs)}
            kterms = [psl.N.make_kernelterms(psl.y, Fmats[psl.name]) for psl in self.psls]

            if rank == 0:
                npsr = len(self.globalgp.Fs)
                ngp = self.globalgp.Fs[0].shape[1]

                P_var_inv = self.globalgp.Phi_inv or self.globalgp.Phi.make_inv()

                def loglike(params):
                    b0 = kh.jnp.zeros((size,), dtype=kh.jnp.float64)
                    b1 = kh.jnp.zeros((npsr, ngp), dtype=kh.jnp.float64)
                    b2 = kh.jnp.zeros((npsr, ngp, ngp), dtype=kh.jnp.float64)

                    t0, t1, t2 = zip(*[kterm(params) for kterm in kterms])

                    b0 = b0.at[0].set(sum(t0))
                    b1 = b1.at[0::size,:].set(kh.jnp.array(t1))
                    b2 = b2.at[0::size,:,:].set(kh.jnp.array(t2))

                    for i in range(1, size):
                        b, tk = mpi4jax.recv(b0[i], source=i, tag=0, comm=jaxcomm)
                        b0 = b0.at[i].set(b)
                        b, tk = mpi4jax.recv(b1[i::size,:], source=i, tag=1, token=tk, comm=jaxcomm)
                        b1 = b1.at[i::size,:].set(b)
                        b, tk = mpi4jax.recv(b2[i::size,:,:], source=i, tag=2, token=tk, comm=jaxcomm)
                        b2 = b2.at[i::size,:,:].set(b)

                    p0 = kh.jnp.sum(b0)
                    FtNmy = b1.flatten()

                    Pinv, ldP = P_var_inv(params)
                    cf = kh.jsp.linalg.cho_factor(Pinv + kh.jsp.linalg.block_diag(*b2))

                    ret = p0 + 0.5 * (FtNmy.T @ kh.jsp.linalg.cho_solve(cf, FtNmy) - ldP - 2.0 * kh.jnp.sum(kh.jnp.log(kh.jnp.diag(cf[0]))))
                    ret, tk = mpi4jax.bcast(ret, root=0, comm=jaxcomm)

                    return ret

                local_list = P_var_inv.params + sorted(set.union(*[set(kterm.params) for kterm in kterms]))
            else:
                def loglike(params):
                    t0, t1, t2 = zip(*[kterm(params) for kterm in kterms])

                    tk = mpi4jax.send(sum(t0), dest=0, tag=0, comm=jaxcomm)
                    tk = mpi4jax.send(kh.jnp.array(t1), dest=0, tag=1, token=tk, comm=jaxcomm)
                    tk = mpi4jax.send(kh.jnp.array(t2), dest=0, tag=2, token=tk, comm=jaxcomm)

                    ret, tk = mpi4jax.bcast(1.0, root=0, comm=jaxcomm)

                    return ret

                local_list = sorted(set.union(*[set(kterm.params) for kterm in kterms]))

            loglike.params = sorted(set([p for l in mpicomm.allgather(local_list) for p in l]))

        return loglike

    @functools.cached_property
    def sample_conditional(self):
        cond = self.conditional
        index = self.globalgp.index

        def sample_cond(key, params):
            mu, cf = cond(params)

            # conditional normal draws are obtained as `mu + y` after solving `cf.T y = x` for a normal deviate `x`
            key, subkey = kh.jnpsplit(key)
            c = mu + kh.jsp.linalg.solve_triangular(cf[0].T, kh.jnpnormal(subkey, mu.shape), lower=False)

            return key, {par: c[sli] for par, sli in index.items()}

        sample_cond.params = cond.params

        return sample_cond

    @functools.cached_property
    def conditional(self):
        if self.globalgp is None:
            raise ValueError("Nothing to predict in GlobalLikelihood without a globalgp!")
        else:
            P_var_inv = self.globalgp.Phi_inv or self.globalgp.Phi.make_inv()
            ndim = 1 if isinstance(self.globalgp.Phi, metamath.NoiseMatrix1D) else 2

            ksolves = [psl.N.make_kernelsolve(psl.y, Fmat) for psl, Fmat in zip(self.psls, self.globalgp.Fs)]

            if len(ksolves) == 0:
                raise ValueError('No PulsarLikelihoods in GlobalLikelihood: ' +
                    'if you provided them using a generator, it may have been consumed already. ' +
                    'In that case you can use a list.')

            if not ksolves[0].params:
                solves = [ksolve({}) for ksolve in ksolves]
                FtNmy = kh.jnp.concatenate([solve[0] for solve in solves])

                # FtNmF = kh.jsp.linalg.block_diag(*[solve[1] for solve in solves])
                FtNmFs = [solve[1] for solve in solves]
                ngp = FtNmFs[0].shape[0]

                def cond(params):
                    Pinv, _ = P_var_inv(params)

                    Sm = kh.jnp.diag(Pinv) if Pinv.ndim == 1 else Pinv
                    for i, FtNmF in enumerate(FtNmFs):
                        Sm = Sm.at[i*ngp:(i+1)*ngp, i*ngp:(i+1)*ngp].add(FtNmF)

                    cf = kh.jsp.linalg.cho_factor(Sm, lower=True)
                    mu = kh.jsp.linalg.cho_solve(cf, FtNmy)

                    return mu, cf

                cond.params = P_var_inv.params
            else:
                def cond(params):
                    # each solve is a tuple TtSy, TtST
                    solves = [ksolve(params) for ksolve in ksolves]

                    FtNmy = kh.jnp.concatenate([solve[0] for solve in solves])

                    Pinv, _ = P_var_inv(params)

                    # phiinv = (kh.jnp.diag(Pinv) if Pinv.ndim == 1 else Pinv)
                    # tnt = kh.jsp.linalg.block_diag(*[solve[1] for solve in solves])
                    # Sm = phiinv + tnt
                    Sm = (kh.jnp.diag(Pinv) if ndim == 1 else Pinv) + kh.jsp.linalg.block_diag(*[solve[1] for solve in solves])

                    # the variance of the normal is S = Sm^-1; but if we want normal deviates y
                    # with that variance, we can use the Cholesky decomposition
                    # S = L L^T => Sm = L^-T L^-1, and then solve L^-T y = x for randn x
                    # where cf = L^-1. See enterprise/signals/utils.py:ConditionalGP

                    # to get the actual covariance, one would use cho_solve(cf, identity matrix)

                    cf = kh.jsp.linalg.cho_factor(Sm, lower=True)
                    mu = kh.jsp.linalg.cho_solve(cf, FtNmy)

                    return mu, cf
                    # return mu, cf, phiinv, tnt

                cond.params = sorted(set.union(*[set(ksolve.params) for ksolve in ksolves])) + P_var_inv.params

        return cond


class ArrayLikelihood(summary.SummaryMixin):
    def __init__(self, psls, *, commongp=None, globalgp=None, transform=None,
                 decenter=False, extsignals=None, reference=None):
        self.psls = psls
        self.commongp = commongp
        self.globalgp = globalgp
        self.transform = transform
        self.decenter = decenter
        # reference+delta (single-precision Half B, ADR 0001/0003): an optional
        # single central params dict theta_ref. When given, each GP level's prior
        # covariance Phi is frozen ONCE at theta_ref in float64 (the "thin top
        # layer", `_freeze_reference`) and the marginal logL is evaluated as
        # logL_ref + Delta logL so float32 holds the O(1) increment. theta_ref is
        # consumed entirely here -- only the frozen Phi_ref constant leaves reach
        # the kernels/graphs (hard guardrail). reference=None -> today's path.
        self.reference = reference
        # extsignals: deterministic signals on their OWN basis (utils.ExtSignal),
        # for signals needing higher frequencies than the GP bases reach (e.g. a
        # CW); see discovery.deterministic.makecw_extsignal. For same-basis
        # deterministic Fourier signals, use makecommongp_fourier(..., means=...)
        # instead.
        self.extsignals = extsignals

    def _freeze_reference(self, Phi):
        """Thin top layer: evaluate a GP level's prior covariance Phi at the
        reference params self.reference (theta_ref) ONCE, in float64, and return
        it as a frozen constant leaf (a metamath.NoiseMatrix). Its .make_inv
        folds to a float64 (Phi_ref^-1, logdet Phi_ref) constant -- the reference
        baseline the refdelta graphs expand around. The covariance (not the
        inverse) is frozen (ADR 0001), so a non-sampled sub-component
        self-cancels (Delta = 0). theta_ref is consumed entirely here; only the
        frozen leaf reaches the kernels/graphs (hard guardrail).
        """
        getN = getattr(Phi, 'getN', None)
        if getN is None:
            getN = Phi.N
        arr = getN(self.reference) if callable(getN) else getN
        return metamath.NoiseMatrix(kh.jnp.asarray(arr))

    @functools.cached_property
    def conditional(self):
        # eventually move to constructor
        if self.commongp is None or self.globalgp is not None:
            raise ValueError("ArrayLikelihood.conditional currently only works with commongp.")

        if not hasattr(self, 'vsm'):
            commongp = metamath.CompoundGP(self.commongp)
            Ns, self.ys = zip(*[(psl.N, psl.y) for psl in self.psls])
            self.vsm = metamath.VectorWoodburyKernel(Ns, commongp.F, commongp.Phi)
            self.vsm.index = getattr(commongp, 'index', None)
            self.vsm.means = getattr(commongp, 'means', None)

        if hasattr(self.vsm, 'make_conditional'):
            return ffunc(self.vsm.make_conditional(self.ys))
        else:
            raise NotImplementedError('No ArrayLikelihood.conditional with this setup so far.')

    @functools.cached_property
    def sample_conditional(self):
        cond = self.conditional
        index = self.vsm.index

        def sample_cond(key, params):
            mu, cf = cond(params)

            key, subkey = kh.jnpsplit(key)
            c = mu + kh.jsp.linalg.solve_triangular(jax.numpy.transpose(cf[0], axes=(0,2,1)),
                                                        kh.jnpnormal(subkey, mu.shape), lower=False)

            # TODO: handling of indices is not consistent with GlobalLikelihood, returning only pulsars here
            return key, {psl.name: ci for psl, ci in zip(self.psls, c)}

        sample_cond.params = cond.params

        return sample_cond

    @functools.cached_property
    def clogL(self):
        if self.commongp is None and self.globalgp is None:
            def loglike(params):
                return sum(psl.clogL(params) for psl in self.psls)
            loglike.params = sorted(set.union(*[set(psl.clogL.params) for psl in self.psls]))

            return loglike
        elif self.commongp is None:
            raise NotImplementedError("ArrayLikelihood does not support a globalgp without a commongp")
        elif self.globalgp is None:
            commongp = metamath.CompoundGP(self.commongp)
        else:
            cgp = self.commongp if isinstance(self.commongp, list) else [self.commongp]
            commongp = metamath.CompoundGP(cgp + [self.globalgp])

        Ns, self.ys = zip(*[(psl.N, psl.y) for psl in self.psls])

        # Both this line and the decentering code below assume N and F are constants.
        self.vsm = metamath.VectorWoodburyKernel(Ns, commongp.F, commongp.Phi)

        if self.decenter:
            # Build a decentering reparam closure. Precomputes per-pulsar
            # NmF / FtNmF / NmFty at trace time (constant N assumption).
            # All metamath kernels expose `make_solve` as a graph and a
            # CompoundGP's `.F` may be either an array or a graph dict;
            # materialize each via `mm.func(...)({}, params={})`.
            def _solve_2d(N, F):
                return metamatrix.func(N.make_solve)(F, params={})

            def _eval_F(F):
                if isinstance(F, dict):
                    return kh.jnp.asarray(metamatrix.func(F)(params={}))
                return kh.jnp.asarray(F)

            vsm_Fs = [_eval_F(F) for F in self.vsm.Fs]
            NmFs, ldNs = zip(*[_solve_2d(N, F) for N, F in zip(self.vsm.Ns, vsm_Fs)])
            FtNmFs = [F.T @ NmF for F, NmF in zip(vsm_Fs, NmFs)]
            NmFtys = [NmF.T @ y for NmF, y in zip(NmFs, self.ys)]
            FtNmF, NmFty = kh.jnparray(FtNmFs), kh.jnparray(NmFtys)

            def decenter_transform(params, c):
                cgp_list = (self.commongp if isinstance(self.commongp, list)
                            else [self.commongp])
                phis_invs_commongp = [gp.Phi.getN(params)**-1 for gp in cgp_list]
                if self.globalgp is not None:
                    # decenter using CURN: just the diagonal of the globalgp Phi
                    phis_invs_globalgp = (kh.jnp.diag(
                        self.globalgp.Phi.getN(params)**-1
                    ).reshape((len(self.psls), -1)))
                    phis_invs = kh.jnp.concatenate(
                        [*phis_invs_commongp, phis_invs_globalgp], axis=1)
                else:
                    phis_invs = kh.jnp.concatenate([*phis_invs_commongp], axis=1)
                i1, i2 = kh.jnp.diag_indices(phis_invs.shape[1], ndim=2)

                cf = kh.matrix_factor(FtNmF.at[:, i1, i2].add(phis_invs), lower=True)
                am = kh.jsp.linalg.solve_triangular(
                    cf[0], c, trans=1, lower=cf[1])
                mus = kh.matrix_solve(cf, NmFty)
                # Jacobian of f^-1 wrt xi: |L|; cf[0] is L^-1.
                ldL = -kh.jnp.logdet(cf[0][:, i1, i2])

                return am + mus, ldL
            decenter_transform.params = []

        if hasattr(commongp, 'prior'):
            self.vsm.prior = commongp.prior
        if hasattr(commongp, 'index'):
            self.vsm.index = commongp.index
        # propagate commongp.means so the GP prior is centered on a0 when set
        self.vsm.means = getattr(commongp, 'means', None)

        # reparam stage: bijections on the GP coefficients; Jacobians compose.
        reparams = []
        if self.decenter:
            reparams.append(decenter_transform)
        if self.transform is not None:
            reparams.extend(self.transform if isinstance(self.transform, (list, tuple))
                            else [self.transform])

        loglike = self.vsm.make_kernelproduct_gpcomponent(
            self.ys, transform=reparams, extsignals=self.extsignals)

        # metamath.VectorWoodburyKernel returns a graph; matrix.py still
        # returns a callable. ffunc converts a graph to a `(params) -> ...`
        # callable at the outer boundary; for an already-callable result it's
        # a no-op.
        return ffunc(loglike)

    @functools.cached_property
    def logL(self):
        if self.commongp is None:
            if self.globalgp is None:
                logls = [psl.logL for psl in self.psls]

                def loglike(params):
                    return sum(logl(params) for logl in logls)
                loglike.params = sorted(set.union(*[set(logl.params) for logl in logls]))
                loglike.graphs = [logl.graph for logl in logls if hasattr(logl, 'graph')]

                return loglike
            else:
                raise NotImplementedError("Currently ArrayLikelihood does not support a globalgp without a commongp")

        commongp = metamath.CompoundGP(self.commongp)

        Ns, self.ys = zip(*[(psl.N, psl.y) for psl in self.psls])
        self.vsm = metamath.VectorWoodburyKernel(Ns, commongp.F, commongp.Phi)
        self.vsm.index = getattr(commongp, 'index', None)
        self.vsm.means = getattr(commongp, 'means', None)

        # reference+delta opt-in: freeze the inner (commongp) prior at theta_ref.
        # The kernel routes to the refdelta twin only when this leaf is present.
        if self.reference is not None:
            self.vsm.P_ref = self._freeze_reference(commongp.Phi)

        if self.globalgp is None:
            loglike = ffunc(self.vsm.make_kernelproduct(self.ys))
        else:
            if isinstance(self.globalgp.Phi, metamath.NoiseMatrix):
                Ns, self.ys = zip(*[(psl.N, psl.y) for psl in self.psls])
                self.gsm = metamath.GlobalWoodburyKernel(self.vsm, self.globalgp.Fs, self.globalgp.Phi)

                # reference+delta opt-in: freeze the outer (globalgp) prior too.
                # With both inner (self.vsm.P_ref) and outer references present the
                # fused kernel routes to the two-level refdelta twins.
                if self.reference is not None:
                    self.gsm.P_ref = self._freeze_reference(self.globalgp.Phi)

                loglike = ffunc(self.gsm.make_kernelproduct(self.ys))
            else:
                P_var_inv = self.globalgp.Phi_inv or self.globalgp.Phi.make_inv()
                kterms = self.vsm.make_kernelterms(self.ys, self.globalgp.Fs)

                npsr = len(self.globalgp.Fs)
                ngp = self.globalgp.Fs[0].shape[1]

                kmeans = getattr(self.globalgp, 'means', None)

                def loglike(params):
                    terms = kterms(params)

                    p0 = kh.jnp.sum(terms[0])
                    FtNmy = terms[1].reshape(npsr * ngp)

                    Pinv, ldP = P_var_inv(params)

                    # alternatives to block_diag (with similar runtimes on CPU, slower on GPU)
                    # for i in range(npsr):
                    #    Pinv = Pinv.at[i*ngp:(i+1)*ngp,i*ngp:(i+1)*ngp].add(terms[2][i,:,:])
                    #    cf = kh.jsp.linalg.cho_factor(Pinv)
                    #
                    #    Pinv = jax.lax.fori_loop(0, npsr,
                    #               lambda i, Pinv: jax.lax.dynamic_update_slice(Pinv,
                    #                   jax.lax.dynamic_slice(Pinv, (i*ngp,i*ngp), (ngp,ngp)) +
                    #                   jax.lax.squeeze(jax.lax.dynamic_slice(terms[2], (i,0,0), (1,ngp,ngp)), [0]),
                    #                   (i*ngp,i*ngp)),
                    #               Pinv)
                    #    cf = kh.jsp.linalg.cho_factor(Pinv)

                    FtNmF = kh.jsp.linalg.block_diag(*terms[2])
                    cf = kh.matrix_factor(Pinv + FtNmF)

                    logp = p0 + 0.5 * (FtNmy.T @ kh.matrix_solve(cf, FtNmy) - ldP - kh.matrix_norm * kh.jnp.sum(kh.jnp.log(kh.jnp.diag(cf[0]))))

                    if kmeans is not None:
                        a0 = kmeans(params)
                        FtNmFa0 = FtNmF @ a0
                        logp = logp - (0.5 * FtNmFa0.T - FtNmy.T) @ (a0 - kh.jsp.linalg.cho_solve(cf, FtNmFa0))

                    return logp

                params_kmeans = kmeans.params if kmeans is not None else []
                loglike.params = sorted(kterms.params + params_kmeans + P_var_inv.params)

        return loglike

    def cglogL(self, cgmaxiter=100, make_logdet='CG-MDL', detmatvecs=5, detsamples=200, clip=None):
        commongp = metamath.CompoundGP(self.commongp)

        Ns, self.ys = zip(*[(psl.N, psl.y) for psl in self.psls])
        self.vsm = metamath.VectorWoodburyKernel(Ns, commongp.F, commongp.Phi)
        self.vsm.index = getattr(commongp, 'index', None)

        if self.globalgp is None:
            loglike = self.vsm.make_kernelproduct(self.ys)
        else:
            factors = self.globalgp.factors
            kterms = self.vsm.make_kernelterms(self.ys, self.globalgp.Fs)

            npsr = len(self.globalgp.Fs)
            ngp = self.globalgp.Fs[0].shape[1]

            logdet_estimator = kh.make_logdet_estimator(npsr * ngp, detmatvecs, detsamples, clip)
            rndkey = jax.random.PRNGKey(1)

            def loglike(params):
                terms = kterms(params)

                p0, FtNmy, FtNmF = kh.jnp.sum(terms[0]), terms[1], terms[2]

                # get Cholesky factors of orf and phi matrices
                orfcf, phicf = factors(params)

                # compute log Phi (not needed for Gseries)
                ldP = (npsr * 2.0 * kh.jnp.sum(kh.jnp.log(kh.jnp.diag(phicf[0]))) +
                       ngp  * 2.0 * kh.jnp.sum(kh.jnp.log(kh.jnp.diag(orfcf[0]))))

                # reconstruct the inverse matrices
                orfinv = kh.jsp.linalg.cho_solve(orfcf, kh.jnp.eye(npsr))
                phiinv = kh.jsp.linalg.cho_solve(phicf, kh.jnp.eye(ngp))

                # define a preconditioner solve M^-1 y with block-diag M_i = FtNmF_i + orfinv[i,i] phi
                precf = kh.jsp.linalg.cho_factor(FtNmF + kh.jnp.diag(orfinv)[:, None, None] * phiinv[None, :, :])
                def precond(FtNmy):
                    return kh.jsp.linalg.cho_solve(precf, FtNmy)

                # define the application of Gamma^-1 x phi^-1 + FtNmF to a "vector" FtNmy (npsr, ngp)
                def matvec(FtNmy):
                    return (kh.jsp.linalg.cho_solve(orfcf, kh.jsp.linalg.cho_solve(phicf, FtNmy.T).T) +
                            kh.jnp.squeeze(FtNmF @ FtNmy[..., None])) # kh.jnp.einsum('kij,kj->ki', FtNmF, FtNmy))

                sol = kh.cgsolve(matvec, FtNmy, M=precond, maxiter=cgmaxiter)

                jnp, jspa = kh.jnp, kh.jsp.linalg

                if make_logdet == 'G-series':
                    # expand in G
                    # log |Phi| = m log |Gamma^-1| + n log |phi^-1| + sum_i Gamma_ii Tr (phi G_i)
                    #                                                 - 1/2 sum_i Gamma_ii^2 Tr (phi G_i)^2
                    #                                                 + 1/3 sum_i Gamma_ii^3 Tr (phi G_i)^3
                    # furthermore the first term cancels with ldP, so ldP not needed

                    phiG = phicf[0].T @ (phicf[0] @ FtNmF)
                    orfdiag = kh.jnp.diag(orfcf[0].T @ orfcf[0])
                    logdet = (orfdiag @ kh.jnp.trace(phiG, axis1=1, axis2=2)
                            -0.5 * orfdiag**2 @ jax.numpy.trace(phiG @ phiG, axis1=1, axis2=2)
                            +(1/3.0) * orfdiag**3 @ jax.numpy.trace(phiG @ phiG @ phiG, axis1=1, axis2=2))

                    return p0 + 0.5 * (kh.jnp.sum(FtNmy * sol) - logdet)
                elif make_logdet == 'D-series':
                    # let Phi = D + B with D diagonal
                    # then log |D + B| = log |D| - 1/2 Tr((D^-1 B)^2) + 1/3 Tr((D^-1 B)^3) - ...
                    # (first order Tr(D^-1 B) vanishes)

                    cfD = jspa.cho_factor(kh.jnp.diag(orfinv)[:,None,None] * phiinv[None,:,:] + FtNmF)
                    i1, i2 = kh.jnp.diag_indices(ngp, ndim=2)
                    logD = 2.0 * kh.jnp.sum(kh.jnp.log(kh.jnp.abs(cfD[0][:, i1, i2])))

                    E = jax.vmap(lambda c, m: jspa.cho_solve((c, False), m), in_axes=(0, None))(cfD[0], phiinv)

                    traces = kh.jnp.einsum('nij,mji->nm', E, E)
                    gamma_prod = orfinv * orfinv.T
                    off_diag_mask = ~kh.jnp.eye(npsr, dtype=bool)

                    traces3 = kh.jnp.einsum('aij,bjk,ckl->abc', E, E, E)
                    gamma_prod3 = kh.jnp.einsum('ij,jk,ki->ijk', orfinv, orfinv, orfinv)
                    i_idx, j_idx, k_idx = kh.jnp.meshgrid(kh.jnp.arange(npsr), kh.jnp.arange(npsr), kh.jnp.arange(npsr), indexing="ij")
                    off_diag_mask3 = (i_idx != j_idx) & (j_idx != k_idx) & (k_idx != i_idx)

                    logdet = logD - 0.5 * kh.jnp.sum(gamma_prod * traces * off_diag_mask) + (1/3.0) * kh.jnp.sum(gamma_prod3 * traces3 * off_diag_mask3)

                    return p0 + 0.5 * (kh.jnp.sum(FtNmy * sol) - ldP - logdet)
                elif make_logdet == 'CG-MDL':
                    # Lanczos-Hutchinson for log |K + F Phi F^T| = log |K| + log |I + F^T K^{-1} F Phi|

                    def detmatvec(y):
                        Y = y.reshape((npsr, ngp))
                        AY = kh.jnp.einsum('akl,al->ak',
                                        FtNmF, kh.jnp.einsum('ab,bc,cl->al',
                                                          orfcf[0].T, orfcf[0], kh.jnp.einsum('li,ij,aj->al',
                                                                                           phicf[0].T, phicf[0], Y))) + Y
                        return AY.reshape(npsr * ngp)

                    logdet = logdet_estimator(detmatvec, rndkey)

                    return p0 + 0.5 * (kh.jnp.sum(FtNmy * sol) - logdet)
                elif make_logdet == 'CG-Woodbury':
                    # Lanczos-Hutchinson for Sigma with preconditioner

                    i1, i2 = kh.jnp.diag_indices(precf[0].shape[1], ndim=2)
                    logpre = 2.0 * kh.jnp.sum(kh.jnp.log(kh.jnp.abs(precf[0][:, i1, i2])))

                    def prematvec(y):
                        Y = y.reshape((npsr, ngp))
                        AY = jspa.cho_solve(precf, matvec(Y))
                        return AY.reshape(npsr * ngp)

                    logdet = logdet_estimator(prematvec, rndkey) + logpre

                    return p0 + 0.5 * (kh.jnp.sum(FtNmy * sol) - ldP - logdet)
                else:
                    raise ValueError("Unknown logdet method: {}".format(make_logdet))

            loglike.params = sorted(kterms.params + factors.params)

        return loglike
