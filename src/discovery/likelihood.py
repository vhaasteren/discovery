import functools

import numpy as np

from . import matrix

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
# ShermanMorrisonKernel can return a ConstantKernel or a VariableKernel

# npta = je.PulsarLikelihood(je.residuals(psr),
#                            je.makenoise_measurement(psr, noisedict),
#                            je.makegp_ecorr(psr, noisedict),
#                            je.makegp_timing(psr),
#                            je.makegp_fourier('red_noise', psr, je.powerlaw, 10),
#                            je.makegp_fourier('red_noise', psr,
#                                              je.makepowerlaw_crn(5), 10, T=Tspan,
#                                              common=['crn_log10_A', 'crn_gamma']),
#                            concat=True)

class PulsarLikelihood:
    def __init__(self, args, concat=True):
        y     = [arg for arg in args if isinstance(arg, np.ndarray)]
        delay = [arg for arg in args if callable(arg)]
        noise = [arg for arg in args if isinstance(arg, matrix.Kernel)]
        cgps  = [arg for arg in args if isinstance(arg, matrix.ConstantGP)]
        vgps  = [arg for arg in args if isinstance(arg, matrix.VariableGP)]

        if len(y) > 1 or len(noise) > 1:
            raise ValueError("Only one residual vector and one noise Kernel allowed.")
        elif len(noise) == 0:
            raise ValueError("I need exactly one noise Kernel.")

        noise, y = noise[0], y[0]

        if cgps:
            if concat:
                cgp = matrix.CompoundGP(cgps)
                csm = matrix.ShermanMorrisonKernel(noise, cgp.F, cgp.Phi)
            else:
                csm = noise
                for cgp in cgps:
                    csm = matrix.ShermanMorrisonKernel(csm, cgp.F, cgp.Phi)
        else:
            csm = noise

        if vgps:
            if concat:
                vgp = matrix.CompoundGP(vgps)
                vsm = matrix.ShermanMorrisonKernel(csm, vgp.F, vgp.Phi)
            else:
                vsm = csm
                for vgp in vgps:
                    vsm = matrix.ShermanMorrisonKernel(vsm, vgp.F, vgp.Phi)
        else:
            vsm = csm

        if len(delay) > 0:
            delay = matrix.CompoundDelay(delay)

        self.y, self.delay, self.N = y, delay, vsm

    @functools.cached_property
    def logL(self):
        if self.delay:
            return self.N.make_kernel(self.y, self.delay)
        else:
            return self.N.make_kernelproduct(self.y)

    @functools.cached_property
    def sample(self):
        return self.N.make_sample()


class GlobalLikelihood:
    def __init__(self, psls, globalgp=None):
        self.psls = psls
        self.globalgp = matrix.CompoundGlobalGP(globalgp) if isinstance(globalgp, list) else globalgp

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

            Fs = [matrix.jnparray(F) for F in self.globalgp.Fs]

            i0, slcs = 0, []
            for F in self.globalgp.Fs:
                slcs.append(slice(i0, i0 + F.shape[1]))
                i0 = i0 + F.shape[1]

            def sampler(key, params):
                key, c = Phi_sample(key, params)

                ys = []
                for sl, F, slc in zip(sls, Fs, slcs):
                    key, y = sl(key, params)
                    ys.append(y + matrix.jnp.dot(F, c[slc]))

                # ys = [key, _ := sl(key, params) + jnp.dot(F, c[slc]) for sl, F, slc in zip(sls, Fs, slcs)]
                return key, ys

            sampler.params = sorted(set.union(*[set(sl.params) for sl in sls])) + Phi_sample.params

        return sampler

    # design the OS
    # rho_ij = y_i Cmi(theta) Fi Phi Ftj Cmj(theta) y_j
    # sigma_ij = tr Fti Cmi(theta) Fi Phi Ftj Cmj(theta) Fj Phi
    @functools.cached_property
    def os(self):
        if self.globalgp is None:
            raise ValueError('GlobalLikelihood.os needs a globalgp.')
        else:
            pairs, angles = self.globalgp.pairs, self.globalgp.angles
            kernelsolves = [psl.N.make_kernelsolve(psl.y, Tmat) for (psl, Tmat) in zip(self.psls, self.globalgp.F)]
            # components = self.globalgp.Fs[0].shape[1]
            getsN = self.globalgp.Phi.make_sqrt()

            def rhosigma(params):
                sN = getsN(params)
                ks = [k(params) for k in kernelsolves]

                ts = [matrix.jnp.dot(sN * ks[i][0], sN * ks[j][0]) for (i,j) in pairs]

                ds = [sN[:,matrix.jnp.newaxis] * k[1] * sN[matrix.jnp.newaxis,:] for k in ks]
                bs = [matrix.jnp.trace(ds[i] @ ds[j]) for (i,j) in pairs]

                return (matrix.jnp.array(ts) / matrix.jnp.array(bs),
                        1.0 / matrix.jnp.sqrt(matrix.jnp.array(bs)))

            rhosigma.params = sorted(set.union(*[set(k.params) for k in kernelsolves], getsN.params))
            rhosigma.pairs = pairs
            rhosigma.angles = angles

            return rhosigma

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
            P_var_inv = self.globalgp.Phi_inv or self.globalgp.Phi.make_inv()
            kterms = [psl.N.make_kernelterms(psl.y, Fmat) for psl, Fmat in zip(self.psls, self.globalgp.Fs)]
            if len(kterms) == 0:
                raise ValueError('No PulsarLikelihoods in GlobalLikelihood: ' +
                    'if you provided them using a generator, it may have been consumed already. ' +
                    'In that case you can use a list.')

            def loglike(params):
                terms = [kterm(params) for kterm in kterms]

                p0 = sum([term[0] for term in terms])
                FtNmy = matrix.jnp.concatenate([term[1] for term in terms])

                Pinv, ldP = P_var_inv(params)
                cf = matrix.jsp.linalg.cho_factor(Pinv + matrix.jsp.linalg.block_diag(*[term[2] for term in terms]))

                return p0 + 0.5 * (FtNmy.T @ matrix.jsp.linalg.cho_solve(cf, FtNmy) - ldP - 2.0 * matrix.jnp.sum(matrix.jnp.log(matrix.jnp.diag(cf[0]))))

            loglike.params = sorted(set.union(*[set(kterm.params) for kterm in kterms])) + P_var_inv.params

        return loglike

    @functools.cached_property
    def sample_conditional(self):
        cond = self.conditional
        index = self.globalgp.index

        def sample_cond(key, params):
            mu, cf = cond(params)

            # conditional normal draws are obtained as `mu + y` after solving `cf.T y = x` for a normal deviate `x`
            key, subkey = matrix.jnpsplit(key)
            c = mu + matrix.jsp.linalg.solve_triangular(cf[0].T, matrix.jnpnormal(subkey, mu.shape), lower=False)

            return key, {par: c[sli] for par, sli in index.items()}

        sample_cond.params = cond.params

        return sample_cond

    @functools.cached_property
    def conditional(self):
        if self.globalgp is None:
            raise ValueError("Nothing to predict in GlobalLikelihood without a globalgp!")
        else:
            P_var_inv = self.globalgp.Phi_inv or self.globalgp.Phi.make_inv()
            ksolves = [psl.N.make_kernelsolve(psl.y, Fmat) for psl, Fmat in zip(self.psls, self.globalgp.Fs)]
            if len(ksolves) == 0:
                raise ValueError('No PulsarLikelihoods in GlobalLikelihood: ' +
                    'if you provided them using a generator, it may have been consumed already. ' +
                    'In that case you can use a list.')

            def cond(params):
                # each solve is a tuple TtSy, TtST
                solves = [ksolve(params) for ksolve in ksolves]

                FtNmy = matrix.jnp.concatenate([solve[0] for solve in solves])

                Pinv, _ = P_var_inv(params)

                # phiinv = (matrix.jnp.diag(Pinv) if Pinv.ndim == 1 else Pinv)
                # tnt = matrix.jsp.linalg.block_diag(*[solve[1] for solve in solves])
                # Sm = phiinv + tnt
                Sm = (matrix.jnp.diag(Pinv) if Pinv.ndim == 1 else Pinv) + matrix.jsp.linalg.block_diag(*[solve[1] for solve in solves])

                # the variance of the normal is S = Sm^-1; but if we want normal deviates y
                # with that variance, we can use the Cholesky decomposition
                # S = L L^T => Sm = L^-T L^-1, and then solve L^-T y = x for randn x
                # where cf = L^-1. See enterprise/signals/utils.py:ConditionalGP

                # to get the actual covariance, one would use cho_solve(cf, identity matrix)

                cf = matrix.jsp.linalg.cho_factor(Sm, lower=True)
                mu = matrix.jsp.linalg.cho_solve(cf, FtNmy)

                return mu, cf
                # return mu, cf, phiinv, tnt

            cond.params = sorted(set.union(*[set(ksolve.params) for ksolve in ksolves])) + P_var_inv.params

        return cond
