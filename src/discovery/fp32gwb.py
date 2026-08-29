"""Fully-fp32, Cholesky-only GWB likelihood: red noise marginalized per pulsar, GWB
coefficients decentered against the live conditional, exact HD prior.

Model (per pulsar p, whitened by the frozen white-noise kernel K_p, timing model
projected out exactly):

    y = M eps + F a_p + F_E a_E,p + n,   n ~ N(0, K_p),  eps ~ flat,
    a_p ~ N(0, diag(phi_RN,p(theta))),   a_E ~ N(0, Gamma (x) diag(phi_GW(theta)))

``a_p`` (red noise, ``2*components`` per pulsar) is integrated out analytically;
``a_E`` (GWB, ``2*components_gw`` per pulsar, on the first columns of the same
Fourier basis) is sampled through the decentering ``a_E = mu(theta) + L(theta)^-T xi``.

Numerics (the whole point of this module; see docs/advanced/fp32_gwb.md):

* Bake once in float64: closed-form ``K^-1/2`` (diagonal + orthogonal-column ECORR),
  QR projection of the whitened timing basis, SVD of the whitened projected Fourier
  basis ``B_perp = U S V^T``.  Directions whose maximum possible information
  ``sigma^2 * phi_ceiling`` is below ``info_tol`` nats carry no likelihood dependence;
  their coordinates integrate out EXACTLY under the prior (the marginal prior on the
  data range is ``Phi_r = V_r^T Phi V_r``).  No basis is dropped from the model.
* Per evaluation only ``I + (bounded PSD)`` matrices are factorized:
  ``M2 = I + S Phi_r S`` (r x r) and ``I + Y~^T Y~`` (2*components_gw square) -- their
  entries are live per-mode SNR^2 and every pivot is >= 1, so a float32 Cholesky is
  accurate.  ``Phi_r`` itself (13-decade spectrum) and any ``I + low-rank`` matrix
  with exact null directions are never formed.  The RN quadratic is
  ``1/2 (||u_r||^2 - ||L2^-1 u_r||^2)`` with ``u_r = U_r^T r_perp`` bounded.
* A soft ceiling on the live per-mode SNR^2, ``phi_f <= kappa / G_ff`` (kappa=1e6):
  a prior broader than that is indistinguishable from flat; it keeps the fp32 range
  at the prior corners.  Units are ns.  ``jax_default_matmul_precision`` must be
  ``'highest'`` on GPU (TF32 matmuls corrupt the Gram entries).
* All theta-independent constants (``-1/2 r^T K^-1 r``, ``log det K``, the
  timing-projection Jacobian, ``2 pi`` factors, ``log det Gamma``) are kept, in
  float64, outside the fp32 path, so the density is usable for evidence work.
"""
import math

import numpy as np
import scipy.sparse as sps

from . import utils as kh
from . import metamath
from .params import Params, make_layout
from .transport import _eval_basis

RENORM = 1e9          # seconds -> nanoseconds
_LN10 = math.log(10.0)


# ---------------------------------------------------------------------------
# whitening: W with W K W^T = I for K = diag(D) + U diag(P) U^T, U 0/1 orthogonal columns
# ---------------------------------------------------------------------------

def _materialize(x):
    if hasattr(x, "N") and not isinstance(x, dict):
        x = x.N
    c, f = metamath._materialize(x)
    return np.asarray(c if f is None else f(params={}), dtype=np.float64)


class _Whitener:
    def __init__(self, D, U=None, P=None):
        self.D = np.asarray(D, dtype=np.float64)
        if np.any(self.D <= 0):
            raise ValueError("fp32gwb: white-noise diagonal must be positive")
        self.isd = 1.0 / np.sqrt(self.D)
        self.logdet = float(np.sum(np.log(self.D)))
        self.U = None
        if U is not None:
            U = np.asarray(U, dtype=np.float64)
            if not np.all((U == 0) | (U == 1)) or np.any(U.sum(axis=1) > 1):
                raise ValueError("fp32gwb: ECORR exposure must be 0/1 with orthogonal columns")
            P = np.asarray(P, dtype=np.float64)
            self.U = sps.csr_matrix(U); self.Ut = self.U.T.tocsr()
            v2 = self.Ut @ (self.isd ** 2)                       # sum_{i in epoch} 1/D_i
            self.beta = ((1.0 + P * v2) ** -0.5 - 1.0) / v2      # (I + P d d^T)^-1/2 = I + beta d d^T
            self.logdet += float(np.sum(np.log1p(P * v2)))       # matrix determinant lemma

    def __call__(self, X):
        X = np.asarray(X, dtype=np.float64); one = X.ndim == 1
        Z = (X[:, None] if one else X) * self.isd[:, None]
        if self.U is not None:
            S = self.Ut @ (Z * self.isd[:, None])
            Z = Z + self.isd[:, None] * (self.U @ (self.beta[:, None] * S))
        return Z[:, 0] if one else Z


def _whitener_from_kernel(kernel):
    """Frozen white-noise kernel -> _Whitener. Accepts a diagonal NoiseMatrix or a
    WoodburyKernel(diagonal, 0/1 exposure, per-epoch variance) as built by
    ``PulsarLikelihood([residuals, makenoise_measurement, makegp_ecorr])``."""
    if isinstance(kernel, metamath.WoodburyKernel):
        return _Whitener(_materialize(kernel.N), np.asarray(kernel.F), _materialize(kernel.P))
    if isinstance(kernel, metamath.NoiseMatrix):
        return _Whitener(_materialize(kernel))
    raise TypeError(f"fp32gwb: unsupported white-noise kernel {type(kernel).__name__}; "
                    "build the PulsarLikelihood from measurement noise (+ ECORR) only, "
                    "without the timing model or red noise")


# ---------------------------------------------------------------------------
# the kernel
# ---------------------------------------------------------------------------

class GWBMarginalFp32:
    """See the module docstring. ``params`` lists the hyperparameter names (Discovery
    conventions, taken from ``commongp`` and ``globalgp``); ``logp(params, xi)`` is the
    named entry point, ``kernel(theta, xi)`` the packed (JIT) one."""

    def __init__(self, psrs, white_kernels, commongp, globalgp, *, extsignals=None, info_tol=1e-6,
                 kappa=1e6, soft_clip_dex=0.5, phi_ceiling=1e-9, phi_floor=1e-18):
        if len(white_kernels) != len(psrs):
            raise ValueError("fp32gwb: one white-noise kernel per pulsar")
        sep = getattr(globalgp, "separable_prior", None)
        if sep is None or getattr(sep, "orf", None) is None:
            raise ValueError("fp32gwb: globalgp needs a separable (ORF x spectrum) Fourier prior")
        self.psrs, self.npsr = list(psrs), len(psrs)
        self.commongp, self.globalgp = commongp, globalgp
        self.info_tol, self.kappa, self.w = float(info_tol), float(kappa), float(soft_clip_dex)
        self.log10_floor, self.log10_ceiling = math.log10(phi_floor), math.log10(phi_ceiling)
        self.getN_rn, self.spectrum_gw = commongp.Phi.getN, sep.spectrum
        # deterministic signals on their own bases: objects with .Fs (per pulsar, ntoa x k_ext)
        # and .coeffs(params) -> (npsr, k_ext) [seconds]; coefficients may also be passed explicitly.
        self.extsignals = list(extsignals or [])
        for ext in self.extsignals:
            if len(ext.Fs) != self.npsr:
                raise ValueError(f"fp32gwb: ExtSignal {getattr(ext, 'name', '?')} basis count != pulsars")
        ext_params = set().union(*(set(getattr(ext, "params", [])) for ext in self.extsignals))
        self.params = sorted(set(self.getN_rn.params) | set(sep.params) | ext_params)
        self.k = int(np.asarray(_eval_basis(commongp.F[0])).shape[1])
        self.kg = int(sep.width)
        if self.kg > self.k:
            raise ValueError("fp32gwb: the GW basis must be a leading subset of the red-noise basis")

        self.Gam = np.asarray(sep.orf, dtype=np.float64)
        self.Lgam = np.linalg.cholesky(self.Gam)
        self._bake(white_kernels)

    # ---- bake (float64) ----------------------------------------------------
    def _bake(self, white_kernels):
        n, K, kg = self.npsr, self.k, self.kg
        baked = []
        for i, (psr, kern) in enumerate(zip(self.psrs, white_kernels)):
            W = _whitener_from_kernel(kern)
            y = np.asarray(psr.residuals, dtype=np.float64)
            M = np.asarray(psr.Mmat, dtype=np.float64)
            F = np.asarray(_eval_basis(self.commongp.F[i]), dtype=np.float64)
            FE = np.asarray(_eval_basis(self.globalgp.Fs[i]), dtype=np.float64)
            if F.shape[1] != K or not np.allclose(F[:, :kg], FE, rtol=0, atol=1e-10 * np.max(np.abs(F))):
                raise ValueError(f"fp32gwb: GW basis of {psr.name} is not the leading columns of its red-noise basis")
            Mu = np.linalg.svd(M, full_matrices=False)[0]          # orthonormal timing basis (span only matters)
            A = W(Mu); Q, RA = np.linalg.qr(A)
            B = W(F); Bp = B - Q @ (Q.T @ B)
            yw = W(y); rp = yw - Q @ (Q.T @ yw)
            const = (-0.5 * float(rp @ rp) - 0.5 * W.logdet
                     - 0.5 * float(2.0 * np.sum(np.log(np.abs(np.diag(RA)))))
                     - 0.5 * (len(y) - Mu.shape[1]) * math.log(2 * math.pi))
            Us, sig, Vt = np.linalg.svd(Bp, full_matrices=False)
            sig_ns = sig / RENORM
            info = sig_ns ** 2 * 10.0 ** (self.log10_ceiling + 18.0)
            r = int(np.sum(info >= self.info_tol))
            G0 = (Bp.T @ Bp) / RENORM ** 2
            exts = []
            for ext in self.extsignals:
                Fx = np.asarray(_eval_basis(ext.Fs[i]), dtype=np.float64)
                Cx = W(Fx); Cx = Cx - Q @ (Q.T @ Cx)                                  # (W F_ext)_perp
                Cxr = Us[:, :r].T @ Cx                                                  # range part of the whitened basis
                Cxn = Cx - Us[:, :r] @ Cxr                                              # part outside the data range
                # The range part of a deterministic signal is absorbed into u (data); only the
                # out-of-range remainder keeps explicit quadratic/linear terms. Splitting at bake
                # time (float64) removes an O(signal power) cancellation from the fp32 hot path.
                exts.append(dict(Cx=Cxr / RENORM,                                       # [1/ns]
                                 Gx=(Cxn.T @ Cxn) / RENORM ** 2, gx=(Cxn.T @ rp) / RENORM,
                                 Gx_full=(Cx.T @ Cx) / RENORM ** 2, gx_full=(Cx.T @ rp) / RENORM,
                                 Bx=(Bp.T @ Cx) / RENORM ** 2))                        # full grams: reference only
            baked.append(dict(r=r, V=Vt[:r].T, sig=sig_ns[:r], u=Us[:, :r].T @ rp, const=const,
                              info_dropped=info[r:].tolist(), G0=G0, b0=(Bp.T @ rp) / RENORM, exts=exts))
        R = max(b["r"] for b in baked)
        self.ranks = [b["r"] for b in baked]
        self.info_dropped = [b["info_dropped"] for b in baked]
        self.G0 = np.stack([b["G0"] for b in baked]); self.b0 = np.stack([b["b0"] for b in baked])
        V = np.zeros((n, K, R)); sig = np.zeros((n, R)); u = np.zeros((n, R))
        for i, b in enumerate(baked):
            V[i, :, :b["r"]] = b["V"]; sig[i, :b["r"]] = b["sig"]; u[i, :b["r"]] = b["u"]
        self._ext_np = []
        for j, ext in enumerate(self.extsignals):
            kx = baked[0]["exts"][j]["Gx"].shape[0]
            Cx = np.zeros((n, R, kx)); Gx = np.zeros((n, kx, kx)); gx = np.zeros((n, kx)); Bx = np.zeros((n, K, kx))
            Gxf = np.zeros((n, kx, kx)); gxf = np.zeros((n, kx))
            for i, b in enumerate(baked):
                e = b["exts"][j]; Cx[i, :b["r"]] = e["Cx"]; Gx[i] = e["Gx"]; gx[i] = e["gx"]; Bx[i] = e["Bx"]
                Gxf[i] = e["Gx_full"]; gxf[i] = e["gx_full"]
            self._ext_np.append(dict(Cx=Cx, Gx=Gx, gx=gx, Bx=Bx, Gx_full=Gxf, gx_full=gxf))
        Gd = np.einsum("pii->pi", self.G0) * RENORM ** 2                 # per-column data precision [1/s^2]
        self._np = dict(
            V=V, sig=sig, u=u, E=V[:, :kg, :].transpose(0, 2, 1), Lgam=self.Lgam,
            ceil_rn=np.minimum(self.log10_ceiling, np.log10(self.kappa / Gd)) if np.isfinite(self.kappa) else np.full(Gd.shape, self.log10_ceiling),
            ceil_gw=np.minimum(self.log10_ceiling, np.log10(self.kappa / Gd[:, :kg].max(axis=0))) if np.isfinite(self.kappa) else np.full(kg, self.log10_ceiling),
        )
        self.const = float(sum(b["const"] for b in baked)
                           - 0.5 * kg * np.linalg.slogdet(self.Gam)[1]
                           - 0.5 * n * kg * math.log(2 * math.pi))
        self.rmax = R
        # constants are materialized eagerly (outside any JAX trace) for both dtypes
        self._const_cache = {}
        for dt in (kh.jnp.float32, kh.jnp.float64) if kh.jax.config.x64_enabled else (kh.jnp.float32,):
            self._const_cache[kh.jnp.dtype(dt)] = self._make_constants(dt)
        self.theta_layout, self.theta_size = make_layout(self.params)
        self.xi_shape = (n, kg)
        self.kernel = self._build_kernel()

    def _make_constants(self, dtype):
        c = {k: kh.jnp.asarray(v, dtype) for k, v in self._np.items()}
        c["const"] = kh.jnp.asarray(self.const, kh.jnp.float64 if kh.jax.config.x64_enabled else dtype)
        c["ext"] = [{k: kh.jnp.asarray(v, dtype) for k, v in e.items() if k in ("Cx", "Gx", "gx")} for e in self._ext_np]
        return c

    def constants(self, dtype):
        return self._const_cache[kh.jnp.dtype(dtype)]

    # ---- evaluation --------------------------------------------------------
    def _log10phi(self, params, c):
        """Soft floor/ceiling in log10 phi [s^2]; returns ns^2 values ((npsr,k), (kg,))."""
        jnp = kh.jnp
        lp_rn = jnp.log10(jnp.asarray(self.getN_rn(params)))
        lp_gw = jnp.log10(jnp.asarray(self.spectrum_gw(params)))
        w, lo = self.w, self.log10_floor
        def clip(lp, hi):
            if w <= 0:                                   # hard clip (test/oracle mode)
                return kh.jnp.clip(lp, lo, hi)
            lp = lo + w * kh.jax.nn.softplus((lp - lo) / w)
            return hi - w * kh.jax.nn.softplus((hi - lp) / w)
        return clip(lp_rn, c["ceil_rn"]) + 18.0, clip(lp_gw, c["ceil_gw"]) + 18.0

    def ext_coefficients(self, params, dtype):
        """Coefficients of every ExtSignal at ``params``: list of (npsr, k_ext) arrays [ns]."""
        return [kh.jnp.asarray(ext.coeffs(params), dtype) * RENORM for ext in self.extsignals]

    def _eval(self, params, xi, c, with_base=False, ext_coeffs=None, parts=None):
        jnp, jsp = kh.jnp, kh.jsp
        dt = c["V"].dtype; xi = jnp.asarray(xi, dt)
        lp_rn, lp_gw = self._log10phi(params, c)
        # deterministic signals: data -> data - sum_x C_x coeffs_x  (whitened, projected), plus their own terms
        u = c["u"]; ext_term = 0.0
        if self.extsignals:
            coeffs = ext_coeffs if ext_coeffs is not None else self.ext_coefficients(params, dt)
            for e, cc in zip(c["ext"], coeffs):
                cc = jnp.asarray(cc, dt)
                u = u - jnp.einsum("prk,pk->pr", e["Cx"], cc)
                ext_term = ext_term + jnp.sum(cc * e["gx"]) - 0.5 * jnp.einsum("pk,pkl,pl->", cc, e["Gx"], cc)
        phi = jnp.exp(_LN10 * lp_rn).astype(dt); phig = jnp.exp(_LN10 * lp_gw).astype(dt)
        R, kg = self.rmax, self.kg; i1, i2 = jnp.diag_indices(R); j1, j2 = jnp.diag_indices(kg)
        X = jnp.sqrt(phi)[:, :, None] * c["V"]                                   # tall prior factor, Phi_r = X^T X (never formed)
        Wp = X * c["sig"][:, None, :]                                            # live-SNR units
        L2 = jnp.linalg.cholesky(jnp.eye(R, dtype=dt)[None] + jnp.swapaxes(Wp, 1, 2) @ Wp)
        q = jsp.linalg.solve_triangular(L2, u, lower=True)                       # u already has the signals removed
        A = 0.5 * (jnp.sum(c["u"] * c["u"]) - jnp.sum(q * q)) - jnp.sum(jnp.log(L2[:, i1, i2])) + ext_term
        y2 = jsp.linalg.solve_triangular(L2, q, trans=1, lower=True)              # M2^-1 u
        beta = jnp.einsum("pri,pr->pi", c["E"], c["sig"] * y2)                  # (n,kg) [1/ns]
        Y = jsp.linalg.solve_triangular(L2, c["sig"][:, :, None] * c["E"], lower=True)
        sq = jnp.sqrt(phig)
        Yh = Y * sq[None, None, :]
        L3 = jnp.linalg.cholesky(jnp.eye(kg, dtype=dt)[None] + jnp.swapaxes(Yh, 1, 2) @ Yh)
        vE = jsp.linalg.solve_triangular(L3, beta * sq[None, :], lower=True)
        mu = sq[None, :] * jsp.linalg.solve_triangular(L3, vE, trans=1, lower=True)
        aE = mu + sq[None, :] * jsp.linalg.solve_triangular(L3, xi, trans=1, lower=True)     # [ns]
        E = 0.5 * jnp.sum(vE * vE) - jnp.sum(jnp.log(L3[:, j1, j2]))
        z = jsp.linalg.solve_triangular(c["Lgam"], aE, lower=True)
        H = -0.5 * jnp.sum((jnp.sum(z * z, axis=0) - jnp.sum(aE * aE, axis=0)) / phig)
        if parts is not None:
            parts.update(A=A, E=E, H=H, ext=ext_term, u2=0.5 * jnp.sum(c["u"] * c["u"]), q2=0.5 * jnp.sum(q * q),
                         ldL2=jnp.sum(jnp.log(L2[:, i1, i2])), vE2=0.5 * jnp.sum(vE * vE), ldL3=jnp.sum(jnp.log(L3[:, j1, j2])))
        logp = (A + E + H).astype(c["const"].dtype) + c["const"]
        if with_base:
            logp = logp - 0.5 * jnp.sum((xi * xi).astype(logp.dtype))
        return logp, aE / RENORM                                                 # coefficients in seconds

    def _build_kernel(self):
        layout = self.theta_layout
        def evaluate(theta, xi, ext_coeffs=None):
            c = self.constants(theta.dtype)
            return self._eval(dict(Params(theta, layout)), xi, c, ext_coeffs=ext_coeffs)
        return kh.jax.jit(evaluate)

    def logp(self, params, xi, dtype=None, with_base=False, ext_coeffs=None, parts=None):
        """Named-parameter density (decentered), ``(logp, gw_coefficients)``; not jitted.
        ``ext_coeffs`` (list of (npsr, k_ext) arrays in ns) overrides ``ext.coeffs(params)``;
        ``parts`` (dict) receives the kernel's intermediate terms (debugging)."""
        dtype = dtype or kh.working_dtype()
        return self._eval(params, xi, self.constants(dtype), with_base=with_base, ext_coeffs=ext_coeffs, parts=parts)

    def pack(self, params):
        return kh.jnp.concatenate([kh.jnp.asarray(params[name]).reshape(-1) for name, *_ in self.theta_layout])

    def unpack(self, theta):
        return dict(Params(theta, self.theta_layout))

    def diagnostics(self):
        return {"npsr": self.npsr, "k": self.k, "kg": self.kg, "ranks": self.ranks,
                "max_info_dropped_nats": [max(d) if d else 0.0 for d in self.info_dropped],
                "theta_size": self.theta_size, "xi_shape": self.xi_shape, "kappa": self.kappa,
                "info_tol": self.info_tol, "const": self.const}

    # ---- float64 references for tests ---------------------------------------
    def reference_logp(self, params, aE_seconds, ext_coeffs=None):
        """Exact (no deflation) RN-marginalized density at physical GW coefficients [s], numpy float64."""
        c = self.constants(kh.jnp.float64)
        lp_rn, lp_gw = self._log10phi(params, c)
        phi_rn = np.asarray(10.0 ** lp_rn); phi_gw = np.asarray(10.0 ** lp_gw)
        aE = np.asarray(aE_seconds, dtype=np.float64) * RENORM
        coeffs = [np.asarray(x, dtype=np.float64) for x in
                  (ext_coeffs if ext_coeffs is not None else self.ext_coefficients(params, kh.jnp.float64))]
        n, K, kg = self.npsr, self.k, self.kg; out = 0.0
        for p in range(n):
            G, b = self.G0[p], self.b0[p].copy(); S = G + np.diag(1.0 / phi_rn[p])
            for e, cc in zip(self._ext_np, coeffs):      # data -> data - F_ext c_ext
                b = b - e["Bx"][p] @ cc[p]
                out += cc[p] @ e["gx_full"][p] - 0.5 * cc[p] @ e["Gx_full"][p] @ cc[p]
            e = np.zeros(K); e[:kg] = aE[p]; v = b - G @ e
            cf = np.linalg.cholesky(S); yv = np.linalg.solve(cf, v)
            out += e @ b - 0.5 * e @ G @ e + 0.5 * yv @ yv - np.sum(np.log(np.diag(cf))) - 0.5 * np.sum(np.log(phi_rn[p]))
        for f in range(kg):
            x = np.linalg.solve(self.Lgam, aE[:, f]); out -= 0.5 * (x @ x) / phi_gw[f] + 0.5 * n * np.log(phi_gw[f])
        return out + self.const

    def reference_jacobian(self, params):
        """``log|d a_E / d xi|`` in ns units for the float64 kernel at ``params`` (tests)."""
        c = self.constants(kh.jnp.float64)
        _, lp_gw = self._log10phi(params, c); phig = np.asarray(10.0 ** lp_gw)
        lp_rn, _ = self._log10phi(params, c); phi = np.asarray(10.0 ** lp_rn)
        jnp = kh.jnp; R, kg = self.rmax, self.kg
        X = np.sqrt(phi)[:, :, None] * self._np["V"]; Wp = X * self._np["sig"][:, None, :]
        ld = 0.0
        for p in range(self.npsr):
            L2 = np.linalg.cholesky(np.eye(R) + Wp[p].T @ Wp[p])
            Y = np.linalg.solve(L2, self._np["sig"][p][:, None] * self._np["E"][p]) * np.sqrt(phig)[None, :]
            L3 = np.linalg.cholesky(np.eye(kg) + Y.T @ Y)
            ld += -np.sum(np.log(np.diag(L3))) + 0.5 * np.sum(np.log(phig))
        return ld

    def reference_marginal(self, params):
        """Exact dense marginal over (a_RN, a_E) of the two-block HD model, numpy float64 (tests)."""
        c = self.constants(kh.jnp.float64)
        lp_rn, lp_gw = self._log10phi(params, c)
        phi_rn = np.asarray(10.0 ** lp_rn); phi_gw = np.asarray(10.0 ** lp_gw)
        n, K, kg = self.npsr, self.k, self.kg; nn = n * (K + kg)
        Lam = np.zeros((nn, nn)); bb = np.zeros(nn); ldP = 0.0
        ir = lambda p: slice(p * (K + kg), p * (K + kg) + K)
        ie = lambda p: slice(p * (K + kg) + K, (p + 1) * (K + kg))
        for p in range(n):
            G = self.G0[p]
            Lam[ir(p), ir(p)] += G + np.diag(1.0 / phi_rn[p]); Lam[ie(p), ie(p)] += G[:kg, :kg]
            Lam[ir(p), ie(p)] += G[:, :kg]; Lam[ie(p), ir(p)] += G[:kg, :]
            ldP += np.sum(np.log(phi_rn[p])); bb[ir(p)] = self.b0[p]; bb[ie(p)] = self.b0[p][:kg]
        Ginv = np.linalg.inv(self.Gam); ldG = np.linalg.slogdet(self.Gam)[1]
        for f in range(kg):
            ii = np.array([p * (K + kg) + K + f for p in range(n)])
            Lam[np.ix_(ii, ii)] += Ginv / phi_gw[f]; ldP += n * np.log(phi_gw[f]) + ldG
        cf = np.linalg.cholesky(Lam); x = np.linalg.solve(cf, bb)
        return (0.5 * x @ x - np.sum(np.log(np.diag(cf))) - 0.5 * ldP + self.const
                + 0.5 * nn * math.log(2 * math.pi) - 0.5 * n * kg * math.log(2 * math.pi) + 0.5 * kg * ldG)


def make_gwb_fp32(psrs, commongp, globalgp, noisedicts=None, extsignals=None, **kwargs):
    """Build :class:`GWBMarginalFp32` with white-noise kernels
    ``PulsarLikelihood([residuals, makenoise_measurement, makegp_ecorr]).N`` from each
    pulsar's noise dictionary (``psr.noisedict`` unless ``noisedicts`` is given)."""
    import discovery as ds          # current kernel binding; requires ds.config(kernels="metamath")
    kernels = []
    for i, psr in enumerate(psrs):
        nd = psr.noisedict if noisedicts is None else noisedicts[i]
        psl = ds.PulsarLikelihood([psr.residuals, ds.makenoise_measurement(psr, nd), ds.makegp_ecorr(psr, nd)])
        kernels.append(psl.N)
    return GWBMarginalFp32(psrs, kernels, commongp, globalgp, extsignals=extsignals, **kwargs)
