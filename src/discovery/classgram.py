"""Class-quantized white-noise Gram for coefficient transports.

For a basis W (n_toa, k), a residual r0 (n_toa,), and a measurement kernel

    N(params) = diag(d(params)) + U diag(lam(params)) U^T        (U column-disjoint)

bake, at reference parameters params0, the exact Gram of the class-quantized
model N^(params) = diag(d^) + U diag(lam(params)) U^T with

    d^_i = d_i^0 / omega_{m(i)}(params),
    omega_m = sum_{i in m} 1/d_i(params) / sum_{i in m} 1/d_i^0,

for TOAs in a baked class m (per backend, fixed-width bins of log10 toaerr^2
holding >= dense_threshold TOAs) and d^_i = d_i for every other TOA (a "dense
row", kept exactly). G^ == W^T N^-1 W at params0, G^ is a real Gram (PSD) for
every params, EFAC moves are exact, dense rows and ECORR enter exactly
(per-epoch Sherman-Morrison on the quantized diagonal)
the only approximation
is that TOAs in one baked class share a precision RATIO.
Cost per step: O(n) scalars + (M + n_dense) k^2 + E k^2.

Pure array code: numpy at bake, numpy or jax.numpy at evaluation (`xp`).
"""
from __future__ import annotations

import collections
import dataclasses
from typing import Callable, Optional

import numpy as np

from . import metamath

__all__ = [
    "MeasurementStructure", "measurement_structure", "white_noise_kernel",
    "response_partition", "ClassLayout", "build_layout",
    "ClassWeights", "ClassGram", "validate_class_gram",
]


# ---------------------------------------------------------------------------
# kernel structure
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MeasurementStructure:
    """(d, U, lam) view of a measurement kernel.

    diag    : callable params -> (n_toa,) variance, with .params (may be empty)
    epoch   : (n_toa,) int, ECORR epoch index or -1
    weight  : (n_toa,) float, exposure weight (0 outside any epoch)
    n_epoch : int
    ecorr   : callable params -> (n_epoch,) variance, with .params, or None
    """
    diag: Callable
    epoch: np.ndarray
    weight: np.ndarray
    n_epoch: int
    ecorr: Optional[Callable]

    @property
    def params(self) -> list[str]:
        out = list(getattr(self.diag, "params", []))
        if self.ecorr is not None:
            out += [p for p in getattr(self.ecorr, "params", []) if p not in out]
        return out


def _as_callable(x):
    """metamath array / callable / graph -> callable(params) -> array, with .params."""
    const, fn = metamath._materialize(x)
    if fn is None:
        arr = np.asarray(const, dtype=np.float64)

        def f(params, _a=arr):
            return _a
        f.params = []
        return f

    def g(params, _fn=fn):
        return _fn(params=params)
    g.params = list(getattr(fn, "params", []))
    return g


def _epoch_structure(F):
    F = np.asarray(F, dtype=np.float64)
    n, n_epoch = F.shape
    if n_epoch == 0:
        return np.full(n, -1, dtype=np.int64), np.zeros(n), 0
    nnz = np.count_nonzero(F, axis=1)
    if nnz.max() > 1:
        raise ValueError(
            f"classgram: ECORR exposure is not column-disjoint (max {nnz.max()} "
            "nonzero columns per TOA); the class Gram requires disjoint epochs")
    epoch = np.where(nnz == 1, np.argmax(F != 0.0, axis=1), -1).astype(np.int64)
    weight = np.zeros(n)
    rows = np.flatnonzero(epoch >= 0)
    weight[rows] = F[rows, epoch[rows]]
    return epoch, weight, int(n_epoch)


def measurement_structure(kernel, params0) -> MeasurementStructure:
    """Structure of a canonical WHITE-noise kernel, or TypeError.

      * metamath.NoiseMatrixSM(N, F, P)          -- diagonal + ECORR (the canonical form)
      * metamath.NoiseMatrix / NoiseMatrix1D(N)   -- diagonal only
    Anything else (a Woodbury stack, a dense NoiseMatrix2D) is refused: use
    `white_noise_kernel(likelihood)` to canonicalize first. `params0` sizes the
    diagonal-only case.
    """
    if isinstance(kernel, metamath.NoiseMatrixSM):
        epoch, weight, n_epoch = _epoch_structure(kernel.F)
        return MeasurementStructure(_as_callable(kernel.N), epoch, weight, n_epoch,
                                    _as_callable(kernel.P))
    if isinstance(kernel, metamath.NoiseMatrix):
        diag = _as_callable(kernel.N)
        d0 = np.asarray(diag(dict(params0)), dtype=np.float64)
        if d0.ndim != 1:
            raise TypeError("classgram: only a DIAGONAL NoiseMatrix is white noise")
        n = int(d0.shape[0])
        return MeasurementStructure(diag, np.full(n, -1, dtype=np.int64), np.zeros(n), 0, None)
    raise TypeError(
        f"classgram: unsupported kernel {type(kernel).__name__}; pass a NoiseMatrix1D or "
        "NoiseMatrixSM (see white_noise_kernel), not a Woodbury stack")


def _is_ecorr_basis(F):
    F = np.asarray(F)
    return (F.ndim == 2 and F.shape[1] > 0 and np.all((F == 0.0) | (F == 1.0))
            and np.all(np.count_nonzero(F, axis=1) <= 1))


def white_noise_kernel(likelihood):
    """Canonical white-noise kernel of a PulsarLikelihood: NoiseMatrix1D or
    NoiseMatrixSM, with a fixed ECORR GP component folded into the SM form.

    Reads `likelihood.signals`: exactly one `Kernel` (the measurement noise),
    plus at most one ECORR `ConstantGP` -- a 0/1 column-disjoint basis with a
    diagonal NoiseMatrix-family prior (`makegp_ecorr(psr, noisedict)`). The
    improper timing GP (dense basis) and Fourier GPs are not white noise and
    are ignored.

    A `VariableGP` with an ECORR basis (`makegp_ecorr` without a noisedict) is
    NOT folded in: in the coefficient likelihood its amplitudes are sampled
    coefficients, so it belongs to the transport as a `gp_block` (nltiming's
    `build_joint_transport` adds every `sampled_gps` entry) and the white noise
    the chart must track is the diagonal alone. Free ECORR *hyperparameters*
    with marginalized amplitudes are `makenoise_measurement(..., ecorr=True)`.
    An SM kernel together with a fixed ECORR GP is a double ECORR and raises.
    """
    from . import _kernels
    from . import utils as kh
    signals = list(getattr(likelihood, "signals", []))
    kernels = [x for x in signals if isinstance(x, kh.Kernel)]
    if len(kernels) != 1:
        raise ValueError(f"white_noise_kernel: expected one measurement kernel; found {len(kernels)}")
    kern = kernels[0]
    if not isinstance(kern, (metamath.NoiseMatrix, metamath.NoiseMatrixSM)):
        raise TypeError(f"white_noise_kernel: measurement kernel {type(kern).__name__} is not white noise")
    ecorr_gps = [g for g in signals if isinstance(g, kh.ConstantGP)
                 and _is_ecorr_basis(getattr(g, "F", None))]
    if not ecorr_gps:
        return kern
    if len(ecorr_gps) > 1:
        raise ValueError("white_noise_kernel: more than one fixed ECORR GP component")
    egp = ecorr_gps[0]
    if isinstance(kern, metamath.NoiseMatrixSM):
        raise ValueError("white_noise_kernel: ECORR appears both in the SM kernel and as a GP")
    P = getattr(egp.Phi, "N", None)
    if P is None:
        raise TypeError("white_noise_kernel: ECORR GP prior must be a NoiseMatrix-family Phi")
    # the SAME factory makenoise_measurement(ecorr=True) uses: no second SM path
    sm = _kernels.NoiseMatrixSM(kern.N, np.asarray(egp.F, dtype=np.float64), P)
    sm.measurement = dict(getattr(kern, "measurement", {}) or {})
    sm.measurement["ecorr"] = True
    return sm


def response_partition(fn, params0, n_out, *, rel_step=1e-3, max_groups=None):
    names = list(getattr(fn, "params", []))
    base = np.asarray(fn(params0), dtype=np.float64)
    if base.shape != (n_out,):
        raise ValueError(f"classgram: expected fn(params0) of shape ({n_out},); got {base.shape}")
    sig = np.zeros((n_out, max(1, len(names))), dtype=bool)
    for j, name in enumerate(names):
        p = dict(params0)
        v = float(p[name])
        p[name] = v + rel_step * (abs(v) if v != 0.0 else 1.0)
        sig[:, j] = np.asarray(fn(p), dtype=np.float64) != base
    _, group = np.unique(sig, axis=0, return_inverse=True)
    group = group.reshape(-1).astype(np.int64)
    n_groups = int(group.max()) + 1
    if max_groups is not None and n_groups > max_groups:
        raise ValueError(
            f"classgram: {n_groups} response groups exceed max_groups={max_groups}; "
            "per-TOA noise parameters (outlier scalings) are not supported")
    return group, n_groups


@dataclasses.dataclass(frozen=True)
class ClassLayout:
    """toa_class : (n_toa,) baked class id in [0, n_class), or -1 for a dense (exact) TOA
    dense     : (n_dense,) indices of the dense TOAs
    n_class   : number of baked classes
    """
    toa_class: np.ndarray
    dense: np.ndarray
    n_class: int
    group: np.ndarray

    @property
    def n_dense(self):
        return int(self.dense.size)


def build_layout(struct: MeasurementStructure, params0, toaerrs, *,
                 sigma_bin_dex=0.2, dense_threshold=16, max_groups=512) -> ClassLayout:
    """Per diagonal response group (backend): fixed-width bins of width
    `sigma_bin_dex` in log10 toaerr^2. A bin with >= dense_threshold TOAs
    becomes a baked k x k class; every other TOA is kept exact (a dense row)."""
    n = int(struct.epoch.shape[0])
    toaerrs = np.asarray(toaerrs, dtype=np.float64)
    if toaerrs.shape != (n,):
        raise ValueError(f"classgram: toaerrs has shape {toaerrs.shape}; expected ({n},)")
    if not np.all(np.isfinite(toaerrs)) or np.any(toaerrs <= 0.0):
        raise ValueError("classgram: toaerrs must be finite and strictly positive")
    if float(sigma_bin_dex) <= 0 or int(dense_threshold) < 2:
        raise ValueError("classgram: sigma_bin_dex must be > 0 and dense_threshold >= 2")
    group, n_groups = response_partition(struct.diag, params0, n, max_groups=max_groups)
    toa_class = np.full(n, -1, dtype=np.int64)
    cid = 0
    l10 = np.log10(toaerrs ** 2)
    for g in range(n_groups):
        rows = np.flatnonzero(group == g)
        b = np.floor((l10[rows] - l10[rows].min()) / float(sigma_bin_dex)).astype(np.int64)
        for bid in np.unique(b):
            sel = rows[b == bid]
            if sel.size >= int(dense_threshold):
                toa_class[sel] = cid
                cid += 1
    dense = np.flatnonzero(toa_class < 0)
    return ClassLayout(toa_class, dense, cid, group)


def _segment_sum(values, seg, n_seg, xp):
    if xp is np:
        out = np.zeros((n_seg,) + np.shape(values)[1:], dtype=np.float64)
        np.add.at(out, seg, np.asarray(values))
        return out
    import jax
    return jax.ops.segment_sum(values, seg, num_segments=n_seg)


ClassWeights = collections.namedtuple(
    "ClassWeights", "omega p_dense omega_e Y_add v_add S")


class ClassGram:
    """Per step:
        p = 1/d(params)                                              O(n)
        omega_m = sum_{i in m} p_i / sum_{i in m} p_i^0              O(n)
        G_diag = sum_m omega_m A_m + F_d^T diag(p_d) F_d             O(M k^2 + n_dense k^2)
        y_e = sum_j omega[cls(e,j)] Y0[e,j] + sum_{i in e, dense} w p F_i     O(E n_sub k + n_dense k)
        t_e, v_e likewise
        S_e = 1/(1/lam_e + t_e)
        G = G_diag - Y^T diag(S) Y
        b analogous                    O(E k^2)
    """

    def __init__(self, W, r0, struct, params0, layout):
        W = np.asarray(W, dtype=np.float64)
        r0 = np.asarray(r0, dtype=np.float64)
        n, k = W.shape
        self.struct, self.layout, self.k, self.n = struct, layout, k, n
        self.params = list(struct.params)
        p0 = 1.0 / np.asarray(struct.diag(params0), dtype=np.float64)
        if not np.all(np.isfinite(p0)) or np.any(p0 <= 0):
            raise ValueError("classgram: reference diagonal must be finite and positive")
        M, tc = layout.n_class, layout.toa_class
        baked = tc >= 0
        tcb = np.where(baked, tc, 0)
        Wp = W * p0[:, None]
        self.A = np.stack([W[tc == m].T @ Wp[tc == m] for m in range(M)]) if M else np.zeros((0, k, k))
        self.a = _segment_sum(Wp * r0[:, None] * baked[:, None], tcb, max(M, 1), np)[:M]
        self.psum0 = _segment_sum(p0 * baked, tcb, max(M, 1), np)[:M]
        self._tcb, self._baked = tcb, baked.astype(np.float64)
        self.dense = layout.dense
        self.Fd = W[self.dense]                       # (n_dense, k)
        self.rd = r0[self.dense]
        E = struct.n_epoch
        self.has_ecorr = E > 0
        if not self.has_ecorr:
            return
        valid = struct.epoch >= 0
        segc = np.where(valid, struct.epoch, 0)
        w = struct.weight * valid
        # baked TOAs: per-epoch tables over the epoch's baked classes; to keep them
        # compact, index by (epoch, local slot) where slots enumerate the distinct
        # baked classes present in the epoch (n_sub = max over epochs).
        rows = np.flatnonzero(valid & baked)
        pairs = np.unique(np.stack([segc[rows], tc[rows]], axis=1), axis=0)  # (P, 2) epoch, class
        slot = np.zeros(pairs.shape[0], dtype=np.int64)
        for e in np.unique(pairs[:, 0]):
            idx = np.flatnonzero(pairs[:, 0] == e)
            slot[idx] = np.arange(idx.size)
        ns = int(slot.max()) + 1 if slot.size else 1
        key = {(int(e), int(c)): int(s) for (e, c), s in zip(pairs, slot)}
        slot_of_row = np.array([key[(int(segc[i]), int(tc[i]))] for i in rows], dtype=np.int64)
        T0 = np.zeros((E, ns))
        Y0 = np.zeros((E, ns, k))
        V0 = np.zeros((E, ns))
        cls = np.zeros((E, ns), dtype=np.int64)
        mask = np.zeros((E, ns))
        np.add.at(T0, (segc[rows], slot_of_row), (w * w * p0)[rows])
        np.add.at(Y0, (segc[rows], slot_of_row), ((w * p0)[rows])[:, None] * W[rows])
        np.add.at(V0, (segc[rows], slot_of_row), (w * p0 * r0)[rows])
        cls[pairs[:, 0], slot] = pairs[:, 1]
        mask[pairs[:, 0], slot] = 1.0
        self.T0, self.Y0, self.V0, self._cls, self._mask = T0, Y0, V0, cls, mask
        # dense TOAs inside epochs
        drows = np.flatnonzero(valid & ~baked)
        self._dseg = segc[drows]
        self._dw = w[drows]
        self._drows = drows
        self._d_in_dense = np.searchsorted(self.dense, drows)   # positions within self.dense
        self.E = E

    @property
    def n_matrices(self):
        return int(self.A.shape[0])

    @property
    def n_dense(self):
        return int(self.dense.size)

    @property
    def n_epoch(self):
        return int(self.struct.n_epoch)

    def arrays(self):
        """The baked arrays the per-step math reads: (A, a, Fd, rd, Y0, V0). The
        transport converts these once to the evaluation dtype/device and passes
        them back through `gram(arrays=...)`."""
        if self.has_ecorr:
            return (self.A, self.a, self.Fd, self.rd, self.Y0, self.V0)
        return (self.A, self.a, self.Fd, self.rd, None, None)

    def batched_weights(self, params, xp=np, *, arrays=None):
        """All per-step scalars, in stackable form: a `ClassWeights` with
        omega (M,), p_dense (n_dense,), omega_e (E, n_sub), Y_add (E, k),
        v_add (E,), S (E,). Contains no k x k product, so ArrayTransport can
        stack these across pulsars and run the contractions once, batched.
        For a pulsar without ECORR the epoch fields are None."""
        A, a, Fd_, rd_, Y0_, V0_ = arrays if arrays is not None else self.arrays()
        p = 1.0 / xp.asarray(self.struct.diag(params))
        M = self.layout.n_class
        omega = (_segment_sum(p * xp.asarray(self._baked), self._tcb, M, xp) / xp.asarray(self.psum0)
                 if M else xp.zeros(0))
        pd = p[self.dense]
        if not self.has_ecorr:
            return ClassWeights(omega, pd, None, None, None, None)
        om_e = (omega[self._cls] if M else xp.zeros(self._cls.shape)) * xp.asarray(self._mask)
        t = xp.sum(om_e * xp.asarray(self.T0), axis=1)
        Y_add = xp.zeros((self.E, self.k))
        v_add = xp.zeros(self.E)
        if self._drows.size:
            Fd = xp.asarray(Fd_)
            rd = xp.asarray(rd_)
            wp = xp.asarray(self._dw) * pd[self._d_in_dense]
            t = t + _segment_sum(xp.asarray(self._dw) * wp, self._dseg, self.E, xp)
            Y_add = _segment_sum(wp[:, None] * Fd[self._d_in_dense], self._dseg, self.E, xp)
            v_add = _segment_sum(wp * rd[self._d_in_dense], self._dseg, self.E, xp)
        lam = xp.asarray(self.struct.ecorr(params))
        S = 1.0 / (1.0 / lam + t)
        return ClassWeights(omega, pd, om_e, Y_add, v_add, S)

    def gram(self, params, xp=np, *, arrays=None):
        """(G^ (k, k), b^ (k,)) -- the three contractions on `batched_weights`."""
        A, a, Fd_, rd_, Y0_, V0_ = arrays if arrays is not None else self.arrays()
        w = self.batched_weights(params, xp, arrays=arrays)
        Fd = xp.asarray(Fd_)
        rd = xp.asarray(rd_)
        if self.layout.n_class:
            G = xp.einsum("m,mij->ij", w.omega, xp.asarray(A))
            b = xp.einsum("m,mi->i", w.omega, xp.asarray(a))
        else:
            G = xp.zeros((self.k, self.k))
            b = xp.zeros(self.k)
        G = G + Fd.T @ (w.p_dense[:, None] * Fd)
        b = b + Fd.T @ (w.p_dense * rd)
        if w.S is not None:
            Y = xp.einsum("ej,ejk->ek", w.omega_e, xp.asarray(Y0_)) + w.Y_add
            v = xp.sum(w.omega_e * xp.asarray(V0_), axis=1) + w.v_add
            G = G - Y.T @ (w.S[:, None] * Y)
            b = b - Y.T @ (w.S * v)
        return 0.5 * (G + G.T), b       # exact symmetry for cho_factor, as the frozen bake does


def validate_class_gram(cg: ClassGram, W, r0, solve, params0, *, rtol=1e-9):
    """Build-time exactness: G^(params0) == W^T N(params0)^-1 W to rtol, using the
    kernel's own solve (rhs, params) -> (N^-1 rhs, logdet). A mismatch means the
    structure extraction is wrong; it is never a tolerance question."""
    NmW, _ = solve(np.asarray(W), params0)
    NmW = np.asarray(NmW, dtype=np.float64)
    G_ref = np.asarray(W).T @ NmW
    b_ref = NmW.T @ np.asarray(r0, dtype=np.float64)
    G, b = cg.gram(params0, np)
    scale = max(float(np.abs(G_ref).max()), np.finfo(np.float64).tiny)
    err = float(np.abs(G - G_ref).max() / scale)
    if err > rtol:
        raise ValueError(f"classgram: G^(params0) differs from the exact Gram (rel {err:.2e} > {rtol:.0e})")
    bscale = max(float(np.abs(b_ref).max()), np.finfo(np.float64).tiny)
    berr = float(np.abs(b - b_ref).max() / bscale)
    if berr > rtol:
        raise ValueError(f"classgram: b^(params0) differs from the exact projection (rel {berr:.2e})")
    return {"gram_rel_err": err, "proj_rel_err": berr,
            "n_classes": cg.n_matrices, "n_dense": cg.n_dense, "n_epoch": cg.n_epoch}
