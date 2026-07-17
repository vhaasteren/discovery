"""Graph-consistent transport: a free-standing reparam object (D7-D13, D24).

`Transport` replaces the in-likelihood decentering closure. Like
`likelihood_metamath.py` it is a *boundary module* (D8): it may call
`metamatrix.func` at CONSTRUCTION time to bake constants, and the object it
returns participates in graphs only through the established FuncLeaf/reparam
contract. Kernel methods in `metamath.py` stay graph-pure.

Per pulsar, given blocks b = 1..B with bases W_b (n_toa, k_b) and live
conditioner precisions p_b(params) (k_b,), a frozen reference noise N0, and an
optional reference residual r0:

    W  = [W_1 | ... | W_B]                        (n_toa, k)
    G0 = W^T N0^-1 W                              baked once
    b0 = W^T N0^-1 r0                             baked once (centering)
    A(params)  = G0 + diag(p(params))            p = concat of p_b
    cf         = cho_factor(A, lower=True)        A = L L^T
    q          = mu + L^-T xi
    mu         = A^-1 b0                          0 if center=False
    ldJ        = -sum_i log L_ii                  log|dq/dxi|

mu is a translation (dmu/dxi = 0): it never enters ldJ. For any invertible
A(params) the map is a bijection with tracked Jacobian, so the transformed
density is exact regardless of how well A approximates the local posterior
precision (§5.2). Failure semantics: §5.7 -- construction raises, `validate`
raises, runtime `apply` is NaN-propagating JAX.
"""
import numpy as np
from dataclasses import dataclass

from . import metamatrix
from . import metamath
from . import utils as kh
from . import _kernels


@dataclass(frozen=True)
class TransportBlock:
    """One basis block of a transport. Construct via the adapters below."""
    name: str            # display name, e.g. 'rednoise'
    F: object            # (n_toa, k_b) ndarray -- constant basis at reference
    index: dict          # ONE coefficient-name -> slice(0, k_b) (localized)
    conditioner_precision: object   # callable params -> (k_b,); carries .params
    conditioner_kind: str = "exact_diagonal"  # or curn_inverse_marginal


def _validate_columns(name, F):
    F = np.asarray(F, dtype=np.float64)
    if F.ndim != 2 or F.shape[1] == 0:
        raise ValueError(f"transport block '{name}': basis must be a 2-D "
                         f"array with at least one column, got shape {F.shape}")
    norms = np.sqrt((F ** 2).sum(axis=0))
    bad = np.flatnonzero(~np.isfinite(norms) | (norms == 0.0))
    if bad.size:
        raise ValueError(
            f"transport block '{name}' has {bad.size} zero or non-finite "
            f"column(s) at indices {bad.tolist()}; every transport column "
            f"must be a finite, nonzero basis vector.")
    return F


def _eval_basis(F):
    """Materialize a constant basis. Arrays pass through; a graph-valued basis
    (dict) is materialized iff it is parameter-free -- mirroring the closure's
    `_eval_F` but refusing live bases."""
    if isinstance(F, dict):
        live = metamatrix.graph_params(F)
        if live:
            raise ValueError(
                f"transport bases must be constant; this basis graph depends "
                f"on parameters {live}. Materialize it at reference values "
                f"before building the transport.")
        return np.asarray(metamatrix.func(F)(params={}))
    return np.asarray(F)


def _single_key_index(index, psr_slot, width, blockname):
    """Select THE ONE coefficient key belonging to `psr_slot` from a factory
    index and localize it to slice(0, width).

    Factory index shapes (§2.2): per-pulsar GP -> one-entry dict (psr_slot must
    be None); commongp/globalgp -> flat dict with one entry per pulsar in
    pulsar order, slices k*i..k*(i+1); CompoundGP -> list of dicts.
    """
    if isinstance(index, list):                       # CompoundGP form
        if psr_slot is None or psr_slot < 0 or psr_slot >= len(index):
            raise ValueError(
                f"transport block '{blockname}': a list-form index requires "
                f"a valid psr_slot; got {psr_slot!r} for {len(index)} entries.")
        index = index[psr_slot]
        items = list(index.items())
        if len(items) != 1:
            raise ValueError(
                f"transport block '{blockname}': selected pulsar index has "
                f"{len(items)} keys; adapters accept one factory GP per block.")
        key, sli = items[0]
        if (sli.stop - sli.start) != width:
            raise ValueError(
                f"transport block '{blockname}': coefficient slice for "
                f"'{key}' has width {sli.stop - sli.start}, but the basis has "
                f"{width} columns.")
        return {key: slice(0, width)}

    items = list(index.items())
    if psr_slot is None:
        if len(items) != 1:
            raise ValueError(
                f"transport block '{blockname}': index has {len(items)} keys; "
                f"pass psr_slot to select one pulsar's coefficients.")
        key, sli = items[0]
    else:
        if psr_slot < 0 or psr_slot >= len(items):
            raise ValueError(
                f"transport block '{blockname}': psr_slot={psr_slot} out of "
                f"range for an index with {len(items)} pulsar entries.")
        key, sli = items[psr_slot]
    if (sli.stop - sli.start) != width:
        raise ValueError(
            f"transport block '{blockname}': coefficient slice for "
            f"'{key}' has width {sli.stop - sli.start}, but the basis has "
            f"{width} columns.")
    return {key: slice(0, width)}


def gp_block(gp, psr_slot=None):
    """Adapt a discovery GP with a DIAGONAL prior to a TransportBlock.

    - per-pulsar GP (makegp_fourier on one pulsar): psr_slot=None;
      Phi.getN(params) has shape (k,).
    - commongp (makecommongp_fourier): psr_slot=i; gp.F is a tuple of
      per-pulsar bases and Phi.getN(params) has shape (npsr, k) -- the pulsar's
      ROW must be selected (exactly how the decenter closure consumes it before
      its axis-1 concat).
    """
    F = _eval_basis(gp.F[psr_slot] if psr_slot is not None else gp.F)
    F = _validate_columns(getattr(gp, 'gpname', 'gp'), F)
    getN = gp.Phi.getN

    if psr_slot is None:
        def conditioner_precision(params):
            return kh.jnp.asarray(getN(params)) ** -1
    else:
        def conditioner_precision(params, _i=psr_slot):
            return kh.jnp.asarray(getN(params))[_i] ** -1
    conditioner_precision.params = list(getattr(getN, 'params', []))

    name = getattr(gp, 'gpname', 'gp')
    return TransportBlock(name, F,
                          _single_key_index(gp.index, psr_slot, F.shape[1], name),
                          conditioner_precision)


def globalgp_curn_block(globalgp, psr_slot, npsr):
    """Per-pulsar conditioner view of a DENSE global GP (D11): elementwise
    reciprocal of the dense Phi diagonal, reshaped per pulsar -- the existing
    decenter convention, kept for parity.

    INVERSE MARGINAL VARIANCE, NOT DENSE PRIOR PRECISION (§5.3 category 2).
    The exact dense prior stays in the likelihood; this view exists only inside
    the transport, and the function name says so at every call site.
    """
    if npsr != len(globalgp.Fs) or npsr != len(globalgp.index):
        raise ValueError(
            f"globalgp_curn_block: npsr={npsr}, but the global GP has "
            f"{len(globalgp.Fs)} bases and {len(globalgp.index)} index entries")
    if psr_slot < 0 or psr_slot >= npsr:
        raise ValueError(
            f"globalgp_curn_block: psr_slot={psr_slot} out of range "
            f"for npsr={npsr}")
    F = _validate_columns(getattr(globalgp, 'gpname', 'gw'),
                          _eval_basis(globalgp.Fs[psr_slot]))
    getN = globalgp.Phi.getN

    def conditioner_precision(params, _i=psr_slot, _n=npsr):
        return kh.jnp.diag(getN(params) ** -1).reshape((_n, -1))[_i]
    conditioner_precision.params = list(getattr(getN, 'params', []))

    name = getattr(globalgp, 'gpname', 'gw')
    return TransportBlock(name, F,
                          _single_key_index(globalgp.index, psr_slot,
                                            F.shape[1], name),
                          conditioner_precision,
                          conditioner_kind="curn_inverse_marginal")


# --------------------------------------------------------------------------
# reference noise (D10)
# --------------------------------------------------------------------------

class _FrozenSolve:
    """One-method reference-noise operator: solve(rhs) -> (N0^-1 rhs, logdet).
    Built at construction; contains no free parameters by construction."""
    def __init__(self, solve_fn, description):
        self._solve, self.description = solve_fn, description

    def solve(self, rhs):
        return self._solve(rhs)


def reference_noise(psr):
    """Diagonal TOA-error reference: N0 = diag(toaerrs**2). EFAC=1, no
    EQUAD/ECORR. The dependency-free default."""
    n0 = np.asarray(psr.toaerrs, dtype=np.float64) ** 2
    kernel = metamath.NoiseMatrix(kh.jnparray(n0))
    f = metamatrix.func(kernel.make_solve)
    return _FrozenSolve(lambda rhs: f(rhs, params={}),
                        f"toaerrs diagonal ({psr.name})")


def reference_noise_frozen(kernel, params0, description=None):
    """Freeze ANY metamath kernel's solve at explicit reference parameters
    params0 (e.g. a noisedict): works for measurement noise incl.
    Sherman-Morrison ECORR. This is the ONLY sanctioned way to use a
    parameterized kernel in a transport -- never call a live kernel with
    params={}."""
    params0 = dict(params0)  # freeze a snapshot; caller mutation cannot alter N0
    make_solve = getattr(kernel, 'make_solve', None)
    if make_solve is None or not isinstance(make_solve, dict):
        raise TypeError(
            f"reference_noise_frozen: expected a metamath kernel exposing "
            f"make_solve as a graph, got {type(kernel).__name__}. "
            f"(Legacy matrix.py kernels are not supported; build the model "
            f"under discovery.config(kernels='metamath').)")
    f = metamatrix.func(make_solve)
    missing = [p for p in f.params if p not in params0]
    if missing:
        raise ValueError(f"reference_noise_frozen: params0 is missing "
                         f"{missing}; a frozen reference must pin every "
                         f"parameter of the kernel it freezes.")
    return _FrozenSolve(lambda rhs: f(rhs, params=params0),
                        description or f"frozen kernel at {sorted(params0)}")


# --------------------------------------------------------------------------
# the Transport object (D1, D13)
# --------------------------------------------------------------------------

_KINDS = ("exact_diagonal", "curn_inverse_marginal")


class Transport:
    """Per-pulsar frozen-reference transport  xi -> (q, ldJ).

    q = mu(params) + L(params)^-T xi,
    A(params) = G0 + diag(conditioner_precision(params)),  A = L L^T,
    ldJ = -sum(log diag L).  Math and failure semantics: §5.2, §5.7.
    """

    def __init__(self, blocks, *, reference_noise, reference_residual=None,
                 center=True):
        _kernels.require_metamath("Transport")                       # D1

        blocks = list(blocks)
        if not blocks:
            raise ValueError("Transport requires at least one block")

        ntoas = {np.asarray(b.F).shape[0] for b in blocks}
        if len(ntoas) != 1:
            raise ValueError(f"transport blocks disagree on n_toa: "
                             f"{sorted(ntoas)}")
        self._ntoa = ntoas.pop()

        # -- index assembly: contiguous, non-overlapping, collision-checked --
        self.blocks = blocks
        offset, self.index = 0, {}
        for b in self.blocks:
            if b.conditioner_kind not in _KINDS:
                raise ValueError(
                    f"block '{b.name}': unknown conditioner_kind "
                    f"{b.conditioner_kind!r}")
            if len(b.index) != 1:
                raise ValueError(
                    f"block '{b.name}': index must contain exactly one "
                    f"coefficient key; got {len(b.index)}")
            for key, sli in b.index.items():
                if key in self.index:
                    raise ValueError(f"duplicate coefficient key '{key}' "
                                     f"across transport blocks")
                if (sli.start, sli.stop) != (0, np.asarray(b.F).shape[1]):
                    raise ValueError(f"block '{b.name}': index slice must be "
                                     f"localized to slice(0, k_b); got {sli}")
                self.index[key] = slice(offset + sli.start, offset + sli.stop)
            offset += np.asarray(b.F).shape[1]
        self.dimension = offset

        # -- bake (construction-time boundary work, D8) ----------------------
        W = np.concatenate([np.asarray(b.F) for b in self.blocks], axis=1)
        self._W = kh.jnparray(W)             # retained for diagnostics/inverse
        N0mW, _ = reference_noise.solve(self._W)
        if not bool(np.all(np.isfinite(np.asarray(N0mW)))):
            raise ValueError(f"reference-noise solve produced non-finite "
                             f"values ({reference_noise.description})")
        self._G0 = kh.jnparray(W.T @ np.asarray(N0mW))
        self.reference_description = reference_noise.description

        self._b0 = None
        if center:
            if reference_residual is None:
                raise ValueError("center=True requires reference_residual")
            r0 = np.asarray(reference_residual, dtype=np.float64)
            if r0.shape != (self._ntoa,):
                raise ValueError(f"reference_residual has shape {r0.shape}; "
                                 f"expected ({self._ntoa},)")
            self._b0 = kh.jnparray(np.asarray(N0mW).T @ r0)

        self.center = center
        self.params = sorted(set(sum(
            [list(b.conditioner_precision.params) for b in self.blocks], [])))

    # -- per-evaluation map (plain JAX; composes as a reparam FuncLeaf) ------
    def _factor(self, params):
        pinv = kh.jnp.concatenate(
            [b.conditioner_precision(params) for b in self.blocks])
        i1, i2 = kh.jnp.diag_indices(self.dimension)
        return (kh.jsp.linalg.cho_factor(
                    self._G0.at[i1, i2].add(pinv), lower=True),
                pinv)

    def apply(self, params, xi):
        cf, _ = self._factor(params)
        q = kh.jsp.linalg.solve_triangular(cf[0], xi, trans=1, lower=cf[1])
        ldJ = -kh.jnp.sum(kh.jnp.log(kh.jnp.diag(cf[0])))
        if self._b0 is not None:
            q = q + kh.jsp.linalg.cho_solve(cf, self._b0)
        return q, ldJ

    def split(self, q):
        """{coefficient-key: q[slice]} view, in self.index order."""
        return {key: q[sli] for key, sli in self.index.items()}

    def as_reparam(self):
        """The metamath reparam contract: rp(params, c) -> (c_out, ldL), with
        TRUE .params (D13)."""
        def rp(params, c):
            return self.apply(params, c)
        rp.params = list(self.params)
        return rp

    def validate(self, params):
        """Eager, non-JIT positivity/PD check (§5.7). Raises ValueError with
        per-block diagnostics; returns the §5.8 diagnostics dict on success."""
        offset = 0
        for b in self.blocks:
            p = np.asarray(b.conditioner_precision(params))
            kb = np.asarray(b.F).shape[1]
            if p.shape != (kb,):
                raise ValueError(f"block '{b.name}': conditioner_precision "
                                 f"returned shape {p.shape}; expected ({kb},)")
            bad = np.flatnonzero(~np.isfinite(p) | (p < 0.0))
            if bad.size:
                raise ValueError(
                    f"block '{b.name}': conditioner_precision has "
                    f"non-finite or negative entries at indices "
                    f"{bad.tolist()} (block-local). No floor is applied; "
                    f"fix the prior or remove the block (D9).")
            offset += kb
        cf, _ = self._factor(params)
        diag = np.asarray(kh.jnp.diag(cf[0]))
        if not np.all(np.isfinite(diag)) or np.any(diag <= 0.0):
            raise ValueError(
                f"transport factorization is not positive definite at the "
                f"given params (min Cholesky diagonal "
                f"{np.nanmin(diag):.3e}). A conditioned direction has no "
                f"prior support or the basis is degenerate; no floor is "
                f"applied (D9). Blocks: "
                f"{[(b.name, np.asarray(b.F).shape[1]) for b in self.blocks]}")
        return self.diagnostics(params)

    def diagnostics(self, params=None, noise_solve=None):
        """Structural report plus optional eager numerical diagnostics.

        ``noise_solve`` is test-scale only. For diagonal-prior blocks its
        eigenvalues are the transformed local target metric. If this Transport
        contains a CURN block they are explicitly a *conditioner metric*: the
        dense cross-pulsar target precision cannot be represented per pulsar.
        """
        out = {
            "blocks": [
                {"name": b.name, "k": int(np.asarray(b.F).shape[1]),
                 "params": list(b.conditioner_precision.params),
                 "keys": list(b.index), "conditioner_kind": b.conditioner_kind}
                for b in self.blocks
            ],
            "dimension": self.dimension,
            "center": self.center,
            "reference_noise": self.reference_description,
        }
        if params is None:
            if noise_solve is not None:
                raise ValueError("diagnostics(noise_solve=...) requires params")
            return out

        cf, pinv = self._factor(params)
        p = np.asarray(pinv)
        d = np.asarray(kh.jnp.diag(cf[0]))
        out.update(
            precision_min=float(np.min(p)),
            precision_max=float(np.max(p)),
            chol_diag_min=float(np.min(d)),
            chol_diag_max=float(np.max(d)),
        )

        if noise_solve is not None:
            NmW, _ = noise_solve(self._W)
            H = np.asarray(self._W).T @ np.asarray(NmW) + np.diag(p)
            L = np.tril(np.asarray(cf[0]))
            Linv = np.linalg.solve(L, np.eye(self.dimension))
            eig = np.linalg.eigvalsh(Linv @ H @ Linv.T)
            out.update(
                metric_kind=("conditioner" if any(
                    b.conditioner_kind == "curn_inverse_marginal"
                    for b in self.blocks)
                    else "local_target"),
                metric_eig_min=float(eig[0]),
                metric_eig_max=float(eig[-1]),
            )
        return out


# --------------------------------------------------------------------------
# ArrayTransport (D24)
# --------------------------------------------------------------------------

class ArrayTransport:
    """Per-pulsar transports behind one array reparam.

    Requirements (validated, D24):
      - at least one transport;
      - EQUAL dimension k across pulsars (the array coefficient contract is a
        rectangular (npsr, k) array, so ragged transports cannot pass through
        it; ragged support is a non-goal, D23);
      - all-or-none centering (mixed centering would silently shift some
        pulsars' coordinates and not others').
    """

    def __init__(self, transports):
        transports = list(transports)
        if not transports:
            raise ValueError("ArrayTransport requires at least one Transport")
        dims = sorted({t.dimension for t in transports})
        if len(dims) != 1:
            raise ValueError(
                f"ArrayTransport requires equal per-pulsar dimension; got "
                f"{dims}. Ragged transports are not supported (D23/D24); "
                f"use equal GP component counts across pulsars.")
        centers = {t.center for t in transports}
        if len(centers) != 1:
            raise ValueError("ArrayTransport requires all-or-none centering")

        self.transports = transports
        self.dimension = dims[0]
        self.npsr = len(transports)
        self.center = centers.pop()
        self.params = sorted(set().union(*[set(t.params) for t in transports]))

        # batched bake: (npsr, k, k) and (npsr, k). This is the SAME batched
        # arithmetic as the decenter closure: preserve its call conventions.
        self._G0 = kh.jnparray([t._G0 for t in self.transports])
        self._b0 = (kh.jnparray([t._b0 for t in self.transports])
                    if self.center else None)

    def _pinv(self, params):
        return kh.jnp.stack([
            kh.jnp.concatenate([b.conditioner_precision(params)
                                for b in t.blocks])
            for t in self.transports])                    # (npsr, k)

    def apply(self, params, c):
        # c: (npsr, k) -- the array coefficient contract.
        i1, i2 = kh.jnp.diag_indices(self.dimension, ndim=2)
        cf = kh.jsp.linalg.cho_factor(
            self._G0.at[:, i1, i2].add(self._pinv(params)), lower=True)
        am = kh.jsp.linalg.solve_triangular(cf[0], c, trans=1, lower=cf[1])
        ldJ = -kh.jnp.logdet(cf[0][:, i1, i2])            # summed, as today
        if self._b0 is not None:
            am = am + kh.jsp.linalg.cho_solve(cf, self._b0)
        return am, ldJ

    def as_reparam(self):
        dim, npsr = self.dimension, self.npsr

        def rp(params, c):
            if c.shape != (npsr, dim):                    # trace-time check
                raise ValueError(
                    f"ArrayTransport: coefficient array has shape {c.shape}; "
                    f"the transport was built for ({npsr}, {dim}). The "
                    f"likelihood's coefficient width and the transport's "
                    f"blocks disagree.")
            return self.apply(params, c)
        rp.params = list(self.params)
        return rp

    def validate(self, params):
        return [t.validate(params) for t in self.transports]

    def diagnostics(self, params=None, noise_solve=None):
        """Aggregate per-pulsar diagnostics.

        noise_solve is None or a sequence of exactly npsr live solve callables.
        """
        if noise_solve is None:
            solves = [None] * self.npsr
        else:
            solves = list(noise_solve)
            if len(solves) != self.npsr:
                raise ValueError(
                    f"ArrayTransport diagnostics expected {self.npsr} "
                    f"noise solves; got {len(solves)}")
        return {
            "per_pulsar": [
                t.diagnostics(params=params, noise_solve=s)
                for t, s in zip(self.transports, solves)
            ],
            "dimension": self.dimension,
            "npsr": self.npsr,
            "params": list(self.params),
        }
