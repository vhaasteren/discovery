"""Graph-consistent transport: a free-standing reparam object.

`Transport` replaces the in-likelihood decentering closure. Like
`likelihood_metamath.py` it is a *boundary module*: it may call
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
    mu         = A^-1 b0                          0 if origin="zero"
    ldJ        = -sum_i log L_ii                  log|dq/dxi|

`origin` names where mu sits; it is NOT a switch for the reparameterization.
The map is the non-centered ("decentered") one either way -- `L^-T` whitens
the scale, and `origin` only chooses the point it whitens about:
`"conditional_mode"` (A^-1 b0) or `"zero"` (the prior mean). The word
"centered" is avoided deliberately: in the decentering literature it names
the *prior* parameterization, which `q = L^-T xi` already is not, so a
`center=` flag read exactly backwards.

mu is a translation (dmu/dxi = 0): it never enters ldJ. For any invertible
A(params) the map is a bijection with tracked Jacobian, so the transformed
density is exact regardless of how well A approximates the local posterior
precision. Failure semantics: construction raises, `validate`
raises, runtime `apply` is NaN-propagating JAX.
"""
import numpy as np
from dataclasses import dataclass

from . import metamatrix
from . import metamath
from . import utils as kh
from . import _kernels


bake_dtype = kh.bake_dtype


def _as_bake(a):
    return kh.jnp.asarray(a, dtype=bake_dtype())


#: Where the affine map's origin sits. See the module docstring.
ORIGINS = ("conditional_mode", "zero")


def _resolve_origin(origin, *, what):
    """Validate `origin=` at the boundary, so nothing downstream can see junk."""
    if origin not in ORIGINS:
        raise ValueError(f"{what}: origin must be one of {list(ORIGINS)}; "
                         f"got {origin!r}")
    return origin


# Relative tolerance on lambda_min(G0) / lambda_max(G0). A float64 bake of a
# rank-deficient Gram sits at ~1e-14; a float32 bake at ~1e-5.
_G0_PSD_RTOL = 1e-9


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

    Factory index shapes: per-pulsar GP -> one-entry dict (psr_slot must
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

    # Reciprocals are taken in the bake dtype: the GP spectrum is emitted in
    # the working dtype, and a float32 reciprocal of a (1 ns)^2-scale variance
    # is quantized and has an overflowing second derivative (phi**-3).
    if psr_slot is None:
        def conditioner_precision(params):
            return _as_bake(getN(params)) ** -1
    else:
        def conditioner_precision(params, _i=psr_slot):
            return _as_bake(getN(params))[_i] ** -1
    conditioner_precision.params = list(getattr(getN, 'params', []))

    name = getattr(gp, 'gpname', 'gp')
    return TransportBlock(name, F,
                          _single_key_index(gp.index, psr_slot, F.shape[1], name),
                          conditioner_precision)


def _conditioner_precision_from_spec(spec, k, name):
    """Normalize an `array_block` precision spec into a callable
    `params -> (k,)` carrying `.params`.

    Accepts a scalar (broadcast), a `(k,)` vector, or a callable with a
    `.params` attribute. Constant specs are positivity-checked eagerly (no
    floors); a live callable is trusted and validated at `validate()` time.
    """
    if callable(spec):
        params = list(getattr(spec, 'params', []))

        def cp(params_in, _f=spec):
            return _as_bake(_f(params_in))
        cp.params = params
        return cp

    arr = np.asarray(spec, dtype=np.float64)
    if arr.ndim == 0:
        vec = np.full((k,), float(arr))
    elif arr.shape == (k,):
        vec = arr
    else:
        raise ValueError(
            f"array_block '{name}': conditioner_precision must be a scalar, a "
            f"({k},) vector, or a callable with .params; got shape {arr.shape}")
    bad = np.flatnonzero(~np.isfinite(vec) | (vec < 0.0))
    if bad.size:
        raise ValueError(
            f"array_block '{name}': constant conditioner_precision has "
            f"non-finite or negative entries at indices {bad.tolist()}; supply "
            f"a proper prior precision (no floors).")
    jvec = kh.jnparray(vec)

    def cp_const(params_in, _v=jvec):
        return _v
    cp_const.params = []
    return cp_const


def array_block(F, index, conditioner_precision, name="external"):
    """Caller-declared external transport block.

    `F` is a plain constant basis discovery does not interpret; `index` is a
    one-key `{name: slice(0, k)}` map naming the caller's coordinate; and
    `conditioner_precision` is MANDATORY -- the exact prior precision in the
    caller's sampling coordinate (a scalar, a `(k,)` vector, or a callable with
    `.params`). Discovery provides no default and no floor. Column
    validation is identical to the GP adapters.
    """
    F = _validate_columns(name, _eval_basis(F))
    k = int(F.shape[1])
    if not isinstance(index, dict) or len(index) != 1:
        raise ValueError(
            f"array_block '{name}': index must be a one-key dict "
            f"{{name: slice(0, k)}}; got {index!r}")
    (key, sli), = index.items()
    if not (isinstance(sli, slice) and (sli.start, sli.stop) == (0, k)):
        raise ValueError(
            f"array_block '{name}': index slice for '{key}' must be "
            f"slice(0, {k}); got {sli!r}")
    cp = _conditioner_precision_from_spec(conditioner_precision, k, name)
    return TransportBlock(name, F, {key: slice(0, k)}, cp)


def globalgp_curn_block(globalgp, psr_slot, npsr):
    """Per-pulsar conditioner view of a DENSE global GP: elementwise
    reciprocal of the dense Phi diagonal, reshaped per pulsar -- the existing
    decenter convention, kept for parity.

    INVERSE MARGINAL VARIANCE, NOT DENSE PRIOR PRECISION.
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
        diagonal = _as_bake(kh.jnp.diag(getN(params)))
        return (diagonal ** -1).reshape((_n, -1))[_i]
    conditioner_precision.params = list(getattr(getN, 'params', []))

    name = getattr(globalgp, 'gpname', 'gw')
    return TransportBlock(name, F,
                          _single_key_index(globalgp.index, psr_slot,
                                            F.shape[1], name),
                          conditioner_precision,
                          conditioner_kind="curn_inverse_marginal")


def _legacy_globalgp_curn_precision(getN, psr_slot, npsr):
    """Previous reciprocal-then-diagonal CURN view. Test/benchmark reference."""
    def conditioner_precision(params, _i=psr_slot, _n=npsr):
        return kh.jnp.diag(getN(params) ** -1).reshape((_n, -1))[_i]
    conditioner_precision.params = list(getattr(getN, 'params', []))
    return conditioner_precision


# --------------------------------------------------------------------------
# reference noise
# --------------------------------------------------------------------------

class _FrozenSolve:
    """One-method reference-noise operator: solve(rhs) -> (N0^-1 rhs, logdet).
    Built at construction; contains no free parameters by construction.

    ``diagonal`` is the exact diagonal of the reference covariance ``N0`` in
    canonical TOA order, retained so the geometry certifier can standardize a
    per-TOA residual remainder. It is stored, never reconstructed
    from live noise parameters."""
    def __init__(self, solve_fn, description, diagonal=None):
        self._solve, self.description = solve_fn, description
        self._diagonal = None if diagonal is None else np.asarray(
            diagonal, dtype=np.float64)

    def solve(self, rhs):
        return self._solve(rhs)

    def diagonal(self):
        """Return the exact ``diag(N0)`` in canonical TOA order."""
        if self._diagonal is None:
            raise NotImplementedError(
                f"reference noise '{self.description}' was built without an "
                f"exact diagonal; diag(N0) is unavailable")
        return self._diagonal


def _frozen_kernel_diagonal(kernel, params0):
    """Exact ``diag(N0)`` for a frozen metamath reference kernel.

    Supports the two families used as timing references: a plain diagonal/dense
    ``NoiseMatrix`` and the Sherman-Morrison ECORR ``NoiseMatrixSM``
    (``diag(N) + F P F^T`` with a 0/1 exposure ``F``, so the diagonal adds
    ``F P`` epoch-wise). Any other kernel raises ``NotImplementedError`` — the
    type is checked before touching ``kernel.N``, so composite kernels do not
    fall into ``_materialize``.
    """
    if not isinstance(kernel, (metamath.NoiseMatrix, metamath.NoiseMatrixSM)):
        raise NotImplementedError(
            f"exact diag(N0) is not implemented for reference kernel "
            f"{type(kernel).__name__}; supply reference_noise built from a "
            f"NoiseMatrix or NoiseMatrixSM")
    Nc, Nf = metamath._materialize(kernel.N)
    diag = np.asarray(Nc if Nf is None else Nf(params=params0), dtype=np.float64)
    if diag.ndim == 2:
        diag = np.diagonal(diag).astype(np.float64)
    if isinstance(kernel, metamath.NoiseMatrixSM):
        Pc, Pf = metamath._materialize(kernel.P)
        P = np.asarray(Pc if Pf is None else Pf(params=params0), dtype=np.float64)
        F = np.asarray(kernel.F, dtype=np.float64)
        diag = diag + (F * F) @ P
    return diag


def _live_kernel_diagonal(kernel, params):
    """Exact ``diag(C(params))`` for a LIVE marginalized metamath kernel.

    The eta-dependent analogue of :func:`_frozen_kernel_diagonal`: recurse through
    the folded Woodbury stack of a marginalized ``PulsarLikelihood.N`` at the
    current ``params``. Used only by
    :meth:`MarginalTransport.live_kernel_standard_deviation` for the decentered-
    mode geometry certifier; never inside NUTS. Never rebuilds ``F``/``P`` from
    signal factories or the pulsar -- reads only the kernel's own attributes.
    """
    # NoiseMatrixSM (ECORR) leaf: same base as the frozen path, but LIVE params.
    if isinstance(kernel, metamath.NoiseMatrixSM):
        Nc, Nf = metamath._materialize(kernel.N)
        diag = np.asarray(Nc if Nf is None else Nf(params=params), dtype=np.float64)
        if diag.ndim == 2:
            diag = np.diagonal(diag).astype(np.float64)
        Pc, Pf = metamath._materialize(kernel.P)
        P = np.asarray(Pc if Pf is None else Pf(params=params), dtype=np.float64)
        F = np.asarray(kernel.F, dtype=np.float64)
        return diag + (F * F) @ P
    # Projection / float32 path: out of scope for v1.
    if isinstance(kernel, metamath.WoodburyProjKernel):
        raise NotImplementedError(
            "_live_kernel_diagonal: WoodburyProjKernel (projection / float32 "
            "path) is not supported in v1; the float64 marginalized "
            "psl.N uses the 1e40 improper GP via WoodburyKernel.")
    # C = N_inner + F P F^T. Recurse on .N only; add diag(F P F^T).
    if isinstance(kernel, metamath.WoodburyKernel):
        base = _live_kernel_diagonal(kernel.N, params)
        F = np.asarray(kernel.F, dtype=np.float64)
        # diag(P(params)) from the prior kernel's own array (NoiseMatrix family).
        # A 1e40-scale improper prior contributes 1e40*(M_f (X) M_f)@1 -- FLOAT64
        # ONLY; it overflows float32 (use the projection path there).
        prior_arr = getattr(kernel.P, "N", None)
        if prior_arr is None:
            raise TypeError(
                f"_live_kernel_diagonal: WoodburyKernel prior "
                f"{type(kernel.P).__name__} exposes no .N array; a metamath "
                f"NoiseMatrix-family prior is required")
        Pc, Pf = metamath._materialize(prior_arr)
        Pmat = np.asarray(Pc if Pf is None else Pf(params=params), dtype=np.float64)
        if Pmat.ndim == 1:
            contrib = (F * F) @ Pmat                          # diagonal prior
        else:
            contrib = np.einsum("ij,jk,ik->i", F, Pmat, F)    # dense prior row form
        return base + contrib
    # Plain diagonal / dense noise leaf (NoiseMatrix, incl. NoiseMatrix1D).
    if isinstance(kernel, metamath.NoiseMatrix):
        Nc, Nf = metamath._materialize(kernel.N)
        diag = np.asarray(Nc if Nf is None else Nf(params=params), dtype=np.float64)
        if diag.ndim == 2:
            diag = np.diagonal(diag).astype(np.float64)
        return diag
    raise TypeError(
        f"_live_kernel_diagonal: unsupported kernel type "
        f"{type(kernel).__name__}; requires a metamath Woodbury stack "
        f"(NoiseMatrix / NoiseMatrixSM / WoodburyKernel).")


def reference_noise(psr):
    """Diagonal TOA-error reference: N0 = diag(toaerrs**2). EFAC=1, no
    EQUAD/ECORR. The dependency-free default."""
    n0 = np.asarray(psr.toaerrs, dtype=np.float64) ** 2
    kernel = metamath.NoiseMatrix(kh.jnparray(n0))
    f = metamatrix.func(kernel.make_solve, working=bake_dtype())
    return _FrozenSolve(lambda rhs: f(rhs, params={}),
                        f"toaerrs diagonal ({psr.name})",
                        diagonal=n0)


def reference_noise_frozen(kernel, params0, description=None):
    """Freeze ANY metamath kernel's solve at explicit reference parameters
    params0 (e.g. a noisedict): works for measurement noise incl.
    Sherman-Morrison ECORR. This is the sanctioned way to FREEZE a
    parameterized kernel in a transport (never call a live kernel with
    params={}); `class_tracking` is the sanctioned way to let its white-noise
    parameters vary."""
    params0 = dict(params0)  # freeze a snapshot; caller mutation cannot alter N0
    make_solve = getattr(kernel, 'make_solve', None)
    if make_solve is None or not isinstance(make_solve, dict):
        raise TypeError(
            f"reference_noise_frozen: expected a metamath kernel exposing "
            f"make_solve as a graph, got {type(kernel).__name__}. "
            f"(Legacy matrix.py kernels are not supported; build the model "
            f"under discovery.config(kernels='metamath').)")
    f = metamatrix.func(make_solve, working=bake_dtype())
    missing = [p for p in f.params if p not in params0]
    if missing:
        raise ValueError(f"reference_noise_frozen: params0 is missing "
                         f"{missing}; a frozen reference must pin every "
                         f"parameter of the kernel it freezes.")
    try:
        diagonal = _frozen_kernel_diagonal(kernel, params0)
    except NotImplementedError:
        diagonal = None
    return _FrozenSolve(lambda rhs: f(rhs, params=params0),
                        description or f"frozen kernel at {sorted(params0)}",
                        diagonal=diagonal)


class ClassTracking(_FrozenSolve):
    """Reference-noise operator that ALSO bakes a class-quantized tracker.

    `.solve` / `.diagonal()` / `.description` are the frozen kernel at params0,
    so every construction-time bake (G0, b0, E0, certifier probes) is exactly
    what `reference_noise_frozen` gives. `.bake(W, r0)` returns the
    `classgram.ClassGram` the transport evaluates per step instead of (G0, b0).
    """
    def __init__(self, kernel, params0, struct, layout, solve_fn, description,
                 diagonal, validate, sigma_bin_dex, dense_threshold):
        super().__init__(solve_fn, description, diagonal=diagonal)
        self._kernel, self._params0 = kernel, dict(params0)
        self._struct, self._layout = struct, layout
        self._validate = bool(validate)
        self.params = list(struct.params)
        self.sigma_bin_dex = float(sigma_bin_dex)
        self.dense_threshold = int(dense_threshold)
        import hashlib
        import json
        self.params0_digest = "sha256:" + hashlib.sha256(json.dumps(
            {k: float(v) for k, v in sorted(self._params0.items()) if k in self.params},
            sort_keys=True).encode("utf-8")).hexdigest()[:16]

    @property
    def layout(self):
        return self._layout

    @property
    def params0(self):
        return dict(self._params0)

    def bake(self, W, r0):
        from . import classgram
        cg = classgram.ClassGram(W, r0, self._struct, self._params0, self._layout)
        if self._validate:
            f = metamatrix.func(self._kernel.make_solve, working=bake_dtype())
            classgram.validate_class_gram(
                cg, W, r0, lambda rhs, p: f(rhs, params=p), self._params0)
        return cg


# backwards-compatible private alias (the public marker is ClassTracking)
_ClassTracking = ClassTracking


def class_tracking(kernel, params0, *, toaerrs, sigma_bin_dex=0.2,
                   dense_threshold=16, validate=True, description=None):
    """Reference noise that tracks white-noise parameters in the transport.

    The transport's Gram becomes the exact Gram of a class-quantized
    white-noise model baked at ``params0``: per backend, TOAs are binned in
    ``log10 toaerr^2`` with width ``sigma_bin_dex``; bins holding at least
    ``dense_threshold`` TOAs share one precision ratio (a baked k x k class),
    every other TOA is kept exactly; ECORR is exact per epoch. Exact at
    ``params0`` and for every EFAC move, positive definite for every value of
    the white-noise parameters; the only approximation is the shared ratio
    inside a baked class.

    ``kernel``  : the canonical WHITE-noise kernel (NoiseMatrix1D / NoiseMatrixSM),
                  i.e. ``PulsarLikelihood.white_noise_kernel`` (ECORR-as-GP already
                  folded in).
    ``params0`` : pins every parameter of that kernel; the bake point. Use the
                  empirical-Bayes / MPE dictionary, never toaerrs (the chart is
                  exact at params0 and degrades smoothly away from it).
    ``toaerrs`` : the pulsar's TOA uncertainties (seconds); defines the sigma bins.
    ``sigma_bin_dex``, ``dense_threshold``: bin width and minimum bin population
                  for a baked class (measured defaults 0.2 dex, 16; raising the
                  threshold improves geometry at a flop cost).
    """
    from . import classgram
    params0 = dict(params0)
    make_solve = getattr(kernel, "make_solve", None)
    if make_solve is None or not isinstance(make_solve, dict):
        raise TypeError(
            f"class_tracking: expected a metamath kernel exposing make_solve as a "
            f"graph; got {type(kernel).__name__}. (Legacy matrix.py kernels are "
            f"not supported; build the model under "
            f"discovery.config(kernels='metamath').)")
    f = metamatrix.func(make_solve, working=bake_dtype())
    missing = [p for p in f.params if p not in params0]
    if missing:
        raise ValueError(f"class_tracking: params0 is missing {missing}; the bake "
                         f"point must pin every white-noise parameter of the kernel")
    struct = classgram.measurement_structure(kernel, params0)
    layout = classgram.build_layout(struct, params0, toaerrs,
                                    sigma_bin_dex=sigma_bin_dex,
                                    dense_threshold=dense_threshold)
    try:
        diagonal = _frozen_kernel_diagonal(kernel, params0)
    except NotImplementedError:
        diagonal = None
    desc = description or (
        f"class-tracked white noise at {sorted(params0)} "
        f"(bin {sigma_bin_dex} dex, threshold {dense_threshold}: "
        f"{layout.n_class} classes, {layout.n_dense} dense rows, "
        f"{struct.n_epoch} epochs)")
    return ClassTracking(kernel, params0, struct, layout,
                          lambda rhs: f(rhs, params=params0), desc, diagonal,
                          validate, sigma_bin_dex, dense_threshold)


# --------------------------------------------------------------------------
# the Transport object
# --------------------------------------------------------------------------

_KINDS = ("exact_diagonal", "curn_inverse_marginal")


def _require_psd_gram(G0, description):
    """G0 = W^T N0^-1 W is a Gram matrix and must be PSD up to bake roundoff.

    A materially negative eigenvalue means the reference-noise solve was not
    accurate enough (e.g. a float32 solve through the timing-model Woodbury):
    A = G0 + diag(p) then goes indefinite as soon as any p_i < |lambda_min|,
    i.e. for perfectly legal hyperparameters, and NUTS sees NaN log-densities
    instead of an error. Diagnose it here, at construction."""
    lam = np.linalg.eigvalsh(G0)
    scale = max(float(np.max(np.abs(lam))), np.finfo(np.float64).tiny)
    if lam[0] < -_G0_PSD_RTOL * scale:
        raise ValueError(
            f"transport: baked Gram W^T N0^-1 W is indefinite "
            f"(lambda_min={lam[0]:.3e}, lambda_max={lam[-1]:.3e}, ratio "
            f"{lam[0] / scale:.1e} < -{_G0_PSD_RTOL:.0e}); the reference-noise "
            f"solve ({description}) is not accurate enough to bake from. "
            f"Bake in float64 (see transport.bake_dtype()).")


class Transport:
    """Per-pulsar frozen-reference transport  xi -> (q, ldJ).

    q = mu(params) + L(params)^-T xi,
    A(params) = G0 + diag(conditioner_precision(params)),  A = L L^T,
    ldJ = -sum(log diag L).
    """

    def __init__(self, blocks, *, reference_noise, reference_residual=None,
                 origin="conditional_mode", origin_extsignals=None,
                 psr_slot=None, softclip=None):
        _kernels.require_metamath("Transport")

        # Resolved and stored once, before anything else runs: no bare
        # parameter survives to be shadowed, and `self.origin` is the only
        # reader downstream.
        self.origin = _resolve_origin(origin, what="Transport")
        centered = self.origin == "conditional_mode"

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
        column, self.index = 0, {}
        self._block_slice = {}       # block name -> assembled q-slice (softclip)
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
                self.index[key] = slice(column + sli.start,
                                        column + sli.stop)
                if b.name in self._block_slice:
                    raise ValueError(
                        f"duplicate block name '{b.name}'; softclip and "
                        f"diagnostics key blocks by name, so names must be "
                        f"unique")
                self._block_slice[b.name] = self.index[key]
            column += np.asarray(b.F).shape[1]
        self.dimension = column

        # -- bake (construction-time boundary work) ---------------------------
        W = np.concatenate([np.asarray(b.F, dtype=np.float64)
                            for b in self.blocks], axis=1)
        self._W = _as_bake(W)                # retained for diagnostics/inverse
        N0mW, _ = reference_noise.solve(self._W)
        N0mW = np.asarray(N0mW, dtype=np.float64)
        if not bool(np.all(np.isfinite(N0mW))):
            raise ValueError(f"reference-noise solve produced non-finite "
                             f"values ({reference_noise.description})")
        G0 = W.T @ N0mW
        G0 = 0.5 * (G0 + G0.T)
        _require_psd_gram(G0, reference_noise.description)
        self._G0 = _as_bake(G0)
        self.reference_description = reference_noise.description
        # Retain the frozen reference-noise operator so the geometry certifier
        # can form N0^-1 quadratics and diag(N0) without reconstructing white/
        # ECORR noise from a notebook dictionary.
        self._reference_noise = reference_noise

        self._b0 = None
        if centered:
            if reference_residual is None:
                raise ValueError(
                    'origin="conditional_mode" requires reference_residual')
            r0 = np.asarray(reference_residual, dtype=np.float64)
            if r0.shape != (self._ntoa,):
                raise ValueError(f"reference_residual has shape {r0.shape}; "
                                 f"expected ({self._ntoa},)")
            self._b0 = _as_bake(N0mW.T @ r0)

        # -- optional white-noise tracking (class-quantized Gram) -----------
        # G0/b0 above stay frozen at params0 (certifier probes, diagnostics);
        # with tracking, the per-evaluation factor uses the tracked (G, b).
        self._tracking = None
        if isinstance(reference_noise, ClassTracking):
            r_bake = (np.asarray(reference_residual, dtype=np.float64)
                      if centered else np.zeros(self._ntoa))
            self._tracking = reference_noise.bake(W, r_bake)
            self._track_arrays = tuple(None if x is None else _as_bake(x)
                                       for x in self._tracking.arrays())

        if self._tracking is not None and origin_extsignals:
            raise NotImplementedError(
                "class_tracking with origin_extsignals is not supported: the ExtSignal "
                "centering term E0 = W^T N0^-1 F_ext is formed from the frozen reference "
                "noise, while tracking makes b = W^T N(theta)^-1 r0 live — the mixed "
                "centering translation is silently wrong. Form E0 through the tracked "
                "operator before enabling this combination.")

        # -- ExtSignal-subtracted centering ----------------------------------
        # Bake E0_e = W^T N0^-1 Fext_i once; per eval subtract E0_e @ coeffs_e[i]
        # from b0 before the centering solve. A translation only; ldJ unchanged.
        self._extsignals = []
        ext_params = []
        if origin_extsignals:
            if not centered:
                raise ValueError(
                    'origin_extsignals requires origin="conditional_mode" '
                    "(it moves the translation)")
            N0mW_np = np.asarray(N0mW)
            for es in origin_extsignals:
                esname = getattr(es, 'name', 'extsignal')
                Fs = getattr(es, 'Fs', None)
                coeffs = getattr(es, 'coeffs', None)
                if Fs is None or coeffs is None:
                    raise TypeError(
                        f"origin_extsignals entry '{esname}' must expose .Fs "
                        f"and .coeffs (a discovery ExtSignal)")
                if psr_slot is None:
                    if len(Fs) != 1:
                        raise ValueError(
                            f"origin_extsignals '{esname}': psr_slot is "
                            f"required to select one of {len(Fs)} pulsar bases")
                    slot = 0
                else:
                    slot = psr_slot
                if slot < 0 or slot >= len(Fs):
                    raise ValueError(
                        f"origin_extsignals '{esname}': psr_slot={slot} out of "
                        f"range for {len(Fs)} pulsar bases")
                Fext = np.asarray(Fs[slot], dtype=np.float64)
                if Fext.ndim != 2 or Fext.shape[1] < 1:
                    raise ValueError(
                        f"origin_extsignals '{esname}': Fs[{slot}] must be "
                        f"2-D (n_toa, k_ext) with k_ext >= 1; got "
                        f"shape {Fext.shape}")
                if Fext.shape[0] != self._ntoa:
                    raise ValueError(
                        f"origin_extsignals '{esname}': Fs[{slot}] has "
                        f"{Fext.shape[0]} rows; expected n_toa={self._ntoa}")
                E0 = _as_bake(N0mW_np.T @ Fext)             # (k, k_ext)
                self._extsignals.append((E0, coeffs, slot, esname))
                ext_params += list(getattr(coeffs, 'params', []))

        # -- soft-clamp on named blocks' centering slices --------------------
        self._softclip = []
        if softclip:
            if not centered:
                raise ValueError(
                    'softclip requires origin="conditional_mode"; it clamps '
                    'the translation, which is zero under origin="zero"')
            for bname, zmax in dict(softclip).items():
                if bname not in self._block_slice:
                    raise ValueError(
                        f"softclip names unknown block {bname!r}; blocks: "
                        f"{sorted(self._block_slice)}")
                if not (float(zmax) > 0.0):
                    raise ValueError(
                        f"softclip[{bname!r}]={zmax!r} must be positive")
                self._softclip.append(
                    (bname, self._block_slice[bname], float(zmax)))

        self.params = sorted(set(
            sum([list(b.conditioner_precision.params) for b in self.blocks], [])
            + ext_params
            + (list(self._tracking.params) if self._tracking is not None else [])))

    # -- per-evaluation map (plain JAX; composes as a reparam FuncLeaf) ------
    def _gram(self, params):
        """(G, b) for this evaluation: frozen (G0, b0) or class-tracked."""
        if self._tracking is None:
            return self._G0, self._b0
        G, b = self._tracking.gram(params, kh.jnp, arrays=self._track_arrays)
        return G, (b if self._b0 is not None else None)

    def _factor(self, params):
        pinv = kh.jnp.concatenate(
            [b.conditioner_precision(params) for b in self.blocks])
        G, b = self._gram(params)
        i1, i2 = kh.jnp.diag_indices(self.dimension)
        return (kh.jsp.linalg.cho_factor(G.at[i1, i2].add(pinv), lower=True),
                pinv, b)

    def apply(self, params, xi):
        cf, _, b = self._factor(params)
        q = kh.jsp.linalg.solve_triangular(cf[0], xi, trans=1, lower=cf[1])
        ldJ = -kh.jnp.sum(kh.jnp.log(kh.jnp.diag(cf[0])))
        if b is not None:
            rhs = b
            for E0, coeffs, slot, _name in self._extsignals:
                rhs = rhs - E0 @ kh.jnp.asarray(coeffs(params))[slot]
            mu = kh.jsp.linalg.cho_solve(cf, rhs)
            for _name, sli, zmax in self._softclip:
                mu = mu.at[sli].set(zmax * kh.jnp.tanh(mu[sli] / zmax))
            q = q + mu
        return q, ldJ

    def reference_noise_quadratic(self, vector):
        """Return ``vector^T N0^-1 vector`` under the frozen reference noise.

        Used by the geometry certifier's global residual-remainder RMS.
        """
        vector = kh.jnp.asarray(vector)
        if vector.shape != (self._ntoa,):
            raise ValueError(
                f"reference_noise_quadratic expects shape ({self._ntoa},); "
                f"got {tuple(vector.shape)}")
        solved, _ = self._reference_noise.solve(vector)
        out = vector @ solved
        if not bool(kh.jnp.isfinite(out)):
            raise ValueError("reference_noise_quadratic produced non-finite "
                             "output")
        return out

    def reference_noise_standard_deviation(self):
        """Return ``sqrt(diag(N0))`` in canonical TOA order."""
        diag = np.asarray(self._reference_noise.diagonal(), dtype=np.float64)
        if diag.shape != (self._ntoa,):
            raise ValueError(
                f"reference noise diagonal has shape {diag.shape}; expected "
                f"({self._ntoa},)")
        if not bool(np.all(diag > 0.0)) or not bool(np.all(np.isfinite(diag))):
            raise ValueError("reference noise diagonal must be finite and "
                             "strictly positive")
        return diag ** 0.5

    def split(self, q):
        """{coefficient-key: q[slice]} view, in self.index order."""
        return {key: q[sli] for key, sli in self.index.items()}

    def as_reparam(self):
        """The metamath reparam contract: rp(params, c) -> (c_out, ldL), with
        TRUE .params."""
        def rp(params, c):
            return self.apply(params, c)
        rp.params = list(self.params)
        return rp

    def validate(self, params):
        """Eager, non-JIT positivity/PD check. Raises ValueError with
        per-block diagnostics; returns the diagnostics dict on success."""
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
                    f"fix the prior or remove the block.")
            offset += kb
        cf, _, _b = self._factor(params)
        diag = np.asarray(kh.jnp.diag(cf[0]))
        if not np.all(np.isfinite(diag)) or np.any(diag <= 0.0):
            raise ValueError(
                f"transport factorization is not positive definite at the "
                f"given params (min Cholesky diagonal "
                f"{np.nanmin(diag):.3e}). A conditioned direction has no "
                f"prior support or the basis is degenerate; no floor is "
                f"applied. Blocks: "
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
            "origin": self.origin,
            "reference_noise": self.reference_description,
        }
        if self._extsignals:
            out["origin_extsignals"] = [name for *_, name in self._extsignals]
        if self._softclip:
            out["softclip"] = {name: zmax for name, _sli, zmax in self._softclip}
        if self._tracking is not None:
            lay = self._tracking.layout
            ref = self._reference_noise
            out["tracking"] = {
                "kind": "class_quantized_white_noise",
                "params": list(self._tracking.params),
                "n_classes": int(lay.n_class),
                "n_dense": int(lay.n_dense),
                "n_epoch": int(self._tracking.n_epoch),
                "sigma_bin_dex": float(ref.sigma_bin_dex),
                "dense_threshold": int(ref.dense_threshold),
                # bake-point VALUES, not just keys: a chart baked at a different
                # dictionary is a different chart and must not reconcile
                "params0_digest": ref.params0_digest,
            }
        if params is None:
            if noise_solve is not None:
                raise ValueError("diagnostics(noise_solve=...) requires params")
            return out

        cf, pinv, _b = self._factor(params)
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

    def fingerprint(self):
        """Stable structural digest of this transport (block names/dims/order,
        parameter dependencies, centering, reference-noise description).

        Digests only the structure returned by ``diagnostics()`` (no params),
        so it is independent of any particular hyperparameter draw. Consumers
        (e.g. the nltiming dynamic run manifest) persist this so a saved run's
        transport can be reconciled without serializing an opaque closure.
        """
        import hashlib
        import json

        payload = json.dumps(
            {"schema": "discovery-transport-v1", "structure": self.diagnostics()},
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# ArrayTransport
# --------------------------------------------------------------------------

def _batched_extsignals(transports):
    """Stack per-pulsar Transport._extsignals into (E0, coeffs, name) triples.

    E0 is (npsr, k, k_ext). coeffs is the shared ExtSignal.coeffs callable
    returning (npsr, k_ext). All-or-none; same names, same coeffs identity,
    psr_slot=i in pulsar order, equal k_ext.
    """
    flags = [bool(getattr(t, "_extsignals", None)) for t in transports]
    if not any(flags):
        return []
    if not all(flags):
        raise ValueError(
            "ArrayTransport requires all-or-none ExtSignal origins")
    n_es = {len(t._extsignals) for t in transports}
    if len(n_es) != 1:
        raise ValueError(
            "ArrayTransport ExtSignal origins: pulsars disagree on the "
            f"number of ExtSignals {sorted(n_es)}")
    npsr = len(transports)
    k = transports[0].dimension
    batched = []
    for e in range(n_es.pop()):
        names = [t._extsignals[e][3] for t in transports]
        if len(set(names)) != 1:
            raise ValueError(
                f"ArrayTransport ExtSignal names disagree at entry {e}: "
                f"{names}")
        coeffs_list = [t._extsignals[e][1] for t in transports]
        if len({id(c) for c in coeffs_list}) != 1:
            raise ValueError(
                "ArrayTransport ExtSignal origins require the same "
                "coeffs callable on every pulsar (pass the same ExtSignal "
                "list to each Transport)")
        slots = [t._extsignals[e][2] for t in transports]
        if slots != list(range(npsr)):
            raise ValueError(
                "ArrayTransport ExtSignal origins require psr_slot=i "
                f"in pulsar order; got slots {slots}")
        E0s = [np.asarray(t._extsignals[e][0]) for t in transports]
        if any(E.ndim != 2 for E in E0s):
            raise ValueError(
                f"ArrayTransport ExtSignal '{names[0]}': each E0 must be "
                f"2-D (k, k_ext); got {[E.shape for E in E0s]}")
        shapes = {E.shape for E in E0s}
        if len(shapes) != 1:
            raise ValueError(
                f"ArrayTransport ExtSignal '{names[0]}' has unequal E0 "
                f"shapes across pulsars: {sorted(shapes)}")
        (k_i, k_ext), = shapes
        if k_i != k:
            raise ValueError(
                f"ArrayTransport ExtSignal '{names[0]}': E0 has {k_i} "
                f"rows; transport dimension is {k}")
        batched.append((kh.jnparray(np.stack(E0s)), coeffs_list[0], names[0]))
    return batched


def _stacked_array_conditioner(transports):
    """Current stacked per-pulsar conditioner, used as the default batched form."""
    def precision(params):
        return kh.jnp.stack([
            kh.jnp.concatenate([
                block.conditioner_precision(params)
                for block in transport.blocks
            ])
            for transport in transports
        ])
    precision.params = sorted(set().union(
        *(set(transport.params) for transport in transports)
    ))
    return precision


def gp_array_conditioner(gp):
    getN = gp.Phi.getN

    def precision(params):
        return _as_bake(getN(params)) ** -1

    precision.params = list(getN.params)
    return precision


def globalgp_curn_array_conditioner(globalgp, npsr):
    separable = getattr(globalgp, "separable_prior", None)
    if separable is not None:
        def precision(params):
            return separable.marginal_precision(params)

        precision.params = list(separable.params)
        return precision

    getN = globalgp.Phi.getN

    def precision(params):
        covariance = _as_bake(getN(params))
        return 1.0 / kh.jnp.diag(covariance).reshape((npsr, -1))

    precision.params = list(getN.params)
    return precision


def concatenate_array_conditioners(conditioners):
    def precision(params):
        return kh.jnp.concatenate(
            [conditioner(params) for conditioner in conditioners],
            axis=1,
        )

    precision.params = sorted(set().union(
        *(set(conditioner.params) for conditioner in conditioners)
    ))
    return precision


class ArrayTransport:
    """Per-pulsar transports behind one array reparam.

    Requirements (validated):
      - at least one transport;
      - EQUAL dimension k across pulsars (the array coefficient contract is a
        rectangular (npsr, k) array, so ragged transports cannot pass through
        it; ragged support is a non-goal);
      - all-or-none origin (a mixed origin would silently shift some
        pulsars' coordinates and not others');
      - all-or-none ExtSignal origins, with the same ExtSignal list in the
        same order on every pulsar (equal k_ext, shared coeffs callable,
        psr_slot=i);
      - no softclip (still per-pulsar only).
    """

    def __init__(self, transports, *, conditioner_precision=None):
        transports = list(transports)
        if not transports:
            raise ValueError("ArrayTransport requires at least one Transport")
        dims = sorted({t.dimension for t in transports})
        if len(dims) != 1:
            raise ValueError(
                f"ArrayTransport requires equal per-pulsar dimension; got "
                f"{dims}. Ragged transports are not supported; "
                f"use equal GP component counts across pulsars.")
        origins = {t.origin for t in transports}
        if len(origins) != 1:
            raise ValueError(
                "ArrayTransport requires all-or-none origin (a mixed origin "
                "would shift some pulsars and not others)")
        if any(getattr(t, "_softclip", None) for t in transports):
            raise ValueError("ArrayTransport does not support softclip")
        self._extsignals = _batched_extsignals(transports)

        self.transports = transports
        self.dimension = dims[0]
        self.npsr = len(transports)
        self.origin = origins.pop()
        self.params = sorted(set().union(*[set(t.params) for t in transports]))
        if conditioner_precision is None:
            conditioner_precision = _stacked_array_conditioner(self.transports)
        self._conditioner_precision = conditioner_precision
        self.params = sorted(
            set(self.params) | set(conditioner_precision.params)
        )

        # batched bake: (npsr, k, k) and (npsr, k). This is the SAME batched
        # arithmetic as the decenter closure: preserve its call conventions.
        self._G0 = kh.jnparray([t._G0 for t in self.transports])
        self._b0 = (kh.jnparray([t._b0 for t in self.transports])
                    if self.origin == "conditional_mode" else None)

        # -- optional white-noise tracking: zero-padded per-pulsar stacks ----
        tracks = [t._tracking for t in self.transports]
        if any(tr is not None for tr in tracks):
            if not all(tr is not None for tr in tracks):
                raise ValueError(
                    "ArrayTransport requires all-or-none white-noise tracking "
                    "(every per-pulsar Transport built with class_tracking, or none)")
            self._tracking = tracks
            k = self.dimension
            M = max(max(tr.A.shape[0] for tr in tracks), 1)
            D = max(max(tr.n_dense for tr in tracks), 1)
            E = max(max(tr.n_epoch for tr in tracks), 1)
            S = max((tr.Y0.shape[1] if tr.has_ecorr else 1) for tr in tracks)
            A = np.zeros((self.npsr, M, k, k))
            a = np.zeros((self.npsr, M, k))
            Fd = np.zeros((self.npsr, D, k))
            rd = np.zeros((self.npsr, D))
            Y0 = np.zeros((self.npsr, E, S, k))
            V0 = np.zeros((self.npsr, E, S))
            for i, tr in enumerate(tracks):
                A[i, :tr.A.shape[0]] = tr.A
                a[i, :tr.A.shape[0]] = tr.a
                Fd[i, :tr.n_dense] = tr.Fd
                rd[i, :tr.n_dense] = tr.rd
                if tr.has_ecorr:
                    Y0[i, :tr.n_epoch, :tr.Y0.shape[1]] = tr.Y0
                    V0[i, :tr.n_epoch, :tr.V0.shape[1]] = tr.V0
            self._A_stack, self._a_stack = _as_bake(A), _as_bake(a)
            self._Fd, self._rd = _as_bake(Fd), _as_bake(rd)
            self._Y0, self._V0 = _as_bake(Y0), _as_bake(V0)
            self._track_arrays = [t._track_arrays for t in self.transports]
            self._M, self._D, self._E, self._S = M, D, E, S
            self.params = sorted(
                set(self.params) | set().union(*[set(tr.params) for tr in tracks]))
        else:
            self._tracking = None

    def _pinv(self, params):
        value = kh.jnp.asarray(self._conditioner_precision(params))
        if value.shape != (self.npsr, self.dimension):
            raise ValueError(
                "batched conditioner precision has shape "
                f"{value.shape}; expected "
                f"({self.npsr}, {self.dimension})"
            )
        return value

    def _gram(self, params):
        """Batched (G, b): frozen stacks, or the class-tracked einsums."""
        if self._tracking is None:
            return self._G0, self._b0
        k, jnp = self.dimension, kh.jnp
        om, pd, om_e, Yadd, vadd, Ss = [], [], [], [], [], []
        for i, tr in enumerate(self._tracking):
            w = tr.batched_weights(params, jnp, arrays=self._track_arrays[i])
            om.append(jnp.pad(w.omega, (0, self._M - w.omega.shape[0])))
            pd.append(jnp.pad(w.p_dense, (0, self._D - w.p_dense.shape[0])))
            if w.S is None:
                om_e.append(jnp.zeros((self._E, self._S), dtype=w.omega.dtype))
                Yadd.append(jnp.zeros((self._E, k), dtype=w.omega.dtype))
                vadd.append(jnp.zeros(self._E, dtype=w.omega.dtype))
                Ss.append(jnp.ones(self._E, dtype=w.omega.dtype))
            else:
                E_i = w.S.shape[0]
                om_e.append(jnp.pad(w.omega_e, ((0, self._E - E_i),
                                                (0, self._S - w.omega_e.shape[1]))))
                Yadd.append(jnp.pad(w.Y_add, ((0, self._E - E_i), (0, 0))))
                vadd.append(jnp.pad(w.v_add, (0, self._E - E_i)))
                Ss.append(jnp.pad(w.S, (0, self._E - E_i), constant_values=1.0))
        om, pd, om_e, Yadd, vadd, S = map(jnp.stack, (om, pd, om_e, Yadd, vadd, Ss))
        Y = jnp.einsum("pej,pejk->pek", om_e, self._Y0) + Yadd
        v = jnp.sum(om_e * self._V0, axis=2) + vadd
        G = (jnp.einsum("pm,pmij->pij", om, self._A_stack)
             + jnp.einsum("pdk,pd,pdl->pkl", self._Fd, pd, self._Fd)
             - jnp.einsum("pek,pe,pel->pkl", Y, S, Y))
        b = (jnp.einsum("pm,pmi->pi", om, self._a_stack)
             + jnp.einsum("pdk,pd->pk", self._Fd, pd * self._rd)
             - jnp.einsum("pek,pe->pk", Y, S * v))
        G = 0.5 * (G + jnp.swapaxes(G, 1, 2))
        return G, (b if self.origin == "conditional_mode" else None)

    def apply(self, params, c):
        # c: (npsr, k) -- the array coefficient contract.
        i1, i2 = kh.jnp.diag_indices(self.dimension, ndim=2)
        G, b = self._gram(params)
        cf = kh.jsp.linalg.cho_factor(
            G.at[:, i1, i2].add(self._pinv(params)), lower=True)
        am = kh.jsp.linalg.solve_triangular(cf[0], c, trans=1, lower=cf[1])
        ldJ = -kh.jnp.logdet(cf[0][:, i1, i2])            # summed, as today
        if b is not None:
            rhs = b
            for E0, coeffs, _name in self._extsignals:
                ccw = kh.jnp.asarray(coeffs(params))      # (npsr, k_ext)
                rhs = rhs - kh.jnp.einsum("ijk,ik->ij", E0, ccw)
            am = am + kh.jsp.linalg.cho_solve(cf, rhs)
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
        out = [t.validate(params) for t in self.transports]
        for E0, coeffs, name in self._extsignals:
            ccw = np.asarray(coeffs(params))
            npsr, k_ext = self.npsr, int(np.asarray(E0).shape[-1])
            if ccw.shape != (npsr, k_ext):
                raise ValueError(
                    f"ArrayTransport ExtSignal '{name}': coeffs(params) has "
                    f"shape {ccw.shape}; expected ({npsr}, {k_ext})")
        batched = np.asarray(self._pinv(params))
        stacked = np.asarray(_stacked_array_conditioner(self.transports)(params))
        if batched.shape != stacked.shape:
            raise ValueError(
                "batched conditioner shape "
                f"{batched.shape} does not match stacked {stacked.shape}")
        if not np.allclose(batched, stacked, rtol=1e-12, atol=0.0):
            raise ValueError(
                "batched conditioner differs from the stacked per-pulsar "
                "reference")
        return out

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
        out = {
            "per_pulsar": [
                t.diagnostics(params=params, noise_solve=s)
                for t, s in zip(self.transports, solves)
            ],
            "dimension": self.dimension,
            "npsr": self.npsr,
            "params": list(self.params),
        }
        if self._extsignals:
            out["origin_extsignals"] = [name for *_, name in self._extsignals]
        return out

    def fingerprint(self):
        """Stable structural digest of the array transport (see
        :meth:`Transport.fingerprint`)."""
        import hashlib
        import json

        payload = json.dumps(
            {"schema": "discovery-array-transport-v1", "structure": self.diagnostics()},
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# MarginalTransport: live-kernel decentering of one external block
# --------------------------------------------------------------------------


class MarginalTransport:
    """Live-kernel decentering for ONE external block:  xi -> (z, ldJ).

    ``z = mu(params) + L(params)^-T xi``,
    ``A(params) = W^T C(params)^-1 W + diag(p(params))``,  ``A = L L^T``,
    ``mu = A^-1 W^T C^-1 y`` (origin="conditional_mode") or ``0``,  ``ldJ = -sum(log diag L)``.

    ``C(params)`` is the LIVE marginalized covariance of the supplied metamath
    kernel (white noise + folded GPs + timing marginal blocks): the eta-dependent
    posterior-metric whitening of the external block. Unlike the joint
    :class:`Transport` (frozen ``N0``, sampled GP coefficients), the coefficients
    are analytically integrated inside ``C`` and only the ``W`` block is sampled.

    For the nltiming timing use case, ``y`` MUST be
    ``TimingLinearization.transport_effective_residual(raw_residual)`` and
    ``block.F`` MUST be ``TimingLinearization.sampled_basis``. Both are fixed at
    context construction / expansion time and never depend on the sampled
    coordinate (D-INV). Do not pass a delay-modified CompoundDelay.
    """

    def __init__(self, kernel, y, block, *, origin="conditional_mode"):
        _kernels.require_metamath("MarginalTransport")

        make_ks = getattr(kernel, "make_kernelsolve", None)
        if make_ks is None:
            raise TypeError(
                "marginal_transport: kernel must be a metamath kernel exposing "
                "make_kernelsolve (pass the assembled PulsarLikelihood.N); got "
                f"{type(kernel).__name__}.")
        if callable(y):                                                   # D-INV
            raise TypeError(
                "marginal_transport: y must be a fixed residual array, not a "
                "delay-modified callable (CompoundDelay); a coordinate-dependent "
                "centering would break the triangular Jacobian (D-INV).")
        y = np.asarray(y, dtype=np.float64)

        if len(block.index) != 1:
            raise ValueError(
                f"MarginalTransport: block must have exactly one coefficient key "
                f"(a single array_block); got {len(block.index)}")
        (key, sli), = block.index.items()
        W = np.asarray(block.F, dtype=np.float64)          # validated by array_block
        if (sli.start, sli.stop) != (0, W.shape[1]):
            raise ValueError(
                f"MarginalTransport block '{key}': index slice must be "
                f"slice(0, {W.shape[1]}); got {sli}")
        if y.shape != (W.shape[0],):
            raise ValueError(f"y has shape {y.shape}; expected ({W.shape[0]},)")

        self.blocks = [block]                              # Transport-compatible
        self.index = {key: slice(0, W.shape[1])}
        self.dimension = int(W.shape[1])
        self._ntoa = int(W.shape[0])
        self.origin = _resolve_origin(origin, what="MarginalTransport")
        self._kernel = kernel
        self._W = _as_bake(W)
        self._y = _as_bake(y)

        self._ks = make_ks(self._y, self._W, working=bake_dtype())  # (W^T C^-1 y, W^T C^-1 W)
        # Live solve for the residual quadratic (NOT frozen at a snapshot).
        self._live_solve = metamatrix.func(kernel.make_solve, working=bake_dtype())
        self.params = sorted(
            set(self._ks.params) | set(block.conditioner_precision.params))

    # -- per-evaluation map -------------------------------------------------
    def _factor(self, params):
        b, G = self._ks(params)
        pinv = self.blocks[0].conditioner_precision(params)
        i1, i2 = kh.jnp.diag_indices(self.dimension)
        A = G.at[i1, i2].add(pinv)
        return kh.jsp.linalg.cho_factor(A, lower=True), b, pinv

    def apply(self, params, xi):
        cf, b, _ = self._factor(params)
        z = kh.jsp.linalg.solve_triangular(cf[0], xi, trans=1, lower=cf[1])
        ldJ = -kh.jnp.sum(kh.jnp.log(kh.jnp.diag(cf[0])))
        if self.origin == "conditional_mode":
            z = z + kh.jsp.linalg.cho_solve(cf, b)
        return z, ldJ

    def split(self, z):
        """{coefficient-key: z[slice]} view, in self.index order."""
        return {key: z[sli] for key, sli in self.index.items()}

    def as_reparam(self):
        """The metamath reparam contract: rp(params, c) -> (c_out, ldL), with
        TRUE .params (parity with Transport)."""
        def rp(params, c):
            return self.apply(params, c)
        rp.params = list(self.params)
        return rp

    # -- live-kernel geometry hooks (offline certifier only) ----------------
    def live_kernel_quadratic(self, params, vector):
        """Return ``vector^T C(params)^-1 vector`` under the LIVE marginal kernel.

        The eta-dependent analogue of ``Transport.reference_noise_quadratic``,
        via the SAME kernel graph -- no dense assembly, no second
        make_kernelsolve, no hand-rolled Woodbury.
        """
        vector = kh.jnp.asarray(np.asarray(vector, dtype=np.float64))
        if vector.shape != (self._ntoa,):
            raise ValueError(
                f"live_kernel_quadratic expects shape ({self._ntoa},); "
                f"got {tuple(vector.shape)}")
        xinv, _ld = self._live_solve(vector, params=params)
        out = float(vector @ xinv)
        if not np.isfinite(out):
            raise ValueError("live_kernel_quadratic produced non-finite output")
        return out

    def live_kernel_standard_deviation(self, params):
        """Return ``sqrt(diag(C(params)))`` in canonical TOA order.

        The eta-dependent analogue of
        ``Transport.reference_noise_standard_deviation``, computed by the
        recursive :func:`_live_kernel_diagonal` walk of the kernel's own Woodbury
        stack.
        """
        diag = np.asarray(_live_kernel_diagonal(self._kernel, params),
                          dtype=np.float64)
        if diag.shape != (self._ntoa,):
            raise ValueError(
                f"live kernel diagonal has shape {diag.shape}; expected "
                f"({self._ntoa},)")
        if not bool(np.all(diag > 0.0)) or not bool(np.all(np.isfinite(diag))):
            raise ValueError("live kernel diagonal must be finite and strictly "
                             "positive")
        return diag ** 0.5

    # -- validation / diagnostics / fingerprint -----------------------------
    def validate(self, params):
        """Eager positivity/PD check (no floors). Mirrors Transport.validate."""
        b = self.blocks[0]
        p = np.asarray(b.conditioner_precision(params))
        kb = np.asarray(b.F).shape[1]
        if p.shape != (kb,):
            raise ValueError(f"block '{b.name}': conditioner_precision returned "
                             f"shape {p.shape}; expected ({kb},)")
        bad = np.flatnonzero(~np.isfinite(p) | (p < 0.0))
        if bad.size:
            raise ValueError(
                f"block '{b.name}': conditioner_precision has non-finite or "
                f"negative entries at indices {bad.tolist()}; no floor is "
                f"applied.")
        cf, _b, _pinv = self._factor(params)
        diag = np.asarray(kh.jnp.diag(cf[0]))
        if not np.all(np.isfinite(diag)) or np.any(diag <= 0.0):
            raise ValueError(
                f"marginal transport factorization is not positive definite at "
                f"the given params (min Cholesky diagonal {np.nanmin(diag):.3e}); "
                f"no floor is applied.")
        return self.diagnostics(params)

    def diagnostics(self, params=None, noise_solve=None):
        """Structural report plus optional eager Cholesky diagnostics.

        ``noise_solve`` is NOT supported: the metric is live, so the honest
        geometry check is the identity / ``certify_decentered_geometry``, not a
        frozen-noise substitute.
        """
        if noise_solve is not None:
            raise ValueError(
                "MarginalTransport.diagnostics does not support noise_solve "
                "(the metric is live; use certify_decentered_geometry)")
        b = self.blocks[0]
        out = {
            "blocks": [
                {"name": b.name, "k": int(np.asarray(b.F).shape[1]),
                 "params": list(b.conditioner_precision.params),
                 "keys": list(b.index), "conditioner_kind": b.conditioner_kind}
            ],
            "dimension": self.dimension,
            "origin": self.origin,
            "reference_noise": "live_kernel",
            "kernel_measurement": getattr(self._kernel, "measurement", None),
        }
        if params is None:
            return out
        cf, _b, pinv = self._factor(params)
        p = np.asarray(pinv)
        d = np.asarray(kh.jnp.diag(cf[0]))
        out.update(
            precision_min=float(np.min(p)),
            precision_max=float(np.max(p)),
            chol_diag_min=float(np.min(d)),
            chol_diag_max=float(np.max(d)),
        )
        return out

    def fingerprint(self):
        """Stable structural digest under schema ``discovery-marginal-transport-v1``."""
        import hashlib
        import json

        payload = json.dumps(
            {"schema": "discovery-marginal-transport-v1",
             "structure": self.diagnostics()},
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def marginal_transport(kernel, y, block, *, origin="conditional_mode"):
    """Public factory for :class:`MarginalTransport`. See its docstring.

    ``kernel`` MUST be the assembled marginalized ``PulsarLikelihood.N`` (a
    metamath Woodbury *stack* exposing ``make_kernelsolve``) — the object whose
    ``C(params) = N0 + F Phi(params) F^T`` folds white noise together with the
    RN/DM and marginalized-timing GP priors. A bare white-noise leaf
    (``metamath.NoiseMatrix``) has no ``make_kernelsolve`` and is rejected with a
    ``TypeError``; do not "simplify" to the WN kernel — the eta-dependence of the
    transport lives entirely in the folded GP priors.
    """
    return MarginalTransport(kernel, y, block, origin=origin)
