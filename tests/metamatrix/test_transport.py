"""Graph-consistent transport.

`Transport` / `ArrayTransport` replace the in-likelihood decenter closure with a
free-standing reparam object that declares its true `.params`, carries a
mandatory exact conditioner precision for diagonal priors (no floors, no
ridges), and has specified failure semantics at construction / validate /
runtime.
"""
from pathlib import Path

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import discovery as ds  # noqa: E402
import discovery.recipes as R  # noqa: E402
from discovery import utils as kh  # noqa: E402
from discovery import metamatrix  # noqa: E402
from discovery import transport as tr  # noqa: E402


@pytest.fixture
def metamath_backend():
    ds.config(kernels="metamath")
    yield
    ds.config(kernels="matrix")


@pytest.fixture
def cholesky_backend(metamath_backend):
    """The transport parity anchor is the closure's DEFAULT Cholesky config."""
    ds.utils.config(backend="jax", factor="cholesky")
    assert ds.utils.matrix_norm == 2.0
    yield


# ==========================================================================
# 1. Closure parity — the deletion gate
# ==========================================================================

def _legacy_decenter_transform(model):
    """Standalone replica of the deleted decenter closure
    (likelihood_metamath.py at the pre-deletion commit;
    `legacy_decenter_transform`). Returns rp(params, c) -> (q, ldL). This is
    the golden source the transport is certified against."""
    vsm, ys = model._coefficient_assembly

    def _solve_2d(N, F):
        return metamatrix.func(N.make_solve)(F, params={})

    def _eval_F(F):
        if isinstance(F, dict):
            return kh.jnp.asarray(metamatrix.func(F)(params={}))
        return kh.jnp.asarray(F)

    vsm_Fs = [_eval_F(F) for F in vsm.Fs]
    NmFs, _ldNs = zip(*[_solve_2d(N, F) for N, F in zip(vsm.Ns, vsm_Fs)])
    FtNmFs = [F.T @ NmF for F, NmF in zip(vsm_Fs, NmFs)]
    NmFtys = [NmF.T @ y for NmF, y in zip(NmFs, ys)]
    FtNmF, NmFty = kh.jnparray(FtNmFs), kh.jnparray(NmFtys)

    def decenter_transform(params, c):
        cgp_list = (model.commongp if isinstance(model.commongp, list)
                    else [model.commongp])
        phis_invs_commongp = [gp.Phi.getN(params) ** -1 for gp in cgp_list]
        if model.globalgp is not None:
            phis_invs_globalgp = (kh.jnp.diag(
                model.globalgp.Phi.getN(params) ** -1
            ).reshape((len(model.psls), -1)))
            phis_invs = kh.jnp.concatenate(
                [*phis_invs_commongp, phis_invs_globalgp], axis=1)
        else:
            phis_invs = kh.jnp.concatenate([*phis_invs_commongp], axis=1)
        i1, i2 = kh.jnp.diag_indices(phis_invs.shape[1], ndim=2)
        cf = kh.matrix_factor(FtNmF.at[:, i1, i2].add(phis_invs), lower=True)
        am = kh.jsp.linalg.solve_triangular(cf[0], c, trans=1, lower=cf[1])
        mus = kh.matrix_solve(cf, NmFty)
        ldL = -kh.jnp.logdet(cf[0][:, i1, i2])
        return am + mus, ldL

    decenter_transform.params = []
    return decenter_transform


@pytest.mark.parametrize("recipe", ["decenter_intrinsic_rn",
                                    "decenter_intrinsic_rn_global_hd"])
def test_closure_parity_is_the_deletion_gate(psrs, cholesky_backend, recipe):
    """Transport-based reparam vs the closure replica at 20 draws.

    logp (the clogL value) and ldJ match to rtol=1e-12; the raw transformed
    coefficients match to rtol=1e-9. The looser coefficient tolerance is honest
    float64 behavior, not a defect: the transport bakes G0/b0 in NumPy while the
    closure builds them in JAX, and the two BLAS paths diverge at ~1e-11 on the
    ill-conditioned Cholesky solve. That divergence washes out of logp/ldJ,
    which is why the LIKELIHOOD matches to machine precision.
    """
    model = getattr(R, recipe)(psrs)
    legacy = _legacy_decenter_transform(model)
    transport = model._build_decenter_transport(model._coefficient_assembly[1])
    rp = transport.as_reparam()

    npsr = len(psrs)
    k = transport.dimension
    hyper = [p for p in sorted(model.clogL.params) if "_coefficients(" not in p]

    np.random.seed(20260716)
    for _ in range(20):
        params = ds.sample_uniform(hyper)
        xi = kh.jnparray(np.random.randn(npsr, k))

        q_old, ldj_old = legacy(params, xi)
        q_new, ldj_new = rp(params, xi)

        # Compare coefficients relative to their overall scale, not per element:
        # near-zero entries carry no meaningful relative tolerance.
        qo, qn = np.asarray(q_old), np.asarray(q_new)
        scale = np.max(np.abs(qo))
        assert np.max(np.abs(qn - qo)) <= 1e-9 * scale
        np.testing.assert_allclose(float(ldj_new), float(ldj_old), rtol=1e-12)


@pytest.mark.parametrize("recipe", ["decenter_intrinsic_rn",
                                    "decenter_intrinsic_rn_global_hd"])
def test_transport_clogl_matches_the_closure_likelihood(psrs, cholesky_backend, recipe):
    """The clogL VALUE under the transport equals the closure's, to rtol=1e-12
    (in practice ~1e-15). This is the parity that actually gates deletion."""
    model = getattr(R, recipe)(psrs)

    vsm, ys = model._coefficient_assembly
    from discovery.likelihood_metamath import ffunc
    closure_clogl = ffunc(vsm.make_kernelproduct_gpcomponent(
        ys, transform=[_legacy_decenter_transform(model)]))

    transport_clogl = model.clogL

    def val(fn, p):
        out = fn(p)
        return float(out[0]) if isinstance(out, tuple) else float(out)

    coeff = [p for p in transport_clogl.params if "_coefficients(" in p]
    hyper = [p for p in transport_clogl.params if p not in coeff]

    np.random.seed(20260716)
    for _ in range(20):
        p0 = ds.sample_uniform(hyper)
        for key in coeff:
            p0[key] = np.random.randn(int(key[key.index("(") + 1:key.index(")")]))
        np.testing.assert_allclose(val(transport_clogl, p0),
                                   val(closure_clogl, p0), rtol=1e-12)


# ==========================================================================
# helpers for the standalone-block tests
# ==========================================================================

def _commongp(psrs, components=10):
    return ds.makecommongp_fourier(psrs, ds.powerlaw, components=components,
                                   T=ds.getspan(psrs), name="rednoise")


def _globalgp(psrs, components=5):
    return ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf,
                                   components=components, T=ds.getspan(psrs),
                                   name="gw")


def _transport_for(psr, psrs, i, *, center=True, extra_globalgp=None):
    gp = _commongp(psrs)
    blocks = [tr.gp_block(gp, psr_slot=i)]
    if extra_globalgp is not None:
        blocks.append(tr.globalgp_curn_block(extra_globalgp, i, len(psrs)))
    return tr.Transport(
        blocks,
        reference_noise=tr.reference_noise(psr),
        reference_residual=np.asarray(psr.residuals) if center else None,
        center=center)


# ==========================================================================
# 2. Jacobian
# ==========================================================================

@pytest.mark.parametrize("center", [False, True])
def test_jacobian_matches_returned_ldJ(psr, metamath_backend, center):
    """jac slogdet of xi -> q equals the returned ldJ (rtol=1e-10), with and
    without centering (centering is a translation and must not change ldJ)."""
    psrs = [psr]
    t = _transport_for(psr, psrs, 0, center=center)
    params = ds.sample_uniform(t.params)

    def q_of_xi(xi):
        return t.apply(params, xi)[0]

    xi = kh.jnparray(np.random.default_rng(0).normal(size=t.dimension))
    J = jax.jacfwd(q_of_xi)(xi)
    sign, logdet = np.linalg.slogdet(np.asarray(J))

    _, ldJ = t.apply(params, xi)

    assert sign > 0
    np.testing.assert_allclose(logdet, float(ldJ), rtol=1e-10)


# ==========================================================================
# 3. Reparam contract — true .params
# ==========================================================================

def test_as_reparam_declares_the_true_params(psrs, metamath_backend):
    """The `decenter_transform.params = []` defect dies: the transport's reparam
    reports the union of its blocks' conditioner hyperparameters."""
    gp = _commongp(psrs)
    t = _transport_for(psrs[0], psrs, 0)

    expected = sorted(set(gp.Phi.getN.params))
    assert t.as_reparam().params == expected
    assert expected           # a powerlaw commongp has hyperparameters


# ==========================================================================
# 4. Scale matching — the anti-ridge regression
# ==========================================================================

def _unit_column_ref(n_toa, n0_scalar):
    """Reference-noise solve for a diagonal N0 = n0_scalar * I over `n_toa`."""
    n0 = np.full(n_toa, float(n0_scalar))

    def solve(rhs):
        rhs = np.asarray(rhs)
        d = n0[:, None] if rhs.ndim == 2 else n0
        return rhs / d, float(np.sum(np.log(n0)))

    return tr._FrozenSolve(solve, f"diag({n0_scalar})")


def _one_by_one_A(transport):
    """The scalar precision A = L^2 of a 1-block, 1-column transport at params={}."""
    cf, _, _b = transport._factor({})
    L = np.asarray(cf[0])
    return float(L[0, 0]) ** 2


def test_anti_ridge_whitening_at_exact_precision_only(metamath_backend):
    """The anti-ridge regression, tested on the TRANSFORMED POSTERIOR directly
    (not via the diagnostics metric, which structurally cannot see a ridge --
    it builds its target with the block's own precision, so the precision
    cancels in H/A).

    Physical posterior precision M = lambda + p (data curvature + TRUE prior
    precision). The transport whitens with A = lambda0 + conditioner. In xi
    coordinates (q = L^-T xi, A = L L^T) the transformed posterior eigenvalue is
    M / A. With the reference noise equal to the true noise (lambda0 = lambda):

        conditioner = p      -> M/A = (lambda+p)/(lambda+p)     = 1     (whitened)
        conditioner = 100 p  -> M/A = (lambda+p)/(lambda+100 p) != 1

    All quantities are O(1) so the two cases are far apart: a transport that
    floored/ridged the conditioner, or ignored it, changes A and fails here.
    """
    rng = np.random.default_rng(1)
    n_toa = 16
    w = rng.normal(size=(n_toa, 1))
    w = w / np.linalg.norm(w)                     # w^T w = 1  ->  lambda = 1
    ref = _unit_column_ref(n_toa, 1.0)            # N0 = N = I  ->  lambda0 = lambda

    lam = float(w[:, 0] @ w[:, 0])                # = 1 (unit column, N=I)
    p_true = 1.0
    M_phys = lam + p_true                         # TRUE posterior precision (indep. of conditioner)

    def transport(precision):
        return tr.Transport([_const_block(w, [precision], name="s")],
                            reference_noise=ref, center=False)

    # exact conditioner -> whitened.
    A_exact = _one_by_one_A(transport(p_true))
    np.testing.assert_allclose(M_phys / A_exact, 1.0, rtol=1e-10)

    # 100x-wrong conditioner -> (lambda+p)/(lambda+alpha), emphatically NOT 1.
    alpha = 100 * p_true
    A_wrong = _one_by_one_A(transport(alpha))
    np.testing.assert_allclose(M_phys / A_wrong, (lam + p_true) / (lam + alpha),
                               rtol=1e-10)
    assert abs(M_phys / A_wrong - 1.0) > 0.9      # genuinely un-whitened, no swamping


def test_diagnostics_metric_reports_reference_noise_mismatch(metamath_backend):
    """Honest coverage of `diagnostics(noise_solve=...)`. The metric it returns is
    (lambda + p)/(lambda0 + p) with lambda from the live noise solve and lambda0
    from the frozen reference -- i.e. it measures how well the reference noise N0
    approximates the true N (at the block's scale), and is exactly 1.0 when
    N0 = N. It does NOT (and cannot) reveal a conditioner ridge; the anti-ridge
    property is covered by `test_anti_ridge_whitening_at_exact_precision_only`.
    """
    rng = np.random.default_rng(3)
    n_toa = 16
    w = rng.normal(size=(n_toa, 1))
    w = w / np.linalg.norm(w)                     # w^T w = 1
    p = 3.0

    t = tr.Transport([_const_block(w, [p], name="s")],
                     reference_noise=_unit_column_ref(n_toa, 2.0), center=False)  # N0 = 2 I

    # true noise N = I  ->  lambda = 1, lambda0 = 1/2
    def true_solve(rhs):
        rhs = np.asarray(rhs)
        return rhs, 0.0                           # N = I

    d = t.diagnostics(params={}, noise_solve=true_solve)
    lam, lam0 = 1.0, 0.5
    np.testing.assert_allclose(d["metric_eig_min"], (lam + p) / (lam0 + p), rtol=1e-8)
    assert d["metric_kind"] == "local_target"     # no CURN block present

    # N0 == N  ->  metric exactly 1.
    d_match = t.diagnostics(params={},
                            noise_solve=lambda rhs: (np.asarray(rhs) / 2.0, 0.0))
    np.testing.assert_allclose(d_match["metric_eig_min"], 1.0, rtol=1e-8)


# ==========================================================================
# 5. Adapter correctness
# ==========================================================================

def test_gp_block_selects_the_pulsar_row(psrs, metamath_backend):
    gp = _commongp(psrs)
    params = ds.sample_uniform(gp.Phi.getN.params)

    block = tr.gp_block(gp, psr_slot=1)
    got = np.asarray(block.conditioner_precision(params))

    expected = np.asarray(gp.Phi.getN(params))[1] ** -1
    np.testing.assert_allclose(got, expected, rtol=1e-12)

    k = np.asarray(gp.F[1]).shape[1]
    assert got.shape == (k,)
    assert len(block.index) == 1
    (key, sli), = block.index.items()
    assert psrs[1].name in key and sli == slice(0, k)


def test_globalgp_curn_block_matches_the_reshaped_diagonal(psrs, metamath_backend):
    gg = _globalgp(psrs)
    params = ds.sample_uniform(gg.Phi.getN.params)

    block = tr.globalgp_curn_block(gg, 1, len(psrs))
    got = np.asarray(block.conditioner_precision(params))

    expected = np.asarray(
        kh.jnp.diag(gg.Phi.getN(params) ** -1).reshape((len(psrs), -1))[1])
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    assert block.conditioner_kind == "curn_inverse_marginal"


# ==========================================================================
# 6. Validation errors (each asserts the message content)
# ==========================================================================

def _diag_ref(psr):
    n0 = np.asarray(psr.toaerrs, dtype=np.float64) ** 2

    def solve(rhs):
        rhs = np.asarray(rhs)
        d = n0[:, None] if rhs.ndim == 2 else n0
        return rhs / d, float(np.sum(np.log(n0)))

    return tr._FrozenSolve(solve, "diag")


def _const_block(F, precision, name="b", kind="exact_diagonal"):
    F = np.asarray(F, dtype=np.float64)
    prec = np.atleast_1d(np.asarray(precision, dtype=np.float64))

    def cond(params):
        return kh.jnp.asarray(prec)
    cond.params = []
    return tr.TransportBlock(name, F, {f"{name}_coefficients({F.shape[1]})":
                                       slice(0, F.shape[1])}, cond, kind)


def test_zero_column_basis_raises_naming_indices(psr, metamath_backend):
    """The adapters' construction-time column validator (`_validate_columns`)
    names the offending column index. It is the gate the GP adapters run before
    building a block."""
    F = np.asarray(psr.residuals)[:, None] * np.array([[1.0, 0.0]])
    with pytest.raises(ValueError, match=r"zero or non-finite column.*\[1\]"):
        tr._validate_columns("rednoise", F)


def test_empty_block_list_raises(psr, metamath_backend):
    with pytest.raises(ValueError, match="at least one block"):
        tr.Transport([], reference_noise=_diag_ref(psr), center=False)


def test_multi_key_block_index_raises(psr, metamath_backend):
    F = np.asarray(psr.residuals)[:, None]

    def cond(params):
        return kh.jnp.array([1.0])
    cond.params = []
    bad = tr.TransportBlock("b", F, {"a(1)": slice(0, 1), "b(1)": slice(0, 1)},
                            cond)
    with pytest.raises(ValueError, match="exactly one coefficient key"):
        tr.Transport([bad], reference_noise=_diag_ref(psr), center=False)


def test_duplicate_coefficient_key_across_blocks_raises(psr, metamath_backend):
    F = np.asarray(psr.residuals)[:, None]
    b1 = _const_block(F, [1.0], name="dup")
    b2 = _const_block(F, [1.0], name="dup")
    with pytest.raises(ValueError, match="duplicate coefficient key"):
        tr.Transport([b1, b2], reference_noise=_diag_ref(psr), center=False)


def test_non_localized_slice_raises(psr, metamath_backend):
    F = np.asarray(psr.residuals)[:, None]

    def cond(params):
        return kh.jnp.array([1.0])
    cond.params = []
    bad = tr.TransportBlock("b", F, {"b_coefficients(1)": slice(3, 4)}, cond)
    with pytest.raises(ValueError, match="localized to slice"):
        tr.Transport([bad], reference_noise=_diag_ref(psr), center=False)


def test_ntoa_mismatch_raises(psr, metamath_backend):
    F1 = np.asarray(psr.residuals)[:, None]
    F2 = np.asarray(psr.residuals)[:-1, None]
    with pytest.raises(ValueError, match="disagree on n_toa"):
        tr.Transport([_const_block(F1, [1.0], name="a"),
                      _const_block(F2, [1.0], name="b")],
                     reference_noise=_diag_ref(psr), center=False)


def test_center_true_without_residual_raises(psr, metamath_backend):
    F = np.asarray(psr.residuals)[:, None]
    with pytest.raises(ValueError, match="center=True requires reference_residual"):
        tr.Transport([_const_block(F, [1.0])], reference_noise=_diag_ref(psr),
                     center=True)


def test_frozen_kernel_with_free_params_lists_missing(psr, metamath_backend):
    kernel = ds.makenoise_measurement(psr, {})       # free EFAC/EQUAD
    with pytest.raises(ValueError, match=r"missing.*efac"):
        tr.reference_noise_frozen(kernel, params0={})


def test_non_metamath_kernel_to_frozen_raises_typeerror(psr):
    ds.config(kernels="matrix")
    try:
        kernel = ds.makenoise_measurement(psr, psr.noisedict)
        with pytest.raises(TypeError, match="metamath kernel"):
            tr.reference_noise_frozen(kernel, params0=psr.noisedict)
    finally:
        ds.config(kernels="matrix")


def test_reference_noise_diagonal_quadratic_and_std(psr, metamath_backend):
    """The diagonal TOA-error reference exposes an exact diag(N0), and the
    transport's N0^-1 quadratic and sqrt(diag(N0)) helpers agree with it
    (geometry certifier support)."""
    ref = tr.reference_noise(psr)
    n0 = np.asarray(psr.toaerrs, dtype=np.float64) ** 2
    assert np.allclose(np.asarray(ref.diagonal()), n0)

    F = np.asarray(psr.residuals)[:, None]
    transport = tr.Transport(
        [_const_block(F, [1.0])], reference_noise=ref, center=False)
    n_toa = n0.shape[0]
    v = np.linspace(-1.0, 1.0, n_toa) * 1e-6
    assert np.isclose(float(transport.reference_noise_quadratic(v)),
                      float(v @ (v / n0)))
    assert np.allclose(
        np.asarray(transport.reference_noise_standard_deviation()), np.sqrt(n0))


def test_frozen_kernel_diagonal_adds_ecorr_exposure(metamath_backend):
    """diag(N0) for a Sherman-Morrison ECORR reference is diag(N) + F P F^T's
    diagonal (F is 0/1 exposure, so it adds F P epoch-wise)."""
    from discovery import metamath as mm

    N = np.array([1.0, 2.0, 3.0, 4.0])
    F = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float64)
    P = np.array([10.0, 20.0])
    kernel = mm.NoiseMatrixSM(kh.jnparray(N), F, kh.jnparray(P))
    diag = tr._frozen_kernel_diagonal(kernel, {})
    assert np.allclose(diag, N + F @ P)


def test_frozen_measurement_reference_diagonal_is_finite_positive(
        psr, metamath_backend):
    """A frozen measurement-noise reference (real EFAC/EQUAD/ECORR at the
    noisedict) yields a strictly positive finite diag(N0) of TOA length, and
    the transport's quadratic/std helpers stay finite."""
    kernel = ds.makenoise_measurement(psr, psr.noisedict)
    ref = tr.reference_noise_frozen(kernel, params0=psr.noisedict)
    diag = np.asarray(ref.diagonal())
    n_toa = np.asarray(psr.toaerrs).shape[0]
    assert diag.shape == (n_toa,)
    assert np.all(np.isfinite(diag)) and np.all(diag > 0.0)

    F = np.asarray(psr.residuals)[:, None]
    transport = tr.Transport(
        [_const_block(F, [1.0])], reference_noise=ref, center=False)
    std = np.asarray(transport.reference_noise_standard_deviation())
    assert np.allclose(std, np.sqrt(diag))
    v = np.linspace(-1.0, 1.0, n_toa) * 1e-6
    assert np.isfinite(float(transport.reference_noise_quadratic(v)))


def test_fourier_sensitivity_weights_with_frozen_reference_noise(
        psr, metamath_backend):
    """The pivot sensitivity weights run against a real frozen
    measurement-noise reference and a real Fourier basis, not a toy solver."""
    from discovery import signals as sig

    f, _df, fmat = sig.fourierbasis(psr, 5)
    ref = tr.reference_noise_frozen(
        ds.makenoise_measurement(psr, psr.noisedict), psr.noisedict)
    w = sig.fourier_sensitivity_weights(fmat, ref)
    assert w.shape == (5,)
    assert np.all(np.isfinite(w)) and np.all(w > 0.0)
    # The sensitivity-weighted pivot lands inside the sampled frequency band.
    freqs = np.asarray(f)[0::2]
    f_pivot = sig.sensitivity_weighted_pivot_frequency(freqs, w)
    assert freqs.min() <= f_pivot <= freqs.max()


def test_matrix_mode_transport_construction_raises(psr):
    ds.config(kernels="matrix")
    try:
        F = np.asarray(psr.residuals)[:, None]
        with pytest.raises(NotImplementedError, match="metamath kernel path"):
            tr.Transport([_const_block(F, [1.0])], reference_noise=_diag_ref(psr),
                         center=False)
    finally:
        ds.config(kernels="matrix")


def test_validate_negative_precision_raises_with_local_indices(psr, metamath_backend):
    F = np.asarray(psr.residuals)[:, None] * np.array([[1.0, 1.0]])
    t = tr.Transport([_const_block(F, [1.0, -2.0], name="neg")],
                     reference_noise=_diag_ref(psr), center=False)
    with pytest.raises(ValueError, match=r"negative.*\[1\]"):
        t.validate({})


def test_validate_rank_deficient_basis_with_zero_precision_fails_pd(psr, metamath_backend):
    """Two NONZERO collinear columns (F[:,1] = 2 F[:,0]) make G0 rank-1. With
    zero conditioner precision the null direction is unconstrained and A is
    singular, so the PD check fails. A strictly positive diagonal precision
    would add eps*I and correctly keep A PD -- the point of forbidding a floor:
    the pathology must surface, not be silently patched."""
    c0 = np.asarray(psr.residuals)[:, None]
    F = np.concatenate([c0, 2.0 * c0], axis=1)
    t = tr.Transport([_const_block(F, [0.0, 0.0], name="rank")],
                     reference_noise=_diag_ref(psr), center=False)
    with pytest.raises(ValueError, match="not positive definite"):
        t.validate({})

    # a strictly positive diagonal precision restores PD.
    t_pd = tr.Transport([_const_block(F, [1e-3, 1e-3], name="rank")],
                        reference_noise=_diag_ref(psr), center=False)
    t_pd.validate({})   # does not raise


def test_globalgp_curn_block_mismatched_npsr_raises(psrs, metamath_backend):
    gg = _globalgp(psrs)
    with pytest.raises(ValueError, match="npsr="):
        tr.globalgp_curn_block(gg, 0, len(psrs) + 1)


def test_globalgp_curn_block_out_of_range_slot_raises(psrs, metamath_backend):
    gg = _globalgp(psrs)
    with pytest.raises(ValueError, match="out of range"):
        tr.globalgp_curn_block(gg, len(psrs), len(psrs))


def test_array_transport_unequal_dimensions_raises(psrs, metamath_backend):
    a = _transport_for(psrs[0], psrs, 0)
    # a second transport with a DIFFERENT dimension (fewer components)
    gp_small = ds.makecommongp_fourier(psrs, ds.powerlaw, components=5,
                                       T=ds.getspan(psrs), name="rednoise")
    b = tr.Transport([tr.gp_block(gp_small, psr_slot=1)],
                     reference_noise=tr.reference_noise(psrs[1]),
                     reference_residual=np.asarray(psrs[1].residuals))
    with pytest.raises(ValueError, match="equal per-pulsar dimension"):
        tr.ArrayTransport([a, b])


def test_array_transport_mixed_centering_raises(psrs, metamath_backend):
    a = _transport_for(psrs[0], psrs, 0, center=True)
    b = _transport_for(psrs[1], psrs, 1, center=False)
    with pytest.raises(ValueError, match="all-or-none centering"):
        tr.ArrayTransport([a, b])


def test_reparam_shape_mismatch_raises(psrs, metamath_backend):
    at = tr.ArrayTransport([_transport_for(p, psrs, i)
                            for i, p in enumerate(psrs)])
    rp = at.as_reparam()
    wrong = kh.jnparray(np.zeros((at.npsr, at.dimension + 1)))
    with pytest.raises(ValueError, match="coefficient array has shape"):
        rp({}, wrong)


def test_arraylikelihood_decenter_and_transport_mutually_exclusive(psrs, metamath_backend):
    at = tr.ArrayTransport([_transport_for(p, psrs, i)
                            for i, p in enumerate(psrs)])
    with pytest.raises(ValueError, match="mutually exclusive"):
        ds.ArrayLikelihood([ds.PulsarLikelihood(
            [p.residuals, ds.makenoise_measurement(p, p.noisedict)]) for p in psrs],
            commongp=_commongp(psrs), decenter=True, transport=at)


def test_arraylikelihood_wrong_transport_pulsar_count_raises(psrs, metamath_backend):
    at = tr.ArrayTransport([_transport_for(p, psrs, i)
                            for i, p in enumerate(psrs)][:2])
    with pytest.raises(ValueError, match="pulsars"):
        ds.ArrayLikelihood([ds.PulsarLikelihood(
            [p.residuals, ds.makenoise_measurement(p, p.noisedict)]) for p in psrs],
            commongp=_commongp(psrs), transport=at)


# ==========================================================================
# 7. Reference-noise geometry, without a false density invariant
# ==========================================================================

def test_inverse_map_round_trip_and_shared_target(psr, metamath_backend):
    """Build two 2-D transports (toaerr-diagonal vs frozen noisedict). For each,
    invert q = mu + L^-T xi via xi = L^T (q - mu) and recover the same physical
    q. The target logp(q) is identical because it is the same likelihood. Do NOT
    assert equality of logp(q)+ldJ across transports: those are densities w.r.t.
    different xi coordinates and their Jacobians legitimately differ."""
    psrs = [psr]
    gp = _commongp(psrs)
    params = ds.sample_uniform(gp.Phi.getN.params)

    def build(ref):
        return tr.Transport([tr.gp_block(gp, psr_slot=0)], reference_noise=ref,
                            center=False)

    t_diag = build(tr.reference_noise(psr))
    t_frozen = build(tr.reference_noise_frozen(
        ds.makenoise_measurement(psr, psr.noisedict), params0=psr.noisedict))

    q_phys = kh.jnparray(np.random.default_rng(3).normal(size=t_diag.dimension))

    for t in (t_diag, t_frozen):
        cf, _, _b = t._factor(params)
        # xi = L^T (q - mu); here center=False so mu=0
        xi = np.asarray(cf[0]).T @ np.asarray(q_phys)
        q_back, _ = t.apply(params, kh.jnparray(xi))
        np.testing.assert_allclose(np.asarray(q_back), np.asarray(q_phys), rtol=1e-8)


# ==========================================================================
# 8. Parameter-dependent basis rejection
# ==========================================================================

def test_parameter_dependent_basis_is_rejected(psrs, metamath_backend):
    """A graph-valued basis carrying a live parameter is refused, naming it."""
    from collections import OrderedDict

    def getF(params):
        return np.ones((5, 2)) * params["x_live"]
    getF.params = ["x_live"]
    live_graph = OrderedDict([("F", metamatrix.FuncLeaf(fn=getF))])

    with pytest.raises(ValueError, match=r"depends on parameters \['x_live'\]"):
        tr._eval_basis(live_graph)


# ==========================================================================
# free-EFAC under decenter=True
# ==========================================================================

def _psl_freewn(psr):
    return ds.PulsarLikelihood([psr.residuals, ds.makenoise_measurement(psr, {})])


def test_decenter_with_free_efac_raises_naming_missing_params(psrs, metamath_backend):
    """decenter=True freezes each per-pulsar kernel at params0={}; a free EFAC
    makes that freeze incomplete, converting the old closure's silent
    constant-N assumption into a diagnosed error."""
    model = ds.ArrayLikelihood([_psl_freewn(p) for p in psrs],
                               commongp=_commongp(psrs), decenter=True)
    with pytest.raises(ValueError, match=r"missing.*efac"):
        model.clogL


def test_free_efac_via_explicit_transport_evaluates_finitely(psrs, metamath_backend):
    """The same varying-white-noise model builds and evaluates when the
    transport uses a dependency-free reference_noise(psr)."""
    per_psr = []
    for i, p in enumerate(psrs):
        per_psr.append(tr.Transport(
            [tr.gp_block(_commongp(psrs), psr_slot=i)],
            reference_noise=tr.reference_noise(p),
            reference_residual=np.asarray(p.residuals), center=True))
    at = tr.ArrayTransport(per_psr)

    model = ds.ArrayLikelihood([_psl_freewn(p) for p in psrs],
                               commongp=_commongp(psrs), transport=at)

    clogl = model.clogL
    coeff = [p for p in clogl.params if "_coefficients(" in p]
    p0 = ds.sample_uniform([p for p in clogl.params if p not in coeff])
    rng = np.random.default_rng(0)
    for key in coeff:
        p0[key] = rng.normal(size=int(key[key.index("(") + 1:key.index(")")]))

    out = clogl(p0)
    logp = float(out[0]) if isinstance(out, tuple) else float(out)
    assert np.isfinite(logp)


# ==========================================================================
# pathology matrix — legal extreme still factorizes
# ==========================================================================

def test_legal_extreme_still_factorizes(psrs, metamath_backend):
    """validate passes at a nominal draw; then a draw that drives the stochastic
    block's precision to a legal extreme (log10_A at its prior bound) still
    factorizes -- PD holds because the exact prior precision is present."""
    t = _transport_for(psrs[0], psrs, 0, center=False)

    nominal = ds.sample_uniform(t.params)
    t.validate(nominal)                              # does not raise

    extreme = dict(nominal)
    for key in t.params:
        if key.endswith("_log10_A"):
            lo, hi = ds.getprior_uniform(key)
            extreme[key] = hi                        # smallest prior variance
    diag = t.validate(extreme)                       # still PD
    assert diag["chol_diag_min"] > 0.0


# ==========================================================================
# serialization position
# ==========================================================================

def test_transport_is_not_picklable(psrs, metamath_backend):
    """Transports hold JAX arrays and closures; no pickle support is promised
    Local closures typically raise AttributeError under stdlib pickle."""
    import pickle
    t = _transport_for(psrs[0], psrs, 0)
    with pytest.raises((TypeError, AttributeError, pickle.PicklingError)):
        pickle.dumps(t)


@pytest.mark.slow
def test_checkpoint_run_with_a_transport_reparam(psrs, metamath_backend, tmp_path):
    """run_nuts_with_checkpoints completes a two-checkpoint run for a model whose
    clogL contains a transport reparam: the sampler-state pickle never touches
    model objects."""
    import jax
    import pandas as pd
    import numpyro
    from numpyro import distributions as dist
    from discovery import prior as ds_prior
    from discovery.samplers import numpyro as ds_numpyro

    model = R.decenter_intrinsic_rn(psrs)
    clogl = model.clogL                              # carries the transport reparam
    assert model.transport is None and model.decenter

    coeff = [p for p in clogl.params if "_coefficients(" in p]
    hyper = [p for p in clogl.params if p not in coeff]

    def numpyro_model():
        params = {p: numpyro.sample(p, dist.Uniform(*ds_prior.getprior_uniform(p)))
                  for p in hyper}
        # decentered coefficients are whitened -> standard-normal xi
        for key in coeff:
            width = int(key[key.index("(") + 1:key.index(")")])
            params[key] = numpyro.sample(key, dist.Normal(0.0, 1.0).expand([width]))
        out = clogl(params)
        numpyro.factor("logl", out[0] if isinstance(out, tuple) else out)
    # feather can't hold the vector coefficient columns; the point is
    # that the checkpoint run completes and pickles sampler state (not model
    # objects), so a scalar-only frame is sufficient here.
    import numpy as _np
    numpyro_model.to_df = lambda chain: pd.DataFrame(
        {k: _np.asarray(v) for k, v in chain.items() if _np.asarray(v).ndim == 1})

    sampler = ds_numpyro.makesampler_nuts(numpyro_model, num_warmup=4,
                                          num_samples=4, progress_bar=False)
    outdir = tmp_path / "chains"
    ds_numpyro.run_nuts_with_checkpoints(sampler, 2, jax.random.PRNGKey(0),
                                         outdir=outdir)

    assert len(pd.read_feather(outdir / "numpyro-samples.feather")) == 4


# ==========================================================================
# Transport.fingerprint — structural digest for run-manifest reconciliation
# (Track J: consumed by nltiming's dynamic run writer)
# ==========================================================================


def test_transport_fingerprint_is_stable_and_structural(psr, metamath_backend):
    F = np.asarray(psr.residuals)[:, None] * np.array([[1.0, 0.5]])
    ref = _diag_ref(psr)

    a = tr.Transport([_const_block(F, [1.0, 1.0], name="rn")],
                     reference_noise=ref, center=False)
    # Same structure -> identical fingerprint (independent of a params draw).
    b = tr.Transport([_const_block(F, [2.0, 3.0], name="rn")],
                     reference_noise=ref, center=False)
    assert a.fingerprint().startswith("sha256:")
    assert a.fingerprint() == b.fingerprint()

    # A different block name is a structural change -> different fingerprint.
    c = tr.Transport([_const_block(F, [1.0, 1.0], name="dm")],
                     reference_noise=ref, center=False)
    assert c.fingerprint() != a.fingerprint()

    # Different dimensionality -> different fingerprint.
    F1 = np.asarray(psr.residuals)[:, None] * np.array([[1.0]])
    d = tr.Transport([_const_block(F1, [1.0], name="rn")],
                     reference_noise=ref, center=False)
    assert d.fingerprint() != a.fingerprint()


def test_array_transport_fingerprint_is_stable(psrs, metamath_backend):
    at1 = tr.ArrayTransport([_transport_for(p, psrs, i)
                             for i, p in enumerate(psrs)])
    at2 = tr.ArrayTransport([_transport_for(p, psrs, i)
                             for i, p in enumerate(psrs)])
    assert at1.fingerprint().startswith("sha256:")
    assert at1.fingerprint() == at2.fingerprint()


# ==========================================================================
# extensions: array_block, ExtSignal centering, softclip
# ==========================================================================


class _FakeExtSignal:
    """Minimal ExtSignal duck for center_extsignals tests: .Fs, .coeffs, .name."""

    def __init__(self, Fs, kext, name="cw"):
        self.Fs, self.name = Fs, name

        def coeffs(params):
            return kh.jnp.asarray(
                [[params[f"{name}_c{j}"] for j in range(kext)] for _ in Fs])

        coeffs.params = [f"{name}_c{j}" for j in range(kext)]
        self.coeffs = coeffs


def test_array_block_precision_specs(psr, metamath_backend):
    F = np.asarray(psr.residuals)[:, None] * np.array([[1.0, 0.5, -0.3]])
    # scalar broadcast
    b = tr.array_block(F, {"tim": slice(0, 3)}, 2.0, name="timing")
    assert b.name == "timing"
    assert list(b.index) == ["tim"]
    assert np.allclose(np.asarray(b.conditioner_precision({})), 2.0)
    assert b.conditioner_precision.params == []
    # (k,) vector
    b2 = tr.array_block(F, {"tim": slice(0, 3)}, [1.0, 2.0, 3.0])
    assert np.allclose(np.asarray(b2.conditioner_precision({})), [1.0, 2.0, 3.0])
    # callable with .params

    def cp(params):
        return kh.jnp.asarray([params["s"]] * 3)

    cp.params = ["s"]
    b3 = tr.array_block(F, {"tim": slice(0, 3)}, cp)
    assert b3.conditioner_precision.params == ["s"]
    assert np.allclose(np.asarray(b3.conditioner_precision({"s": 4.0})), 4.0)


def test_array_block_validation(psr, metamath_backend):
    F = np.asarray(psr.residuals)[:, None] * np.array([[1.0, 0.5]])
    with pytest.raises(ValueError, match="one-key dict"):
        tr.array_block(F, {"a": slice(0, 2), "b": slice(0, 2)}, 1.0)
    with pytest.raises(ValueError, match=r"slice\(0, 2\)"):
        tr.array_block(F, {"a": slice(0, 1)}, 1.0)
    with pytest.raises(ValueError, match="non-finite or negative"):
        tr.array_block(F, {"a": slice(0, 2)}, [-1.0, 1.0])
    Fbad = np.asarray(psr.residuals)[:, None] * np.array([[1.0, 0.0]])
    with pytest.raises(ValueError, match="zero or non-finite"):
        tr.array_block(Fbad, {"a": slice(0, 2)}, 1.0)


def test_extsignal_subtracted_centering_matches_manual(psr, metamath_backend):
    r0 = np.asarray(psr.residuals, dtype=float)
    ntoa = r0.shape[0]
    F = r0[:, None] * np.array([[1.0, 0.5, -0.3]])
    rng = np.random.default_rng(3)
    Fext = rng.standard_normal((ntoa, 2))
    es = _FakeExtSignal([Fext], 2, name="cw")
    ref = _diag_ref(psr)

    t = tr.Transport([tr.array_block(F, {"tim": slice(0, 3)}, 1.0, name="timing")],
                     reference_noise=ref, reference_residual=r0, center=True,
                     center_extsignals=[es], psr_slot=0)
    assert set(["cw_c0", "cw_c1"]).issubset(set(t.params))

    params = {"cw_c0": 0.7, "cw_c1": -0.3}
    q, _ = t.apply(params, kh.jnp.zeros(3))

    n0 = np.asarray(psr.toaerrs) ** 2
    A = F.T @ (F / n0[:, None]) + np.eye(3)
    b0 = F.T @ (r0 / n0)
    E0 = F.T @ (Fext / n0[:, None])
    mu = np.linalg.solve(A, b0 - E0 @ np.array([0.7, -0.3]))
    assert np.allclose(np.asarray(q), mu)
    assert t.diagnostics()["center_extsignals"] == ["cw"]


def test_softclip_clamps_centering_slice(psr, metamath_backend):
    r0 = np.asarray(psr.residuals, dtype=float)
    F = r0[:, None]                                   # 1-column, near-collinear
    ref = _diag_ref(psr)
    # A pathological reference residual drives mu far outside [-4, 4].
    big = np.zeros_like(r0)
    big[0] = 1e6
    t = tr.Transport([tr.array_block(F, {"tim": slice(0, 1)}, 1e-10, name="timing")],
                     reference_noise=ref, reference_residual=big, center=True,
                     softclip={"timing": 4.0})
    q, _ = t.apply({}, kh.jnp.zeros(1))
    assert abs(float(np.asarray(q)[0])) <= 4.0 + 1e-9
    assert t.diagnostics()["softclip"] == {"timing": 4.0}


def test_softclip_and_extsignals_require_center(psr, metamath_backend):
    F = np.asarray(psr.residuals)[:, None]
    ref = _diag_ref(psr)
    with pytest.raises(ValueError, match="softclip requires center"):
        tr.Transport([tr.array_block(F, {"tim": slice(0, 1)}, 1.0, name="timing")],
                     reference_noise=ref, center=False, softclip={"timing": 4.0})
    es = _FakeExtSignal([np.asarray(psr.residuals)[:, None]], 1)
    with pytest.raises(ValueError, match="center_extsignals requires center"):
        tr.Transport([tr.array_block(F, {"tim": slice(0, 1)}, 1.0)],
                     reference_noise=ref, center=False, center_extsignals=[es])


def test_softclip_unknown_block_raises(psr, metamath_backend):
    F = np.asarray(psr.residuals)[:, None]
    ref = _diag_ref(psr)
    with pytest.raises(ValueError, match="unknown block 'nope'"):
        tr.Transport([tr.array_block(F, {"tim": slice(0, 1)}, 1.0, name="timing")],
                     reference_noise=ref, reference_residual=np.asarray(psr.residuals),
                     center=True, softclip={"nope": 4.0})


def test_array_transport_rejects_softclip(psr, metamath_backend):
    r0 = np.asarray(psr.residuals, dtype=float)
    F = r0[:, None]
    ref = _diag_ref(psr)
    t = tr.Transport(
        [tr.array_block(F, {"tim": slice(0, 1)}, 1.0, name="timing")],
        reference_noise=ref,
        reference_residual=r0,
        center=True,
        softclip={"timing": 4.0},
    )
    with pytest.raises(ValueError, match="does not support softclip"):
        tr.ArrayTransport([t])


def test_array_transport_batches_extsignal_centering(psrs, metamath_backend):
    """ArrayTransport.apply equals stacking per-pulsar Transport.apply."""
    rng = np.random.default_rng(4)
    npsr, k, k_ext = 2, 3, 2
    Fs_ext, packs = [], []
    for psr in psrs[:npsr]:
        r0 = np.asarray(psr.residuals, dtype=float)
        F = r0[:, None] * np.array([[1.0, 0.5, -0.3]])
        Fs_ext.append(rng.standard_normal((r0.shape[0], k_ext)))
        packs.append((psr, r0, F))
    es = _FakeExtSignal(Fs_ext, k_ext, name="cw")
    transports = [
        tr.Transport(
            [tr.array_block(F, {"tim": slice(0, k)}, 1.0, name="timing")],
            reference_noise=_diag_ref(psr),
            reference_residual=r0,
            center=True,
            center_extsignals=[es],
            psr_slot=i,
        )
        for i, (psr, r0, F) in enumerate(packs)
    ]
    at = tr.ArrayTransport(transports)

    assert set(es.coeffs.params) <= set(at.params)
    assert at.diagnostics()["center_extsignals"] == ["cw"]

    params = {"cw_c0": 0.7, "cw_c1": -0.3}
    xi = kh.jnp.zeros((npsr, k))
    q_batch, ldJ_batch = at.apply(params, xi)
    q_loop, ldJ_loop = [], 0.0
    for i, t in enumerate(transports):
        qi, ldi = t.apply(params, xi[i])
        q_loop.append(np.asarray(qi))
        ldJ_loop = ldJ_loop + ldi
    np.testing.assert_allclose(np.asarray(q_batch), np.stack(q_loop), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(ldJ_batch), np.asarray(ldJ_loop), rtol=1e-12, atol=0.0)

    # z=0 mean depends on the ExtSignal (loud deterministic shift).
    q0, _ = at.apply({"cw_c0": 0.0, "cw_c1": 0.0}, xi)
    assert np.linalg.norm(np.asarray(q_batch) - np.asarray(q0)) > 1e-6

    # ldJ is independent of ExtSignal parameters.
    _, ldJ_zero = at.apply({"cw_c0": 0.0, "cw_c1": 0.0}, xi)
    np.testing.assert_allclose(np.asarray(ldJ_batch), np.asarray(ldJ_zero), rtol=1e-12, atol=0.0)


def test_array_transport_extsignals_all_or_none(psrs, metamath_backend):
    psr0, psr1 = psrs[0], psrs[1]
    r0 = np.asarray(psr0.residuals, dtype=float)
    r1 = np.asarray(psr1.residuals, dtype=float)
    F0 = r0[:, None]
    F1 = r1[:, None]
    es = _FakeExtSignal([F0], 1, name="cw")
    t0 = tr.Transport(
        [tr.array_block(F0, {"tim": slice(0, 1)}, 1.0, name="timing")],
        reference_noise=_diag_ref(psr0),
        reference_residual=r0,
        center=True,
        center_extsignals=[es],
        psr_slot=0,
    )
    t1 = tr.Transport(
        [tr.array_block(F1, {"tim": slice(0, 1)}, 1.0, name="timing")],
        reference_noise=_diag_ref(psr1),
        reference_residual=r1,
        center=True,
    )
    with pytest.raises(ValueError, match="all-or-none ExtSignal centering"):
        tr.ArrayTransport([t0, t1])


def test_array_transport_extsignals_require_shared_coeffs(psrs, metamath_backend):
    psr0, psr1 = psrs[0], psrs[1]
    r0 = np.asarray(psr0.residuals, dtype=float)
    r1 = np.asarray(psr1.residuals, dtype=float)
    F0 = r0[:, None]
    F1 = r1[:, None]
    es0 = _FakeExtSignal([F0, F1], 1, name="cw")
    es1 = _FakeExtSignal([F0, F1], 1, name="cw")  # different coeffs identity
    t0 = tr.Transport(
        [tr.array_block(F0, {"tim": slice(0, 1)}, 1.0, name="timing")],
        reference_noise=_diag_ref(psr0),
        reference_residual=r0,
        center=True,
        center_extsignals=[es0],
        psr_slot=0,
    )
    t1 = tr.Transport(
        [tr.array_block(F1, {"tim": slice(0, 1)}, 1.0, name="timing")],
        reference_noise=_diag_ref(psr1),
        reference_residual=r1,
        center=True,
        center_extsignals=[es1],
        psr_slot=1,
    )
    with pytest.raises(ValueError, match="same coeffs callable"):
        tr.ArrayTransport([t0, t1])


def _two_psr_timing_transports(psrs, es, *, slots=None, extra_es=None):
    """Two 1-column timing transports sharing `es` (and optional `extra_es`)."""
    slots = [0, 1] if slots is None else slots
    out = []
    for i, sl in enumerate(slots):
        psr = psrs[i]
        r0 = np.asarray(psr.residuals, dtype=float)
        F = r0[:, None]
        kwargs = dict(center=True, center_extsignals=[es] + (extra_es or []), psr_slot=sl)
        out.append(
            tr.Transport(
                [tr.array_block(F, {"tim": slice(0, 1)}, 1.0, name="timing")],
                reference_noise=_diag_ref(psr),
                reference_residual=r0,
                **kwargs,
            )
        )
    return out


def test_array_transport_extsignals_name_mismatch(psrs, metamath_backend):
    F0 = np.asarray(psrs[0].residuals)[:, None]
    F1 = np.asarray(psrs[1].residuals)[:, None]
    es_a = _FakeExtSignal([F0, F1], 1, name="cw")
    es_b = _FakeExtSignal([F0, F1], 1, name="other")
    # Force the same coeffs identity so the name check is what fires.
    es_b.coeffs = es_a.coeffs
    t0, t1 = _two_psr_timing_transports(psrs, es_a)
    t1 = tr.Transport(
        [tr.array_block(F1, {"tim": slice(0, 1)}, 1.0, name="timing")],
        reference_noise=_diag_ref(psrs[1]),
        reference_residual=np.asarray(psrs[1].residuals, dtype=float),
        center=True,
        center_extsignals=[es_b],
        psr_slot=1,
    )
    with pytest.raises(ValueError, match="names disagree"):
        tr.ArrayTransport([t0, t1])


def test_array_transport_extsignals_count_mismatch(psrs, metamath_backend):
    F0 = np.asarray(psrs[0].residuals)[:, None]
    F1 = np.asarray(psrs[1].residuals)[:, None]
    es1 = _FakeExtSignal([F0, F1], 1, name="cw")
    es2 = _FakeExtSignal([F0, F1], 1, name="cw2")
    t0 = _two_psr_timing_transports(psrs, es1, extra_es=[es2])[0]
    t1 = _two_psr_timing_transports(psrs, es1)[1]
    with pytest.raises(ValueError, match="number of ExtSignals"):
        tr.ArrayTransport([t0, t1])


def test_array_transport_extsignals_bad_slots(psrs, metamath_backend):
    # Each Transport reads Fs[psr_slot], so the stored slot's basis must
    # match that pulsar's n_toa. Both claim slot 0 so the order check fires.
    r0 = np.asarray(psrs[0].residuals, dtype=float)
    r1 = np.asarray(psrs[1].residuals, dtype=float)
    es0 = _FakeExtSignal([r0[:, None], r1[:, None]], 1, name="cw")
    es1 = _FakeExtSignal([r1[:, None], r0[:, None]], 1, name="cw")
    es1.coeffs = es0.coeffs
    t0 = tr.Transport(
        [tr.array_block(r0[:, None], {"tim": slice(0, 1)}, 1.0, name="timing")],
        reference_noise=_diag_ref(psrs[0]),
        reference_residual=r0,
        center=True,
        center_extsignals=[es0],
        psr_slot=0,
    )
    t1 = tr.Transport(
        [tr.array_block(r1[:, None], {"tim": slice(0, 1)}, 1.0, name="timing")],
        reference_noise=_diag_ref(psrs[1]),
        reference_residual=r1,
        center=True,
        center_extsignals=[es1],
        psr_slot=0,
    )
    with pytest.raises(ValueError, match="psr_slot=i"):
        tr.ArrayTransport([t0, t1])


def test_array_transport_validate_rejects_bad_coeff_shape(psrs, metamath_backend):
    F0 = np.asarray(psrs[0].residuals)[:, None]
    F1 = np.asarray(psrs[1].residuals)[:, None]
    es = _FakeExtSignal([F0, F1], 2, name="cw")

    def bad(params):
        return kh.jnp.asarray([params["cw_c0"], params["cw_c1"]])  # (2,) not (2, 2)

    bad.params = es.coeffs.params
    es.coeffs = bad
    at = tr.ArrayTransport(_two_psr_timing_transports(psrs, es))
    with pytest.raises(ValueError, match="coeffs\\(params\\) has shape"):
        at.validate({"cw_c0": 0.1, "cw_c1": -0.2})


def test_transport_rejects_1d_extsignal_basis(psr, metamath_backend):
    r0 = np.asarray(psr.residuals, dtype=float)
    F = r0[:, None]
    es = _FakeExtSignal([r0], 1, name="cw")  # Fs[0] is 1-D
    with pytest.raises(ValueError, match="must be 2-D"):
        tr.Transport(
            [tr.array_block(F, {"tim": slice(0, 1)}, 1.0, name="timing")],
            reference_noise=_diag_ref(psr),
            reference_residual=r0,
            center=True,
            center_extsignals=[es],
            psr_slot=0,
        )


def test_array_transport_sums_two_extsignals(psrs, metamath_backend):
    rng = np.random.default_rng(5)
    npsr, k_ext = 2, 2
    Fs_a, Fs_b, trans = [], [], []
    for i, psr in enumerate(psrs[:npsr]):
        r0 = np.asarray(psr.residuals, dtype=float)
        F = r0[:, None] * np.array([[1.0, 0.5, -0.3]])
        Fa = rng.standard_normal((r0.shape[0], k_ext))
        Fb = rng.standard_normal((r0.shape[0], k_ext))
        Fs_a.append(Fa)
        Fs_b.append(Fb)
        trans.append((psr, r0, F))
    esa = _FakeExtSignal(Fs_a, k_ext, name="cwa")
    esb = _FakeExtSignal(Fs_b, k_ext, name="cwb")
    transports = [
        tr.Transport(
            [tr.array_block(F, {"tim": slice(0, 3)}, 1.0, name="timing")],
            reference_noise=_diag_ref(psr),
            reference_residual=r0,
            center=True,
            center_extsignals=[esa, esb],
            psr_slot=i,
        )
        for i, (psr, r0, F) in enumerate(trans)
    ]
    at = tr.ArrayTransport(transports)
    params = {"cwa_c0": 0.4, "cwa_c1": -0.1, "cwb_c0": 0.2, "cwb_c1": 0.3}
    xi = kh.jnp.zeros((npsr, 3))
    q_batch, _ = at.apply(params, xi)
    q_loop = np.stack([np.asarray(t.apply(params, xi[i])[0]) for i, t in enumerate(transports)])
    np.testing.assert_allclose(np.asarray(q_batch), q_loop, rtol=1e-10, atol=1e-12)


def test_decenter_sugar_centers_extsignals(psrs, metamath_backend):
    model = R.decenter_extsignal_cw(psrs)
    ys = model._coefficient_assembly[1]
    at = model._build_decenter_transport(ys)
    cw_names = set(model.extsignals[0].params)  # includes {psr}_cw_phi_psr
    assert cw_names
    assert cw_names <= set(at.params)
    assert cw_names <= set(at.as_reparam().params)
    assert at.diagnostics()["center_extsignals"] == ["cw"]


def test_decenter_sugar_without_extsignals_unchanged_params(psrs, metamath_backend):
    """No ExtSignal names appear on a GP-only decenter transport."""
    model = R.decenter_intrinsic_rn(psrs)
    ys = model._coefficient_assembly[1]
    at = model._build_decenter_transport(ys)
    assert "center_extsignals" not in at.diagnostics()
    cw_names = set(R.decenter_extsignal_cw(psrs).extsignals[0].params)
    assert cw_names.isdisjoint(at.params)


def test_explicit_transport_does_not_absorb_likelihood_extsignals(psrs, metamath_backend):
    """Caller-owned transport= is not rewritten from al.extsignals."""
    T = ds.getspan(psrs)
    commongp = ds.makecommongp_fourier(psrs, ds.powerlaw, components=10, T=T, name="rednoise")
    # Same frozen-WN skeleton as recipes._psl_skeleton (required so
    # reference_noise_frozen(..., params0={}) succeeds).
    psls = [
        ds.PulsarLikelihood(
            [
                p.residuals,
                ds.makenoise_measurement(p, p.noisedict),
                ds.makegp_ecorr(p, p.noisedict),
                ds.makegp_timing(p, svd=True),
            ]
        )
        for p in psrs
    ]
    cw = ds.makecw_extsignal(psrs, components=8, T=T, pulsarterm=True, name="cw")
    per = []
    for i, psl in enumerate(psls):
        per.append(
            tr.Transport(
                [tr.gp_block(commongp, psr_slot=i)],
                reference_noise=tr.reference_noise_frozen(psl.N, params0={}, description=f"n{i}"),
                reference_residual=np.asarray(psl.y),
                center=True,
            )
        )
        # no center_extsignals
    at = tr.ArrayTransport(per)
    al = ds.ArrayLikelihood(psls, commongp=commongp, transport=at, extsignals=[cw])
    assert "center_extsignals" not in al.transport.diagnostics()
    assert set(cw.params).isdisjoint(al.transport.params)


def test_apply_grad_log10_h0_is_the_centering_path(psrs, metamath_backend):
    """d(apply)/d(cw_log10_h0) is finite, matches FD, and is absent without centering."""
    model = R.decenter_extsignal_cw(psrs)
    ys = model._coefficient_assembly[1]
    at = model._build_decenter_transport(ys)
    name = "cw_log10_h0"
    assert name in at.params
    np.random.seed(0)
    p0 = ds.sample_uniform(list(at.params))
    p0[name] = -6.0  # loud enough that dμ/d log10_h0 is not ~0
    xi = kh.jnp.zeros((at.npsr, at.dimension))

    def mean_norm(x):
        q = dict(p0)
        q[name] = x
        a, _ = at.apply(q, xi)
        return kh.jnp.sum(a * a)

    x0 = float(p0[name])
    g = float(jax.grad(mean_norm)(x0))
    assert np.isfinite(g)
    assert abs(g) > 1e-8
    eps = 1e-5
    fd = float((mean_norm(x0 + eps) - mean_norm(x0 - eps)) / (2 * eps))
    np.testing.assert_allclose(g, fd, rtol=2e-3, atol=1e-3)

    # Same model, caller-owned transport without center_extsignals: zero.
    psls, commongp = model.psls, model.commongp
    per = []
    for i, psl in enumerate(psls):
        blocks = [tr.gp_block(commongp, psr_slot=i)]
        per.append(
            tr.Transport(
                blocks,
                reference_noise=tr.reference_noise_frozen(psl.N, params0={}, description=f"n{i}"),
                reference_residual=np.asarray(ys[i]),
                center=True,
            )
        )
    at_bare = tr.ArrayTransport(per)
    assert name not in at_bare.params

    def mean_norm_bare(x):
        q = dict(p0)
        q[name] = x
        a, _ = at_bare.apply(q, xi)
        return kh.jnp.sum(a * a)

    assert abs(float(jax.grad(mean_norm_bare)(x0))) < 1e-12


def test_array_transport_fingerprint_changes_with_extsignals(psrs, metamath_backend):
    model = R.decenter_extsignal_cw(psrs)
    ys = model._coefficient_assembly[1]
    at_cw = model._build_decenter_transport(ys)
    at_plain = R.decenter_intrinsic_rn(psrs)._build_decenter_transport(
        R.decenter_intrinsic_rn(psrs)._coefficient_assembly[1]
    )
    assert at_cw.fingerprint().startswith("sha256:")
    assert at_cw.fingerprint() != at_plain.fingerprint()
    at_cw2 = R.decenter_extsignal_cw(psrs)._build_decenter_transport(
        R.decenter_extsignal_cw(psrs)._coefficient_assembly[1]
    )
    assert at_cw.fingerprint() == at_cw2.fingerprint()


def test_decenter_extsignal_cross_matches_residual(psrs, metamath_backend):
    """clogl_form residual vs cross agree when ExtSignal centering is on."""
    T = ds.getspan(psrs)
    commongp = ds.makecommongp_fourier(psrs, ds.powerlaw, components=10, T=T, name="rednoise")
    psls = [
        ds.PulsarLikelihood(
            [
                p.residuals,
                ds.makenoise_measurement(p, p.noisedict),
                ds.makegp_ecorr(p, p.noisedict),
                ds.makegp_timing(p, svd=True),
            ]
        )
        for p in psrs
    ]
    cw = ds.makecw_extsignal(psrs, components=8, T=T, pulsarterm=True, name="cw")
    kwargs = dict(psls=psls, commongp=commongp, decenter=True, extsignals=[cw])
    cross = ds.ArrayLikelihood(**kwargs, clogl_form="cross")
    resid = ds.ArrayLikelihood(**kwargs, clogl_form="residual")
    rng = np.random.default_rng(1)
    p0 = ds.sample_uniform([p for p in cross.clogL.params if not p.endswith(")")])
    for p in cross.clogL.params:
        if p.endswith(")"):
            n = int(p[p.index("(") + 1 : -1])
            p0[p] = 1e-6 * rng.standard_normal(n)
    lo, ln = cross.clogL(p0), resid.clogL(p0)
    if isinstance(lo, tuple):
        np.testing.assert_allclose(float(lo[0]), float(ln[0]), rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(np.asarray(lo[1]), np.asarray(ln[1]), rtol=1e-9, atol=1e-8)
    else:
        np.testing.assert_allclose(float(lo), float(ln), rtol=1e-9, atol=1e-8)


# ==========================================================================
# MarginalTransport
# ==========================================================================

from discovery import metamath  # noqa: E402


class _MockPsr:
    """Minimal pulsar for the GP signal factories (name / pos / toas)."""

    name = "J0000+0000"

    def __init__(self, n, span_days=3000.0):
        self.toas = 53000.0 * 86400.0 + np.linspace(0.0, span_days * 86400.0, n)
        self.pos = np.array([1.0, 0.0, 0.0])


def _build_toy_marginal(rng, n=40, comps=3):
    """Fold WN + powerlaw RN + improper timing (1e40) + unit-normal z-prior into
    one metamath WoodburyKernel stack (exactly the shape of a marginalized
    ``PulsarLikelihood.N``), plus the pieces for an independent NumPy oracle."""
    psr = _MockPsr(n)
    n0 = rng.uniform(0.5, 1.5, n)
    gp_rn = ds.makegp_fourier(psr, ds.powerlaw, comps, name="rednoise")
    gp_imp = ds.makegp_improper(psr, rng.standard_normal((n, 2)), name="tm")
    gp_sn = ds.makegp_standard_normal(psr, rng.standard_normal((n, 2)), name="zprior")

    K = metamath.NoiseMatrix(kh.jnparray(n0))
    Fs, Phis = [], []
    for gp in (gp_rn, gp_imp, gp_sn):
        F = np.asarray(gp.F, dtype=np.float64)
        K = metamath.WoodburyKernel(K, kh.jnparray(F), gp.Phi)
        Fs.append(F)
        Phis.append(gp.Phi)
    F_all = np.concatenate(Fs, axis=1)

    def phi_of(eta):
        out = []
        for Ph in Phis:
            c, f = metamath._materialize(Ph.N)
            out.append(np.asarray(c if f is None else f(params=eta), dtype=np.float64))
        return np.concatenate(out)

    eta_names = list(gp_rn.Phi.N.params)  # powerlaw log10_A / gamma
    return dict(n0=n0, K=K, F_all=F_all, phi_of=phi_of, eta_names=eta_names)


def _eta(names, rng):
    out = {}
    for nm in names:
        out[nm] = float(rng.uniform(-1.0, 0.0)) if "log10_A" in nm else float(rng.uniform(1.5, 5.0))
    return out


def _oracle_products(n0, F_all, phi, W, y):
    """Independent NumPy Woodbury: (W^T C^-1 W, W^T C^-1 y) for
    C = diag(n0) + F diag(phi) F^T. Stable with a 1e40 improper entry
    (Phi^-1 -> 1e-40), unlike a dense inverse of C."""
    Ninv = 1.0 / n0
    NiF = Ninv[:, None] * F_all
    NiW = Ninv[:, None] * W
    Niy = Ninv * y
    inner = np.diag(1.0 / phi) + F_all.T @ NiF
    cfa = np.linalg.cholesky(inner)

    def isolve(rhs):
        return np.linalg.solve(cfa.T, np.linalg.solve(cfa, rhs))

    WtNiF = W.T @ NiF
    WtCiW = W.T @ NiW - WtNiF @ isolve(F_all.T @ NiW)
    WtCiy = W.T @ Niy - WtNiF @ isolve(F_all.T @ Niy)
    return WtCiW, WtCiy


def _oracle_quadratic(n0, F_all, phi, v):
    Ninv = 1.0 / n0
    Niv = Ninv * v
    NiF = Ninv[:, None] * F_all
    inner = np.diag(1.0 / phi) + F_all.T @ NiF
    cfa = np.linalg.cholesky(inner)
    FtNiv = F_all.T @ Niv
    return float(v @ Niv - FtNiv @ np.linalg.solve(cfa.T, np.linalg.solve(cfa, FtNiv)))


def _make_block(W):
    key = f"{_MockPsr.name}_timing_timing_z"
    return tr.array_block(W, {key: slice(0, W.shape[1])},
                          conditioner_precision=1.0, name="timing"), key


def test_marginal_transport_matches_dense_oracle(metamath_backend):
    """T-D1: A(eta) and mu(eta) from MarginalTransport._factor match the
    independent Woodbury oracle W^T C^-1 W + I / A^-1 W^T C^-1 y at 5 eta draws."""
    rng = np.random.default_rng(0)
    toy = _build_toy_marginal(rng)
    n = toy["n0"].shape[0]
    W = rng.standard_normal((n, 3))
    y = rng.standard_normal(n)
    block, _ = _make_block(W)
    t = tr.marginal_transport(toy["K"], y, block, center=True)

    for _ in range(5):
        eta = _eta(toy["eta_names"], rng)
        phi = toy["phi_of"](eta)
        WtCiW, WtCiy = _oracle_products(toy["n0"], toy["F_all"], phi, W, y)
        A_o = WtCiW + np.eye(3)
        mu_o = np.linalg.solve(A_o, WtCiy)

        cf, b, pinv = t._factor(eta)
        A_t = np.asarray(cf[0]) @ np.asarray(cf[0]).T  # L L^T (lower factor)
        mu_t = np.asarray(t.apply(eta, np.zeros(3))[0])
        assert np.allclose(A_t, A_o, rtol=1e-8, atol=1e-10)
        assert np.allclose(np.asarray(b), WtCiy, rtol=1e-8, atol=1e-10)
        assert np.allclose(mu_t, mu_o, rtol=1e-8, atol=1e-10)


def test_marginal_transport_jacobian_matches_ldj(metamath_backend):
    """T-D2: jacfwd slogdet of z(xi) equals ldJ, center on and off."""
    import jax.numpy as jnp
    rng = np.random.default_rng(1)
    toy = _build_toy_marginal(rng)
    n = toy["n0"].shape[0]
    W = rng.standard_normal((n, 3))
    block, _ = _make_block(W)
    eta = _eta(toy["eta_names"], rng)
    for center in (True, False):
        t = tr.marginal_transport(toy["K"], rng.standard_normal(n), block, center=center)
        jac = np.asarray(jax.jacfwd(lambda xi: t.apply(eta, xi)[0])(jnp.zeros(3)))
        _, sld = np.linalg.slogdet(jac)
        ldJ = float(t.apply(eta, jnp.zeros(3))[1])
        assert np.isclose(sld, ldJ, rtol=1e-10, atol=1e-10)


def test_marginal_transport_params_propagation(metamath_backend):
    """T-D3: t.params == union of kernel hyper names and block conditioner params."""
    rng = np.random.default_rng(2)
    toy = _build_toy_marginal(rng)
    n = toy["n0"].shape[0]
    block, _ = _make_block(rng.standard_normal((n, 3)))
    t = tr.marginal_transport(toy["K"], rng.standard_normal(n), block)
    assert t.params == sorted(set(toy["eta_names"]) | set(block.conditioner_precision.params))
    assert set(toy["eta_names"]).issubset(set(t.params))


def test_marginal_transport_centering_is_gls_solution(metamath_backend):
    """T-D4: mu(eta) equals the dense GLS solution A^-1 W^T C^-1 y (pins b sign
    and the y_t waveform convention)."""
    rng = np.random.default_rng(3)
    toy = _build_toy_marginal(rng)
    n = toy["n0"].shape[0]
    W = rng.standard_normal((n, 3))
    y = rng.standard_normal(n)
    block, _ = _make_block(W)
    t = tr.marginal_transport(toy["K"], y, block, center=True)
    eta = _eta(toy["eta_names"], rng)
    WtCiW, WtCiy = _oracle_products(toy["n0"], toy["F_all"], toy["phi_of"](eta), W, y)
    gls = np.linalg.solve(WtCiW + np.eye(3), WtCiy)
    mu = np.asarray(t.apply(eta, np.zeros(3))[0])
    assert np.allclose(mu, gls, rtol=1e-8, atol=1e-10)


def test_marginal_transport_failure_semantics(metamath_backend):
    """T-D5: callable y, kernel without make_kernelsolve, multi-key block,
    matrix mode, and negative conditioner precision each fail as specified."""
    rng = np.random.default_rng(4)
    toy = _build_toy_marginal(rng)
    n = toy["n0"].shape[0]
    block, _ = _make_block(rng.standard_normal((n, 3)))
    y = rng.standard_normal(n)

    with pytest.raises(TypeError, match="CompoundDelay|D-INV"):
        tr.marginal_transport(toy["K"], (lambda p: y), block)
    with pytest.raises(TypeError, match="make_kernelsolve"):
        tr.marginal_transport(object(), y, block)
    # multi-key index rejected.
    W = rng.standard_normal((n, 3))
    bad = tr.TransportBlock("timing", W,
                            {"a": slice(0, 2), "b": slice(2, 3)},
                            block.conditioner_precision)
    with pytest.raises(ValueError, match="exactly one coefficient key"):
        tr.marginal_transport(toy["K"], y, bad)
    # negative conditioner precision -> validate error (no floor). A constant
    # negative spec is rejected by array_block eagerly, so use a live callable.
    def negcp(params):
        return kh.jnp.asarray([1.0, -1.0, 1.0])
    negcp.params = []
    negblock = tr.TransportBlock("timing", W, {"k": slice(0, 3)}, negcp)
    t = tr.marginal_transport(toy["K"], y, negblock)
    with pytest.raises(ValueError, match="negative|no floor"):
        t.validate(_eta(toy["eta_names"], rng))


def test_marginal_transport_matrix_mode_raises():
    """T-D5 (matrix mode): require_metamath -> NotImplementedError."""
    ds.config(kernels="matrix")
    try:
        rng = np.random.default_rng(5)
        W = rng.standard_normal((10, 2))
        block, _ = _make_block(W)
        with pytest.raises(NotImplementedError, match="metamath"):
            tr.marginal_transport(object(), rng.standard_normal(10), block)
    finally:
        ds.config(kernels="matrix")


def test_marginal_transport_live_kernel_hooks(metamath_backend):
    """T-D6: live_kernel_quadratic and live_kernel_standard_deviation match the
    dense v^T C^-1 v and sqrt(diag C) at two eta points; bad inputs raise."""
    rng = np.random.default_rng(6)
    toy = _build_toy_marginal(rng)
    n = toy["n0"].shape[0]
    W = rng.standard_normal((n, 3))
    block, _ = _make_block(W)
    t = tr.marginal_transport(toy["K"], rng.standard_normal(n), block)

    for _ in range(2):
        eta = _eta(toy["eta_names"], rng)
        phi = toy["phi_of"](eta)
        v = rng.standard_normal(n)
        assert np.isclose(t.live_kernel_quadratic(eta, v),
                          _oracle_quadratic(toy["n0"], toy["F_all"], phi, v),
                          rtol=1e-8, atol=1e-10)
        diag_C = toy["n0"] + (toy["F_all"] ** 2) @ phi
        assert np.allclose(t.live_kernel_standard_deviation(eta),
                           np.sqrt(diag_C), rtol=1e-8, atol=0.0)

    eta = _eta(toy["eta_names"], rng)
    with pytest.raises(ValueError, match="shape"):
        t.live_kernel_quadratic(eta, np.ones(n + 1))


def test_live_kernel_diagonal_type_dispatch(metamath_backend):
    """Type dispatch: WoodburyProjKernel -> NotImplementedError,
    an unsupported kernel type -> TypeError."""
    from discovery.transport import _live_kernel_diagonal

    rng = np.random.default_rng(7)
    n = 20
    psr = _MockPsr(n)
    gp_rn = ds.makegp_fourier(psr, ds.powerlaw, 3, name="rednoise")
    eta = {p: (-0.5 if "log10_A" in p else 3.0) for p in gp_rn.Phi.N.params}

    proj = metamath.WoodburyProjKernel(
        metamath.NoiseMatrix(kh.jnparray(rng.uniform(0.5, 1.5, n))),
        rng.standard_normal((n, 2)),   # M (timing, projected out)
        np.asarray(gp_rn.F),           # F (GP basis, kept)
        gp_rn.Phi,                     # P
    )
    with pytest.raises(NotImplementedError, match="WoodburyProjKernel|projection"):
        _live_kernel_diagonal(proj, eta)
    with pytest.raises(TypeError, match="unsupported kernel type"):
        _live_kernel_diagonal(object(), eta)


# ==========================================================================
# Batched conditioner
# ==========================================================================

def test_batched_conditioner_equals_stacked(psrs, metamath_backend):
    model = R.decenter_intrinsic_rn_global_hd(psrs)
    transport = model._build_decenter_transport(model._coefficient_assembly[1])
    stacked = tr._stacked_array_conditioner(transport.transports)
    rng = np.random.default_rng(11)
    for _ in range(20):
        params = ds.sample_uniform(transport.params)
        np.testing.assert_allclose(
            transport._pinv(params), stacked(params), rtol=1e-12)


def test_separable_curn_precision_equals_dense_diagonal(psrs, metamath_backend):
    gp = _globalgp(psrs, components=5)
    params = ds.sample_uniform(gp.Phi.getN.params)
    batched = tr.globalgp_curn_array_conditioner(gp, len(psrs))(params)
    dense = 1.0 / np.diag(np.asarray(gp.Phi.getN(params))).reshape((len(psrs), -1))
    np.testing.assert_allclose(batched, dense, rtol=1e-12)


def test_legacy_curn_reciprocal_matches_diagonal_first(psrs, metamath_backend):
    gp = _globalgp(psrs, components=5)
    params = ds.sample_uniform(gp.Phi.getN.params)
    new = tr.globalgp_curn_block(gp, 1, len(psrs)).conditioner_precision(params)
    old = tr._legacy_globalgp_curn_precision(gp.Phi.getN, 1, len(psrs))(params)
    np.testing.assert_allclose(new, old, rtol=1e-12)
    assert np.all(np.isfinite(new))


def test_malformed_batched_conditioner_shape_raises(psrs, metamath_backend):
    model = R.decenter_intrinsic_rn(psrs)
    transport = model._build_decenter_transport(model._coefficient_assembly[1])

    def bad(params):
        return kh.jnp.ones((len(psrs), transport.dimension + 1))
    bad.params = []

    broken = tr.ArrayTransport(
        transport.transports, conditioner_precision=bad)
    with pytest.raises(ValueError, match="batched conditioner precision"):
        broken._pinv(ds.sample_uniform(transport.params))


def test_user_arraytransport_without_explicit_callable_still_works(
        psrs, metamath_backend):
    model = R.decenter_intrinsic_rn(psrs)
    per_psr = model._build_decenter_transport(
        model._coefficient_assembly[1]).transports
    transport = tr.ArrayTransport(per_psr)
    assert transport._conditioner_precision is not None
    params = ds.sample_uniform(transport.params)
    value = transport._pinv(params)
    assert value.shape == (transport.npsr, transport.dimension)
    transport.validate(params)


def test_arraytransport_params_include_batched_conditioner(psrs, metamath_backend):
    model = R.decenter_intrinsic_rn_global_hd(psrs)
    transport = model._build_decenter_transport(model._coefficient_assembly[1])
    for name in transport._conditioner_precision.params:
        assert name in transport.params


# ==========================================================================
# Bake precision is independent of the sampling (working) dtype
# ==========================================================================

def _with_float32_working(build):
    ds.utils.config(backend="jax", factor="cholesky", working=jax.numpy.float32)
    try:
        return build()
    finally:
        ds.utils.config(backend="jax", factor="cholesky")


def test_func_working_override_beats_the_float32_config(psrs, cholesky_backend):
    """`metamatrix.func(..., working=float64)` materializes in float64 even
    under `config(working=float32)`; without the override the config wins."""
    model = R.decenter_intrinsic_rn(psrs)
    N = model.psls[0].N
    rhs = np.asarray(model._coefficient_assembly[1][0], dtype=np.float64)

    def build():
        return (metamatrix.func(N.make_solve)(rhs, params={})[0],
                metamatrix.func(N.make_solve,
                                working=jax.numpy.float64)(rhs, params={})[0])
    default, forced = _with_float32_working(build)
    assert default.dtype == jax.numpy.float32
    assert forced.dtype == jax.numpy.float64


def test_bake_is_float64_under_float32_working(psrs, cholesky_backend):
    """G0/b0 and the conditioner precision are baked in float64 regardless of
    the working dtype, and the float32-config bake matches the float64 one.

    Before this guarantee a float32 solve through the timing-model Woodbury
    left G0 indefinite (lambda_min ~ -1e-5 lambda_max), so G0 + diag(p) went
    indefinite for legal hyperparameters and NUTS saw NaN log-densities."""
    m64 = R.decenter_intrinsic_rn_global_hd(psrs)
    t64 = m64._build_decenter_transport(m64._coefficient_assembly[1])

    def build():
        m32 = R.decenter_intrinsic_rn_global_hd(psrs)
        return m32._build_decenter_transport(m32._coefficient_assembly[1])
    t32 = _with_float32_working(build)

    assert t32._G0.dtype == jax.numpy.float64
    assert t32._b0.dtype == jax.numpy.float64
    G64, G32 = np.asarray(t64._G0), np.asarray(t32._G0)
    scale = np.max(np.abs(G64))
    assert np.max(np.abs(G32 - G64)) <= 1e-10 * scale
    for G in G32:
        lam = np.linalg.eigvalsh(G)
        assert lam[0] >= -tr._G0_PSD_RTOL * lam[-1]

    # conditioner precision: float64 and twice differentiable at the clipped
    # low-amplitude edge (phi = 1e-18 s^2, where a float32 phi**-3 overflows)
    hyper = ds.sample_uniform(t32.params)
    name = next(p for p in t32.params if p.endswith("log10_A"))

    def precision_sum(x):
        return jax.numpy.sum(t32._pinv({**hyper, name: x}))
    pinv = t32._pinv(hyper)
    assert pinv.dtype == jax.numpy.float64
    for x in (-20.0, -14.0):
        h = jax.hessian(precision_sum)(jax.numpy.asarray(x))
        assert np.isfinite(float(h))


def test_indefinite_reference_gram_raises_at_construction(psrs, metamath_backend):
    """A reference-noise solve too inaccurate to bake from is diagnosed at
    construction instead of surfacing as NaN factorizations at runtime."""
    model = R.decenter_intrinsic_rn(psrs)
    block = tr.gp_block(model.commongp, psr_slot=0)

    class Flipped:
        description = "sign-flipped solve"

        def solve(self, rhs):
            return -np.asarray(rhs), 0.0

    with pytest.raises(ValueError, match="indefinite"):
        tr.Transport([block], reference_noise=Flipped(), center=False)


# ==========================================================================
# white-noise tracking (class_tracking): the transport seam
# ==========================================================================

def _wn_kernel(psr):
    return ds.makenoise_measurement(psr, {})


def _wn_params0(psr):
    return {k: v for k, v in psr.noisedict.items()
            if k.endswith(("_efac", "_log10_t2equad"))}


def _blocks_for(psrs, i):
    return [tr.gp_block(_commongp(psrs), psr_slot=i)]


def _tracked_and_frozen(psr, psrs, i, **kw):
    kern = _wn_kernel(psr)
    p0 = _wn_params0(psr)
    r0 = np.asarray(psr.residuals)
    tracked = tr.Transport(
        _blocks_for(psrs, i),
        reference_noise=tr.class_tracking(kern, p0, toaerrs=psr.toaerrs, **kw),
        reference_residual=r0, center=True)
    frozen = tr.Transport(
        _blocks_for(psrs, i),
        reference_noise=tr.reference_noise_frozen(kern, params0=p0),
        reference_residual=r0, center=True)
    return tracked, frozen, kern, p0


def _hyper_params(transport, psr, seed=0):
    """A params dict for every transport parameter, white noise at the bake point."""
    p = ds.sample_uniform(transport.params)
    p.update(_wn_params0(psr))
    return p


def test_tracked_transport_reproduces_frozen_at_bake_point(psrs, metamath_backend):
    psr = psrs[0]
    tracked, frozen, _kern, _p0 = _tracked_and_frozen(psr, psrs, 0)
    params = _hyper_params(tracked, psr)
    xi = kh.jnparray(np.random.default_rng(3).standard_normal(tracked.dimension))
    q_t, ld_t = tracked.apply(params, xi)
    q_f, ld_f = frozen.apply(params, xi)
    assert np.allclose(np.asarray(q_t), np.asarray(q_f), rtol=1e-12, atol=0)
    assert abs(float(ld_t) - float(ld_f)) < 1e-10 * abs(float(ld_f))
    assert tracked.validate(params)["tracking"]["n_epoch"] == 0


def test_tracked_transport_declares_white_noise_params(psrs, metamath_backend):
    psr = psrs[0]
    tracked, frozen, _kern, p0 = _tracked_and_frozen(psr, psrs, 0)
    assert set(p0) <= set(tracked.as_reparam().params)
    assert not (set(p0) & set(frozen.as_reparam().params))


def test_tracked_metric_is_exact_under_efac_moves(psrs, metamath_backend):
    psr = psrs[0]
    tracked, frozen, kern, p0 = _tracked_and_frozen(psr, psrs, 0)
    params = _hyper_params(tracked, psr)
    # per-backend, non-uniform EFAC moves (a uniform scale is invisible to cond)
    for j, k in enumerate(sorted(k for k in p0 if k.endswith("_efac"))):
        params[k] = p0[k] * (0.6 + 0.5 * j)
    live = metamatrix.func(kern.make_solve)

    def noise_solve(X, _p=dict(params)):
        return live(X, params=_p)
    d_t = tracked.diagnostics(params, noise_solve=noise_solve)
    d_f = frozen.diagnostics(params, noise_solve=noise_solve)
    assert d_t["metric_eig_max"] / d_t["metric_eig_min"] < 1 + 1e-8
    assert d_f["metric_eig_max"] / d_f["metric_eig_min"] > 1.3


def test_tracked_fingerprint_distinguishes_bake_points(psrs, metamath_backend):
    psr = psrs[0]
    tracked, frozen, kern, p0 = _tracked_and_frozen(psr, psrs, 0)
    again, _, _, _ = _tracked_and_frozen(psr, psrs, 0)
    assert tracked.fingerprint() != frozen.fingerprint()
    assert tracked.fingerprint() == again.fingerprint()
    other = tr.Transport(
        _blocks_for(psrs, 0),
        reference_noise=tr.class_tracking(
            kern, {k: (v * 1.1 if k.endswith("_efac") else v) for k, v in p0.items()},
            toaerrs=psr.toaerrs),
        reference_residual=np.asarray(psr.residuals), center=True)
    assert other.fingerprint() != tracked.fingerprint()
    assert "tracking" not in frozen.diagnostics()
    assert frozen._gram({})[0] is frozen._G0


def test_class_tracking_rejects_incomplete_bake_point(psrs, metamath_backend):
    psr = psrs[0]
    p0 = _wn_params0(psr)
    p0.pop(next(iter(p0)))
    with pytest.raises(ValueError, match="missing"):
        tr.class_tracking(_wn_kernel(psr), p0, toaerrs=psr.toaerrs)


def test_array_transport_tracking_matches_stacked_per_pulsar(psrs, metamath_backend):
    per, params = [], {}
    for i, p in enumerate(psrs):
        kern = ds.makenoise_measurement(p, {}, ecorr=True, enterprise=True)
        p0 = {k: v for k, v in p.noisedict.items()
              if k.endswith(("_efac", "_log10_t2equad", "_log10_ecorr"))}
        per.append(tr.Transport(
            _blocks_for(psrs, i),
            reference_noise=tr.class_tracking(kern, p0, toaerrs=p.toaerrs),
            reference_residual=np.asarray(p.residuals), center=True))
        params.update({k: (v * 1.2 if k.endswith("_efac") else v + 0.3) for k, v in p0.items()})
    assert len({t._tracking.n_epoch for t in per}) > 1          # unequal padding exercised
    at = tr.ArrayTransport(per)
    params.update(ds.sample_uniform([q for q in at.params if q not in params]))
    c = kh.jnparray(np.random.default_rng(1).standard_normal((len(psrs), at.dimension)))
    am, ld = at.apply(params, c)
    qs, lds = zip(*[t.apply(params, c[i]) for i, t in enumerate(per)])
    assert np.allclose(np.asarray(am), np.stack([np.asarray(q) for q in qs]), rtol=1e-11, atol=0)
    assert abs(float(ld) - sum(float(x) for x in lds)) < 1e-9 * abs(float(ld))
    with pytest.raises(ValueError, match="all-or-none"):
        tr.ArrayTransport([per[0], _transport_for(psrs[1], psrs, 1)])


def test_arraylikelihood_decenter_params0_tracks_free_white_noise(psrs, metamath_backend):
    p0 = {}
    for p in psrs:
        p0.update(_wn_params0(p))
    model = ds.ArrayLikelihood([_psl_freewn(p) for p in psrs],
                               commongp=_commongp(psrs), decenter=True,
                               decenter_params0=p0)
    clogl = model.clogL
    assert set(p0) <= set(clogl.params)
    coeff = [q for q in clogl.params if "_coefficients(" in q]
    pp = ds.sample_uniform([q for q in clogl.params if q not in coeff])
    rng = np.random.default_rng(0)
    for key in coeff:
        pp[key] = rng.normal(size=int(key[key.index("(") + 1:key.index(")")]))
    out = clogl(pp)
    logp = float(out[0]) if isinstance(out, tuple) else float(out)
    assert np.isfinite(logp)
    # packed clogL needs the cross form (frozen noise): loud failure, not silent
    from discovery.packed import PackedClogLUnsupported
    with pytest.raises(PackedClogLUnsupported, match="cross"):
        model.make_packed_clogL()
    with pytest.raises(ValueError, match="decenter_params0"):
        ds.ArrayLikelihood([_psl_freewn(p) for p in psrs], commongp=_commongp(psrs),
                           decenter_params0=p0)
