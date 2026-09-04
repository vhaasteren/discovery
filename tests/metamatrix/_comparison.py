"""Scale-aware comparison helpers for matrix.py-vs-metamath parity tests.

Tolerance picked by `kind`:
  - 'logL'      : log-likelihood scalars (can be O(1e6)). atol scales with |old|.
  - 'residuals' : seconds. atol fixed at 1e-16 (sub-clock).
  - 'coeffs'    : fourier amplitudes / latent vectors. dimensionless-ish.

Cholesky *factors* are not compared elementwise: a tiny perturbation of a
poorly-conditioned Sigma rearranges L while leaving log-det, ``cho_solve``,
and the realization map ``L^{-T} z`` (what ``sample_conditional`` uses) stable.
Use :func:`assert_cho_close` for ``conditional``'s ``cf``.
"""

import numpy as np
from scipy import linalg as sla


def assert_close(new, old, *, kind, name=""):
    new = np.asarray(new)
    old = np.asarray(old)

    if kind == "logL":
        scale = max(1.0, float(np.max(np.abs(old))))
        np.testing.assert_allclose(new, old, rtol=1e-10, atol=1e-8 * scale,
                                   err_msg=f"{name} logL diverged")
    elif kind == "residuals":
        np.testing.assert_allclose(new, old, rtol=1e-10, atol=1e-16,
                                   err_msg=f"{name} residuals diverged")
    elif kind == "coeffs":
        scale = max(1.0, float(np.max(np.abs(old))))
        np.testing.assert_allclose(new, old, rtol=1e-9, atol=1e-10 * scale,
                                   err_msg=f"{name} coefficients diverged")
    else:
        raise ValueError(f"unknown comparison kind: {kind!r}")


def _lower_factor(cf):
    """``cho_factor`` result -> lower-triangular L with Sigma = L L^T."""
    c, lower = cf
    L = np.asarray(c, dtype=np.float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(
            f"expected a square Cholesky factor; got shape {L.shape}")
    return np.tril(L) if lower else np.triu(L).T


def assert_cho_close(cf_new, cf_old, *, name="", n_draw=8, seed=0):
    """Compare two ``cho_factor`` results by the operators they define.

    Checks log-det(Sigma), ``cho_solve`` on a few unit-Gaussian right-hand
    sides, and the realization map ``L^{-T} z`` used by ``sample_conditional``.
    Does not compare L elementwise.
    """
    L_new, L_old = _lower_factor(cf_new), _lower_factor(cf_old)
    assert L_new.shape == L_old.shape, (
        f"{name} Cholesky shape {L_new.shape} != {L_old.shape}")
    assert bool(cf_new[1]) == bool(cf_old[1]), (
        f"{name} cho_factor lower flags differ: {cf_new[1]!r} vs {cf_old[1]!r}")

    ld_new = 2.0 * float(np.sum(np.log(np.diag(L_new))))
    ld_old = 2.0 * float(np.sum(np.log(np.diag(L_old))))
    assert_close(ld_new, ld_old, kind="logL", name=f"{name}.logdet")

    z = np.random.default_rng(seed).standard_normal((L_old.shape[0], n_draw))
    assert_close(
        sla.cho_solve((L_new, True), z), sla.cho_solve((L_old, True), z),
        kind="coeffs", name=f"{name}.solve")
    # sample_conditional: y = L^{-T} z  (N(0, Sigma^{-1}) draws)
    assert_close(
        sla.solve_triangular(L_new.T, z, lower=False),
        sla.solve_triangular(L_old.T, z, lower=False),
        kind="coeffs", name=f"{name}.draw")


def assert_params_equal(new_fn, old_fn, name=""):
    new_p, old_p = set(new_fn.params), set(old_fn.params)
    only_old = old_p - new_p
    only_new = new_p - old_p
    assert not (only_old or only_new), (
        f"{name} param drift — only old: {sorted(only_old)}, "
        f"only new: {sorted(only_new)}"
    )
