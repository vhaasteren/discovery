"""Residual-swap cache invalidation on the three metamath likelihood classes."""
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import discovery as ds  # noqa: E402
import discovery.recipes as R  # noqa: E402


@pytest.fixture
def metamath_backend():
    ds.config(kernels="metamath")
    yield
    ds.config(kernels="matrix")


def _zeros(psr):
    return np.zeros_like(psr.residuals)


def _params(model, rng=None):
    """Hyperparameters from the standard priors plus coefficient draws.

    `clogL` requires the coefficient keys, which are not in `logL.params`.
    """
    keys = sorted(set(model.logL.params) | set(model.clogL.params))
    coeff = [k for k in keys if "_coefficients(" in k]
    p0 = ds.sample_uniform([k for k in keys if k not in coeff])
    rng = np.random.default_rng(0) if rng is None else rng
    for k in coeff:
        width = int(k[k.index("(") + 1:k.index(")")])
        p0[k] = rng.normal(size=width)
    return p0


def test_pulsar_residual_swap_before_first_logl(psr, metamath_backend):
    """(i) psl.residuals = r before first logL access changes logL/clogL."""
    template = R.full_rn(psr)
    p0 = _params(template)
    L_orig = float(template.logL(p0))

    psl = R.full_rn(psr)
    assert "logL" not in psl.__dict__
    psl.residuals = _zeros(psr)
    L_swapped = float(psl.logL(p0))
    clog_swapped = psl.clogL(p0)
    clog_val = clog_swapped[0] if isinstance(clog_swapped, tuple) else clog_swapped

    assert L_swapped != L_orig
    assert np.isfinite(float(clog_val))


def test_pulsar_residual_swap_invalidates_clogl_and_conditional(psr, metamath_backend):
    """(ii) after logL was accessed, swap invalidates clogL/conditional too."""
    psl = R.full_rn(psr)
    p0 = _params(psl)
    L_orig = float(psl.logL(p0))
    _ = psl.clogL(p0)
    cond_orig = psl.conditional(p0)
    assert "logL" in psl.__dict__
    assert "clogL" in psl.__dict__
    assert "conditional" in psl.__dict__

    psl.residuals = _zeros(psr)
    assert "logL" not in psl.__dict__
    assert "clogL" not in psl.__dict__
    assert "conditional" not in psl.__dict__

    L_new = float(psl.logL(p0))
    clog_new = psl.clogL(p0)
    cond_new = psl.conditional(p0)
    clog_val = clog_new[0] if isinstance(clog_new, tuple) else clog_new

    assert L_new != L_orig
    assert np.isfinite(float(clog_val))
    assert not np.allclose(cond_orig[0], cond_new[0])


def test_global_residual_swap_changes_logl(psrs, metamath_backend):
    """(iii) GlobalLikelihood.residuals = ys changes logL (globalgp is None)."""
    template = R.no_global(psrs)
    p0 = ds.sample_uniform(template.logL.params)
    L_orig = float(template.logL(p0))

    gl = R.no_global(psrs)
    _ = float(gl.logL(p0))
    gl.residuals = [_zeros(p) for p in psrs]
    L_swapped = float(gl.logL(p0))
    assert L_swapped != L_orig


def test_array_transport_reassignment_raises(psrs, metamath_backend):
    """(iv) al.transport = ... post-init raises AttributeError."""
    al = R.intrinsic_rn(psrs)
    with pytest.raises(AttributeError, match="transport"):
        al.transport = None


def test_array_residual_swap_changes_built_logl(psrs, metamath_backend):
    """(v) al.residuals = ys changes an already-built al.logL."""
    al = R.intrinsic_rn(psrs)
    p0 = ds.sample_uniform(al.logL.params)
    L_orig = float(al.logL(p0))
    al.residuals = [_zeros(p) for p in psrs]
    L_swapped = float(al.logL(p0))
    assert L_swapped != L_orig
