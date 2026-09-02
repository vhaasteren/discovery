"""Tracer guards: graphs must fold outside jit/grad, not cache leaked tracers."""
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


def test_first_logl_inside_jit_raises(psr, metamath_backend):
    """(1) jax.jit(lambda p: model.logL(p)) on a fresh model → TypeError."""
    template = R.full_rn(psr)
    p0 = ds.sample_uniform(template.logL.params)
    psl = R.full_rn(psr)
    assert "logL" not in psl.__dict__
    with pytest.raises(TypeError, match="JAX tracer"):
        jax.jit(lambda p: psl.logL(p))(p0)


def test_residual_swap_inside_jit_raises(psr, metamath_backend):
    """(2) assigning residuals inside a jitted function → TypeError."""
    psl = R.full_rn(psr)
    p0 = ds.sample_uniform(psl.logL.params)
    psl.build()
    y = np.zeros_like(psr.residuals)

    @jax.jit
    def f(p, residual):
        psl.residuals = residual
        return psl.logL(p)

    with pytest.raises(TypeError, match="JAX tracer"):
        f(p0, y)


def test_build_then_jit_logl_works(psr, metamath_backend):
    """(3) model.build(); jax.jit(model.logL) works."""
    psl = R.full_rn(psr)
    p0 = ds.sample_uniform(psl.logL.params)
    psl.build()
    jlogL = jax.jit(psl.logL)
    val = float(jlogL(p0))
    assert np.isfinite(val)
    np.testing.assert_allclose(val, float(psl.logL(p0)), rtol=1e-10)
