"""Structured Fourier prior and existing-graph covariance optimizations."""

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import discovery as ds
import discovery.recipes as R
from discovery import metamath
from discovery import signals
from discovery.structured import SeparableFourierPrior, separable_contrib


@pytest.fixture
def metamath_backend():
    ds.config(kernels="metamath")
    yield
    ds.config(kernels="matrix")


def _spd_orf(npsr, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(npsr, npsr))
    return a @ a.T + npsr * np.eye(npsr)


@pytest.mark.parametrize("npsr,width", [(2, 4), (3, 8), (8, 12)])
def test_separable_prior_matches_dense_value(npsr, width):
    rng = np.random.default_rng(1)
    orf = _spd_orf(npsr, seed=2)
    prior = SeparableFourierPrior.build(orf, lambda params: params["phi"], width)
    for _ in range(5):
        phi = np.exp(rng.normal(size=width))
        c = rng.normal(size=(npsr, width))
        got = float(separable_contrib(
            jnp.asarray(c), jnp.asarray(phi),
            prior.orf_cholesky, prior.orf_logdet,
        ))
        dense = float(metamath.dense_coefficient_logprior_legacy(
            jnp.asarray(c),
            jnp.kron(jnp.asarray(orf), jnp.diag(jnp.asarray(phi))),
        ))
        np.testing.assert_allclose(got, dense, rtol=2e-12)


def test_separable_prior_matches_dense_spectrum_gradient():
    npsr, width = 4, 6
    orf = _spd_orf(npsr, seed=3)
    prior = SeparableFourierPrior.build(orf, lambda params: params["phi"], width)
    c = jnp.asarray(np.random.default_rng(4).normal(size=(npsr, width)))

    def sep(logphi):
        phi = jnp.exp(logphi)
        return separable_contrib(c, phi, prior.orf_cholesky, prior.orf_logdet)

    def dense(logphi):
        phi = jnp.exp(logphi)
        return metamath.dense_coefficient_logprior_legacy(
            c, jnp.kron(jnp.asarray(orf), jnp.diag(phi)))

    logphi = jnp.asarray(np.random.default_rng(5).normal(size=width))
    np.testing.assert_allclose(
        jax.grad(sep)(logphi), jax.grad(dense)(logphi), rtol=2e-11)


def test_separable_prior_pulsar_major_order():
    orf = np.array([[2.0, 0.3], [0.3, 1.5]])
    phi = np.array([0.4, 0.9, 1.2, 0.7])
    c = np.array([[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0]])
    prior = SeparableFourierPrior.build(orf, lambda params: params["phi"], 4)
    got = float(separable_contrib(
        jnp.asarray(c), jnp.asarray(phi),
        prior.orf_cholesky, prior.orf_logdet,
    ))
    dense = float(metamath.dense_coefficient_logprior_legacy(
        jnp.asarray(c), jnp.kron(jnp.asarray(orf), jnp.diag(jnp.asarray(phi)))))
    np.testing.assert_allclose(got, dense, rtol=2e-12)


def test_kron_matches_block_expression():
    rng = np.random.default_rng(6)
    orf = _spd_orf(5, seed=7)
    phi = np.exp(rng.normal(size=8))
    kron = signals._orf_spectrum_covariance(jnp.asarray(phi), jnp.asarray(orf))
    block = signals._orf_spectrum_covariance_block(jnp.asarray(phi), jnp.asarray(orf))
    np.testing.assert_allclose(kron, block, rtol=1e-13)


def test_one_factor_lu_matches_legacy_dense():
    rng = np.random.default_rng(8)
    a = rng.normal(size=(12, 12))
    # Nonsingular, not necessarily SPD.
    phi = a @ a.T + 0.1 * np.eye(12)
    phi[0, 1] += 0.2
    c = rng.normal(size=(3, 4))
    np.testing.assert_allclose(
        metamath.dense_coefficient_logprior(jnp.asarray(c), jnp.asarray(phi)),
        metamath.dense_coefficient_logprior_legacy(jnp.asarray(c), jnp.asarray(phi)),
        rtol=2e-12,
    )


def test_analytic_inverse_matches_dense():
    orf = _spd_orf(3, seed=9)
    phi = np.exp(np.linspace(-1.0, 0.5, 6))
    cov = np.kron(orf, np.diag(phi))
    inverse = np.linalg.inv(cov)
    _, logabsdet = np.linalg.slogdet(cov)
    c = np.random.default_rng(10).normal(size=(3, 6))
    np.testing.assert_allclose(
        metamath.inverse_coefficient_logprior(
            jnp.asarray(c), jnp.asarray(inverse), logabsdet),
        metamath.dense_coefficient_logprior_legacy(
            jnp.asarray(c), jnp.asarray(cov)),
        rtol=2e-12,
    )


def test_non_positive_definite_orf_uses_dense_fallback_without_jitter():
    orf = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="positive definite"):
        SeparableFourierPrior.build(orf, lambda params: params["phi"], 2)


def test_asymmetric_orf_uses_dense_fallback():
    orf = np.array([[2.0, 0.4], [0.1, 1.5]])
    with pytest.raises(ValueError, match="symmetric"):
        SeparableFourierPrior.build(orf, lambda params: params["phi"], 2)


def test_single_hd_factory_has_separable_metadata(psrs, metamath_backend):
    gp = ds.makeglobalgp_fourier(
        psrs, ds.powerlaw, ds.hd_orf, components=5, T=ds.getspan(psrs), name="gw")
    assert getattr(gp, "separable_prior", None) is not None
    assert gp.separable_prior.width == 10
    assert ds.powerlaw.fourier_covariance == "diagonal"


def test_compound_or_untagged_globalgp_has_no_separable_metadata(psrs, metamath_backend):
    def untagged(f, df, log10_A, gamma):
        return ds.powerlaw(f, df, log10_A, gamma)

    gp = ds.makeglobalgp_fourier(
        psrs, untagged, ds.hd_orf, components=4, T=ds.getspan(psrs), name="gw")
    assert getattr(gp, "separable_prior", None) is None


def test_mixed_logprior_uses_separable_node_not_dense_phi(psrs, metamath_backend):
    model = R.decenter_intrinsic_rn_global_hd(psrs)
    commongp = model.commongp
    globalgp = model.globalgp
    prior = metamath.CompoundGP([commongp, globalgp]).prior
    descriptions = [
        getattr(node, "description", "") or ""
        for node in prior.values()
    ]
    assert any("separable Fourier logprior" in text for text in descriptions)
    assert not any("dense logprior" in text for text in descriptions)


def test_general_dense_fallback_unchanged(psrs, metamath_backend):
    def untagged(f, df, log10_A, gamma):
        return ds.powerlaw(f, df, log10_A, gamma)

    gp = ds.makeglobalgp_fourier(
        psrs, untagged, ds.hd_orf, components=4, T=ds.getspan(psrs), name="gw")
    assert gp.Phi_inv is not None
    prior = metamath.CompoundGP([model_commongp(psrs), gp]).prior
    descriptions = [
        getattr(node, "description", "") or ""
        for node in prior.values()
    ]
    assert any("analytic-inverse logprior" in text for text in descriptions)


def model_commongp(psrs):
    return ds.makecommongp_fourier(
        psrs, ds.powerlaw, components=6, T=ds.getspan(psrs), name="rednoise")
