"""Gates for the fully-fp32 RN-marginalized / GWB-decentered kernel (discovery.fp32gwb)."""
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import discovery as ds
from discovery import fp32gwb
from discovery.samplers import numpyro as ds_numpyro


PRIORDICT = {r"(.*_)?rednoise_log10_A.*": [-20.0, -10.0], r"(.*_)?rednoise_gamma.*": [0.0, 7.0],
             r"gw_(.*_)?log10_A": [-20.0, -10.0], r"gw_(.*_)?gamma": [0.0, 7.0]}


@pytest.fixture(scope="module")
def built(psrs):
    ds.config(kernels="metamath")
    try:
        T = ds.getspan(psrs)
        commongp = ds.makecommongp_fourier(psrs, ds.powerlaw, components=10, T=T, name="rednoise")
        globalgp = ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, components=5, T=T, name="gw")
        kern = fp32gwb.make_gwb_fp32(psrs, commongp, globalgp)
        # hard-clip, no-ceiling twin: evaluates exactly Discovery's prior (for the marginal gate)
        kern_exact = fp32gwb.make_gwb_fp32(psrs, commongp, globalgp, soft_clip_dex=0.0, kappa=np.inf)
        # Discovery's own two-block HD model, for the marginal gate
        model = ds.ArrayLikelihood(
            [ds.PulsarLikelihood([p.residuals, ds.makenoise_measurement(p, p.noisedict),
                                  ds.makegp_ecorr(p, p.noisedict), ds.makegp_timing(p, svd=True)]) for p in psrs],
            commongp=commongp, globalgp=globalgp)
        yield kern, model, kern_exact
    finally:
        ds.config(kernels="matrix")


def _draw(kern, rng):
    return {name: rng.uniform(*ds.getprior_uniform(name, PRIORDICT)) for name in kern.params}


def test_layout_and_diagnostics(built):
    kern, _, _ = built
    d = kern.diagnostics()
    assert d["npsr"] == 3 and d["k"] == 20 and d["kg"] == 10 and d["xi_shape"] == (3, 10)
    assert all(1 <= r <= 20 for r in d["ranks"])
    assert all(x < kern.info_tol for x in d["max_info_dropped_nats"])
    assert set(kern.params) == {f"{p.name}_rednoise_{s}" for p in kern.psrs for s in ("log10_A", "gamma")} | {"gw_log10_A", "gw_gamma"}


def test_marginal_matches_discovery_logL_up_to_a_constant(built):
    """Dense marginal of the (RN, GWB) two-block HD model == ArrayLikelihood.logL + const
    (hard-clip kernel twin, so both evaluate exactly the same prior)."""
    _, model, kern = built
    rng = np.random.default_rng(1); offs = []
    for _ in range(4):
        p = _draw(kern, rng)
        offs.append(float(model.logL(p)) - kern.reference_marginal(p))
    assert np.ptp(offs) < 1e-6 * max(1.0, abs(offs[0]))


def test_kernel_equals_density_plus_jacobian(built):
    """float64 kernel == exact RN-marginalized density at the physical GW coefficients + log|da/dxi| + const."""
    kern, _, _ = built
    rng = np.random.default_rng(2); offs = []
    for _ in range(4):
        p = _draw(kern, rng); xi = rng.normal(size=kern.xi_shape)
        lp, aE = kern.logp(p, xi, dtype=jnp.float64, with_base=True)
        offs.append(float(lp) - (kern.reference_logp(p, np.asarray(aE)) + kern.reference_jacobian(p)))
    assert np.ptp(offs) < 1e-3          # deflation loss is bounded by info_tol per dropped direction


def test_fp32_matches_fp64_everywhere_and_is_finite(built):
    kern, _, _ = built
    rng = np.random.default_rng(3); dl, dg = [], []
    theta64 = lambda p: kern.pack(p).astype(jnp.float64)
    f = lambda th, xi: kern.kernel(th, xi)[0]
    g = jax.jit(jax.grad(lambda th, xi: kern.kernel(th, xi)[0], argnums=(0, 1)))
    for _ in range(30):
        p = _draw(kern, rng); xi = rng.normal(size=kern.xi_shape)
        th = theta64(p)
        a, b = float(f(th, xi)), float(f(th.astype(jnp.float32), jnp.asarray(xi, jnp.float32)))
        assert np.isfinite(b)
        dl.append(abs(a - b))
        ga = np.concatenate([np.asarray(x).ravel() for x in g(th, jnp.asarray(xi))])
        gb = np.concatenate([np.asarray(x).ravel() for x in g(th.astype(jnp.float32), jnp.asarray(xi, jnp.float32))])
        dg.append(np.linalg.norm(ga - gb) / np.linalg.norm(ga))
    assert np.median(dl) < 0.05 and max(dl) < 1.0
    assert np.median(dg) < 1e-3 and max(dg) < 0.1


def test_packed_and_named_agree(built):
    kern, _, _ = built
    rng = np.random.default_rng(4); p = _draw(kern, rng); xi = rng.normal(size=kern.xi_shape)
    lp_named, _ = kern.logp(p, xi, dtype=jnp.float64)
    lp_packed, _ = kern.kernel(kern.pack(p).astype(jnp.float64), xi)
    assert abs(float(lp_named) - float(lp_packed)) < 1e-6
    assert kern.unpack(kern.pack(p)) .keys() == p.keys()


def test_numpyro_model_density_is_finite(built):
    import numpyro
    from numpyro.infer.util import log_density
    kern, _, _ = built
    model = ds_numpyro.makemodel_gwb_fp32(kern, PRIORDICT)
    rng = np.random.default_rng(5); p = _draw(kern, rng)
    ld, _ = log_density(model, (), {}, {"theta": kern.pack(p), "xi": jnp.zeros(kern.xi_shape)})
    assert np.isfinite(float(ld))
    df = model.to_df({"theta": np.asarray(kern.pack(p))[None, :]})
    assert set(df.columns) == set(kern.params)


def test_extsignal_cw_matches_reference(psrs):
    """With a CW ExtSignal (discovery.makecw_extsignal) the kernel still equals density + Jacobian + const."""
    ds.config(kernels="metamath")
    try:
        T = ds.getspan(psrs)
        commongp = ds.makecommongp_fourier(psrs, ds.powerlaw, components=10, T=T, name="rednoise")
        globalgp = ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, components=5, T=T, name="gw")
        cw = ds.makecw_extsignal(psrs, components=20, T=T, pulsarterm=True, name="cw")
        kern = fp32gwb.make_gwb_fp32(psrs, commongp, globalgp, extsignals=[cw])
    finally:
        ds.config(kernels="matrix")
    assert set(cw.params) <= set(kern.params)
    rng = np.random.default_rng(6); offs = []
    for _ in range(3):
        p = {name: rng.uniform(*ds.getprior_uniform(name, PRIORDICT)) for name in kern.params}
        p["cw_log10_h0"] = rng.uniform(-16.0, -14.0)      # a CW the data could plausibly host
        xi = rng.normal(size=kern.xi_shape)
        lp, aE = kern.logp(p, xi, dtype=jnp.float64, with_base=True)
        assert np.isfinite(float(lp))
        offs.append(float(lp) - (kern.reference_logp(p, np.asarray(aE)) + kern.reference_jacobian(p)))
        # Discovery's makefourier_binary coefficient map is not fp32-safe (sine differences over a
        # small delta-omega); evaluate the coefficients in float64 and feed them to the fp32 kernel.
        coeffs = [jnp.asarray(cc, jnp.float32) for cc in kern.ext_coefficients(p, jnp.float64)]
        lp32, _ = kern.kernel(kern.pack(p).astype(jnp.float32), jnp.asarray(xi, jnp.float32), ext_coeffs=coeffs)
        lp64_nobase, _ = kern.logp(p, xi, dtype=jnp.float64, with_base=False)     # kernel() carries no -1/2 xi^T xi
        assert np.isfinite(float(lp32)) and abs(float(lp32) - float(lp64_nobase)) < 0.1 + 1e-5 * abs(float(lp) - kern.const)
    assert np.ptp(offs) < 1e-3
