"""Packed clogL layout, eligibility, and fused-kernel parity."""

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import discovery as ds
import discovery.recipes as R
from discovery.packed import (
    CoefficientArrayLayout,
    PackedClogLUnsupported,
    pack_with_layout,
    packed_clogl_diagnostics,
)
from discovery.params import Params, make_layout


@pytest.fixture
def metamath_backend():
    ds.config(kernels="metamath")
    yield
    ds.config(kernels="matrix")


def _fill_params(model, seed=0):
    clogl = model.clogL
    coeff = [p for p in clogl.params if "_coefficients(" in p]
    hyper = [p for p in clogl.params if p not in coeff]
    known, unknown = [], []
    for name in hyper:
        try:
            ds.sample_uniform([name])
            known.append(name)
        except KeyError:
            unknown.append(name)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    p0 = ds.sample_uniform(known) if known else {}
    for name in unknown:
        p0[name] = float(rng.normal())
    for name in coeff:
        width = int(name[name.index("(") + 1:name.index(")")])
        p0[name] = rng.normal(size=width)
    return p0


def _logp(out):
    return float(out[0] if isinstance(out, tuple) else out)


def test_theta_layout_round_trip():
    names = ["scalar", "vec(3)", "mat(2,2)"]
    template = {
        "scalar": 1.5,
        "vec(3)": np.arange(3.0),
        "mat(2,2)": np.arange(4.0).reshape(2, 2),
    }
    layout, size = make_layout(names, template=template)
    raw = pack_with_layout(template, layout)
    assert raw.shape == (size,)
    restored = dict(Params(raw, layout))
    assert float(restored["scalar"]) == 1.5
    np.testing.assert_array_equal(restored["vec(3)"], template["vec(3)"])
    np.testing.assert_array_equal(restored["mat(2,2)"], template["mat(2,2)"])


def test_unsuffixed_array_uses_template():
    names = ["plain"]
    template = {"plain": np.arange(4.0)}
    layout, size = make_layout(names, template=template)
    assert size == 4
    raw = pack_with_layout(template, layout)
    with pytest.raises(ValueError, match="flattened size"):
        pack_with_layout({"plain": 1.0}, layout)


def test_coefficient_layout_localizes_transport_slices(psrs, metamath_backend):
    model = R.decenter_intrinsic_rn_global_hd(psrs)
    transport = model._build_decenter_transport(model._coefficient_assembly[1])
    layout = CoefficientArrayLayout.build(transport.transports)
    assert layout.shape[0] == len(psrs)
    first = transport.transports[0].index
    offset = 0
    for name, local in first.items():
        assert local.start == offset
        offset = local.stop


def test_unequal_row_widths_raise(psrs, metamath_backend):
    model = R.decenter_intrinsic_rn(psrs)
    transport = model._build_decenter_transport(model._coefficient_assembly[1])
    transport.transports[1].index[next(iter(transport.transports[1].index))] = slice(0, 1)
    with pytest.raises(ValueError, match="equal row widths"):
        CoefficientArrayLayout.build(transport.transports)


def test_packed_named_parity_intrinsic_rn(psrs, metamath_backend):
    model = R.decenter_intrinsic_rn(psrs)
    packed = model.make_packed_clogL()
    p0 = _fill_params(model)
    theta, xi = packed.pack(p0)
    fused = packed(theta, xi)
    named = packed.oracle(theta, xi)
    np.testing.assert_allclose(_logp(fused), _logp(named), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(fused[1]), np.asarray(named[1]), rtol=1e-10)
    assert len(jax.tree_util.tree_leaves((theta, xi))) == 2


def test_packed_named_parity_rn_hd(psrs, metamath_backend):
    model = R.decenter_intrinsic_rn_global_hd(psrs)
    packed = model.make_packed_clogL()
    p0 = _fill_params(model)
    theta, xi = packed.pack(p0)
    fused = packed(theta, xi)
    named = packed.oracle(theta, xi)
    np.testing.assert_allclose(_logp(fused), _logp(named), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(fused[1]), np.asarray(named[1]), rtol=1e-10)


def test_packed_named_value_and_gradient_parity_rn_hd_cw(psrs, metamath_backend):
    model = R.decenter_extsignal_cw_global_hd(psrs)
    packed = model.make_packed_clogL()
    p0 = _fill_params(model, seed=3)
    theta, xi = packed.pack(p0)

    def fused_logp(t, x):
        return packed(t, x)[0]

    def oracle_logp(t, x):
        return packed.oracle(t, x)[0]

    np.testing.assert_allclose(
        float(fused_logp(theta, xi)), float(oracle_logp(theta, xi)), rtol=1e-12)

    v_theta = jnp.asarray(np.random.default_rng(4).normal(size=theta.shape))
    v_xi = jnp.asarray(np.random.default_rng(5).normal(size=xi.shape))
    fused_dir = jax.jvp(fused_logp, (theta, xi), (v_theta, v_xi))[1]
    oracle_dir = jax.jvp(oracle_logp, (theta, xi), (v_theta, v_xi))[1]
    np.testing.assert_allclose(float(fused_dir), float(oracle_dir), rtol=2e-10)


def test_wrong_shapes_raise(psrs, metamath_backend):
    model = R.decenter_intrinsic_rn(psrs)
    packed = model.make_packed_clogL()
    p0 = _fill_params(model)
    theta, xi = packed.pack(p0)
    with pytest.raises(ValueError, match="theta has shape"):
        packed.unpack(theta[:-1], xi)
    with pytest.raises(ValueError, match="xi has shape"):
        packed.coefficients.unpack(xi[:, :-1])


@pytest.mark.parametrize("builder,fragment", [
    (lambda psrs: R.intrinsic_rn(psrs), "ArrayTransport/decenter"),
    (lambda psrs: ds.ArrayLikelihood(
        R.decenter_intrinsic_rn(psrs).psls,
        commongp=R.decenter_intrinsic_rn(psrs).commongp,
        decenter=True,
        clogl_form="residual",
    ), "cross form"),
    (lambda psrs: ds.ArrayLikelihood(
        R.decenter_intrinsic_rn(psrs).psls,
        commongp=R.decenter_intrinsic_rn(psrs).commongp,
        decenter=True,
        transform=lambda params, c: (c, 0.0),
    ), "user transforms"),
    (lambda psrs: R.means_on_commongp(psrs), "GP prior means"),
])
def test_ineligibility_reasons(psrs, metamath_backend, builder, fragment):
    model = builder(psrs)
    report = packed_clogl_diagnostics(model)
    assert report["eligible"] is False
    assert any(fragment in reason for reason in report["reasons"])
    with pytest.raises(PackedClogLUnsupported, match=fragment):
        model.make_packed_clogL()


def test_non_separable_global_is_ineligible(psrs, metamath_backend):
    def untagged(f, df, log10_A, gamma):
        return ds.powerlaw(f, df, log10_A, gamma)

    T = ds.getspan(psrs)
    model = ds.ArrayLikelihood(
        R.decenter_intrinsic_rn(psrs).psls,
        commongp=ds.makecommongp_fourier(
            psrs, ds.powerlaw, components=6, T=T, name="rednoise"),
        globalgp=ds.makeglobalgp_fourier(
            psrs, untagged, ds.hd_orf, components=4, T=T, name="gw"),
        decenter=True,
    )
    report = packed_clogl_diagnostics(model)
    assert report["eligible"] is False
    assert any("separable" in reason for reason in report["reasons"])


def test_jit_pytree_has_two_leaves(psrs, metamath_backend):
    model = R.decenter_intrinsic_rn(psrs)
    packed = model.make_packed_clogL()
    p0 = _fill_params(model)
    theta, xi = packed.pack(p0)
    compiled = jax.jit(lambda t, x: packed(t, x)[0])
    assert np.isfinite(float(compiled(theta, xi)))
    assert len(jax.tree_util.tree_leaves((theta, xi))) == 2


def test_fused_jaxpr_has_no_dense_global_covariance(psrs, metamath_backend):
    model = R.decenter_intrinsic_rn_global_hd(psrs)
    packed = model.make_packed_clogL()
    p0 = _fill_params(model)
    theta, xi = packed.pack(p0)
    closed = jax.make_jaxpr(lambda t, x: packed(t, x)[0])(theta, xi)
    npsr, k_gw = len(psrs), model.globalgp.separable_prior.width
    forbidden = (npsr * k_gw, npsr * k_gw)
    shapes = []
    for eqn in closed.jaxpr.eqns:
        for out in eqn.outvars:
            aval = getattr(out, "aval", None)
            shape = getattr(aval, "shape", None)
            if shape:
                shapes.append(tuple(shape))
    assert forbidden not in shapes


def test_named_and_packed_independently_callable(psrs, metamath_backend):
    model = R.decenter_intrinsic_rn(psrs)
    p0 = _fill_params(model)
    packed = model.make_packed_clogL()
    theta, xi = packed.pack(p0)
    named_first = _logp(model.clogL(p0))
    packed_val = _logp(packed(theta, xi))
    named_again = _logp(model.clogL(p0))
    np.testing.assert_allclose([packed_val, named_again], named_first, rtol=1e-12)


def test_diagnostics_eligible_shape(psrs, metamath_backend):
    model = R.decenter_intrinsic_rn_global_hd(psrs)
    packed = model.make_packed_clogL()
    report = packed.diagnostics()
    assert report["eligible"] is True
    assert report["npsr"] == len(psrs)
    assert report["xi_shape"] == packed.xi_shape
    assert report["global_prior"] == "separable_fourier"
    assert report["jit_input_leaves"] == 2


def test_numpyro_sites_and_base_density_cancellation(psrs, metamath_backend):
    from numpyro.infer.util import log_density
    from discovery.samplers import numpyro as ds_numpyro

    model = R.decenter_intrinsic_rn(psrs)
    packed = model.make_packed_clogL()
    numpyro_model = ds_numpyro.makemodel_packed(model)
    p0 = _fill_params(model, seed=6)
    theta, xi = packed.pack(p0)
    params = {"theta": theta, "xi": xi}

    joint, _ = log_density(numpyro_model, (), {}, params)
    logp = float(packed(theta, xi)[0])
    n_xi = int(np.prod(xi.shape))
    lows, highs = [], []
    for name, start, stop, _shape in packed.theta_layout:
        lo, hi = ds.getprior_uniform(name)
        lows.extend([lo] * (stop - start))
        highs.extend([hi] * (stop - start))
    uniform = -np.sum(np.log(np.asarray(highs) - np.asarray(lows)))
    expected = logp + uniform - 0.5 * n_xi * np.log(2.0 * np.pi)
    np.testing.assert_allclose(float(joint), expected, rtol=1e-10)

    xi2 = xi + 0.3
    joint2, _ = log_density(numpyro_model, (), {}, {"theta": theta, "xi": xi2})
    logp2 = float(packed(theta, xi2)[0])
    np.testing.assert_allclose(float(joint2 - joint), logp2 - logp, rtol=1e-10)

    df = packed.samples_to_df({
        "theta": np.stack([np.asarray(theta), np.asarray(theta)]),
        "xi": np.stack([np.asarray(xi), np.asarray(xi)]),
    })
    assert df.shape[0] == 2
    assert packed.theta_names[0] in df.columns or any(
        col.startswith(packed.theta_names[0].split("(")[0]) for col in df.columns
    )


def test_packed_checkpoint_round_trip(psrs, metamath_backend, tmp_path):
    import pandas as pd
    from discovery.samplers import numpyro as ds_numpyro

    model = R.decenter_intrinsic_rn(psrs)
    numpyro_model = ds_numpyro.makemodel_packed(model)
    sampler = ds_numpyro.makesampler_nuts(
        numpyro_model, num_warmup=4, num_samples=4, progress_bar=False,
    )
    outdir = tmp_path / "packed-chains"
    ds_numpyro.run_nuts_with_checkpoints(
        sampler, 2, jax.random.PRNGKey(1), outdir=outdir,
    )
    df = pd.read_feather(outdir / "numpyro-samples.feather")
    packed = numpyro_model.packed_clogL
    assert len(df) == 4
    assert packed.theta_names[0] in df.columns or any(
        col.startswith(packed.theta_names[0].split("(")[0]) for col in df.columns
    )


def test_fused_constants_are_baked_in_float64_under_float32_working(psrs, metamath_backend):
    """The fused kernel's frozen products (F^T N^-1 F, N^-1 F^T y, y^T N^-1 y,
    log det N) and the transport's G0/b0 are baked in float64 regardless of
    `config(working=float32)`, and the packed log-density under the float32
    configuration matches the float64 one. Before this guarantee the float32
    bake left F^T N^-1 F indefinite and the frozen scalars (~1e7) resolved to
    O(1), which froze NUTS at step sizes ~1e-10."""
    model64 = R.decenter_intrinsic_rn_global_hd(psrs)
    packed64 = model64.make_packed_clogL()
    p0 = _fill_params(model64)
    theta, xi = packed64.pack(p0)
    ref = _logp(packed64(theta, xi))

    ds.utils.config(backend="jax", factor="cholesky", working=jnp.float32)
    try:
        model32 = R.decenter_intrinsic_rn_global_hd(psrs)
        packed32 = model32.make_packed_clogL()
        out = packed32(theta, xi)
    finally:
        ds.utils.config(backend="jax", factor="cholesky")

    assert packed32.transport._G0.dtype == jnp.float64
    assert packed32.transport._b0.dtype == jnp.float64
    assert out[0].dtype == jnp.float64
    for G in np.asarray(packed32.transport._G0):
        lam = np.linalg.eigvalsh(G)
        assert lam[0] >= -1e-9 * lam[-1]
    np.testing.assert_allclose(_logp(out), ref, rtol=1e-9)
