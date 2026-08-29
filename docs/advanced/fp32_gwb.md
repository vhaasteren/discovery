# Fully-fp32 GWB likelihood (`discovery.fp32gwb`)

`GWBMarginalFp32` is a Cholesky-only, float32-safe likelihood for the canonical
"intrinsic red noise + HD-correlated GWB" model. Per pulsar the red-noise
coefficients are marginalized analytically; the GWB coefficients
(`2*components_gw` per pulsar, on the leading columns of the same Fourier basis)
are sampled through a decentering `a = mu(theta) + L(theta)^-T xi` against the
live conditional; the HD prior is exact (`Gamma (x) diag(phi_GW)` through a
constant `chol(Gamma)`). It evaluates the same marginal model as
`ArrayLikelihood(commongp=..., globalgp=...)` (tested to a constant), with every
theta-independent constant kept in float64.

```python
ds.config(kernels="metamath")
T = ds.getspan(psrs)
commongp = ds.makecommongp_fourier(psrs, ds.powerlaw, components=30, T=T, name="red_noise")
globalgp = ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, components=14, T=T, name="gw")
kern = ds.fp32gwb.make_gwb_fp32(psrs, commongp, globalgp)       # bake (float64, once)

jax.config.update("jax_default_matmul_precision", "highest")       # mandatory on GPU
model = ds.samplers.numpyro.makemodel_gwb_fp32(kern, priordict)
# sites: theta (Uniform, Discovery parameter names), xi ~ N(0, I) of shape kern.xi_shape
```

## Why it is float32-safe

* **Bake (float64):** closed-form `K^-1/2` whitening (diagonal + ECORR), exact
  timing-model projection (QR of the whitened `M` basis; the flat-prior
  marginal depends only on the span), SVD of the whitened projected Fourier
  basis. Directions carrying less than `info_tol` (1e-6) nats of information at
  the prior ceiling have no likelihood dependence; their coordinates integrate
  out exactly under the prior, leaving the marginal prior `V_r^T Phi V_r` on the
  data range. Nothing is dropped from the model; `diagnostics()` reports the
  ranks and the information in the removed directions.
* **Hot path:** only `I + (bounded PSD)` matrices are factorized (`I + S Phi_r S`
  and `I + Y~^T Y~`), whose entries are live per-mode SNR^2 and whose pivots are
  >= 1. The prior covariance `Phi_r` (13-decade spectrum) and any
  `I + low-rank` matrix with exact null directions are never formed; the
  red-noise quadratic is `1/2(||u||^2 - ||L^-1 u||^2)` with bounded whitened
  residual projections `u`.
* **Ceiling:** `phi_f <= kappa / G_ff` (soft, `kappa = 1e6`): a prior wider than
  a million times the data variance at a mode is indistinguishable from flat,
  and it bounds the fp32 range at the prior corners (NG15's loudest mode is
  ~1e5). Units are ns.
* Baked constants (`~1e5-1e7`) are added in float64; `jax_default_matmul_precision`
  must be `'highest'` (TF32 Gram entries cost 1-2% gradient error).

Measured on NANOGrav 15yr (67 pulsars, RTX 4090): fp32 vs fp64 kernel
`|dlogp|` median 7e-3 nat over the whole prior box, gradient relative error
median 3e-5, no non-finite values; NUTS 1000+4000 in ~700 s with 127 leapfrogs
per iteration, matching Prometheus's chain.

## Assumptions / limits

* White noise is frozen (noise dictionary); the timing model is `psr.Mmat`
  (rank-deficient `M` is a user error); the GW basis must be the leading
  columns of the red-noise basis (same `T`).
* Only diagonal red-noise priors and a separable (`ORF x spectrum`) GW prior.
* Below the (1 ns)^2 floor the density is flat in the amplitude; warmup chains
  can wander there. A soft pull is not part of this module.
