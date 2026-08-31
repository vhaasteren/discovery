# Metamatrix: graph-based kernels (user guide)

This page is for **users** of Discovery who build and sample timing-array
models. For internals (graph DSL, house rules, parity suite, porting guidance)
see [Metamatrix developer guide](metamatrix_dev).

## What metamatrix is

A PTA likelihood evaluates quantities like

$$
\log L(\theta) = -\tfrac12\, y^\top \Sigma(\theta)^{-1} y - \tfrac12\log\det\Sigma(\theta)
$$

with $\Sigma = N + F\,P\,F^\top$ (white noise $N$, design/basis $F$, GP prior
$P$). Historically Discovery implemented every fixed/variable combination of
$N$, $F$, and $P$ as separate classes in `matrix.py`. **Metamatrix** replaces
that with a small **graph** representation: the same math is written once;
constant pieces fold away at construction time and parameter-dependent pieces
stay live.

You almost never touch the graph layer directly. User-facing factories
(`makegp_*`, `makenoise_measurement`, recipes, likelihoods) build models the
same way as before. The default backend is the graph path (**metamath**).

## Defaults and the kernel switch

At import time Discovery selects the graph backend:

```python
import discovery as ds
assert ds.config() == "metamath"   # default
```

To force the legacy closure path (reference implementation, still maintained
for comparison and rollback):

```python
ds.config(kernels="matrix")
# build models *after* this call
```

To switch back:

```python
ds.config(kernels="metamath")
```

Notes:

- Call `ds.config(...)` **before** constructing likelihoods. Already-imported
  class objects are not retroactively rebound.
- This is independent of `discovery.utils.config(backend=...)` (numpy vs JAX
  numerics).
- Both backends remain available so you can compare results or recover known
  matrix-path behaviour. The long-term plan is a single backend; until then
  both are first-class.

## Building models (unchanged surface)

Typical single-pulsar and array construction is unchanged:

```python
import discovery as ds
import discovery.recipes as R

# Recipe zoo (same objects the tests and cookbook exercise)
pl = R.intrinsic_rn(psr)                 # PulsarLikelihood
al = R.intrinsic_rn_plus_global_hd(psrs) # ArrayLikelihood

logL = al.logL
print(float(logL(params)))
```

Or hand-assembled:

```python
noise = ds.makenoise_measurement(psr, noisedict)
rn = ds.makegp_fourier(psr, ds.powerlaw, nmodes, Tspan=T)
pl = ds.PulsarLikelihood([psr.residuals, noise, rn])
```

Factories (`signals`, `measurement_noise`, `deterministic`) are backend-agnostic:
they go through an internal kernel factory so the same recipe builds matrix or
metamath objects depending on `ds.config()`.

See also:

- [Model cookbook](tutorials/cookbook_models) — runnable gallery of tested models
- [CW ExtSignal tutorial](tutorials/cw_extsignal_example)
- [Single-precision tutorial](advanced/single_precision)
- [Model summary guide](guide/model_summary)

## Likelihood frontends

### Marginal likelihood: `logL`

Integrates GP coefficients analytically (Woodbury). Use this for standard PTA
noise analysis (red noise, DM, HD common process, …).

```python
logL = al.logL
value = logL(params)          # scalar
names = logL.params           # parameter names this callable reads
```

### Coefficient likelihood: `clogL`

Samples GP coefficients jointly with hyperparameters. Needed for decentered
parameterizations, continuous-wave ExtSignals, and any analysis that keeps
coefficients as free parameters.

```python
clogL = al.clogL
value = clogL({**hyperparams, **coeff_params})
```

For a decentered cross-form array, an opt-in packed path evaluates the
same density from one hyperparameter vector `theta` and one coefficient
array `xi`. See [Packed coefficient likelihood](advanced/packed_clogl.md).

```python
packed = al.make_packed_clogL()
value, c = packed(theta, xi)
```

#### Choosing the algebra: `clogl_form`

`ArrayLikelihood` accepts:

```python
al = ds.ArrayLikelihood(
    psls,
    commongp=commongp,
    clogl_form="auto",   # default
)
```

| Value | Behaviour |
|---|---|
| `"cross"` | Historical form: forms $F^\top N^{-1} F$ per pulsar, then quadratic in coefficients. |
| `"residual"` | Forms the full residual $r = y - Fc - \sum_e F_{\mathrm{ext},e}c_e$ and applies one $N^{-1}$ solve to $r$. |
| `"auto"` | Uses `"residual"` if any per-pulsar noise solve has free parameters; otherwise `"cross"`. |

Inspect the resolved choice:

```python
al.clogl_form_resolved   # "cross" or "residual"
```

**When it matters.** With fixed white noise, both forms agree (products fold at
construction). With free EFAC/EQUAD/ECORR, the residual form avoids pushing
$(n_{\mathrm{toa}}\times k)$ matrices through a noise solve every evaluation and
is usually much faster for large $k$.

**Known issue with ≥2 ExtSignals.** The cross form currently omits
*inter*-ExtSignal cross-terms when two or more non-orthogonal ExtSignals are
present; the residual form is correct. Because `"auto"` may pick different
forms depending on whether white noise is free, multi-ExtSignal models should
prefer:

```python
clogl_form="residual"
```

Tracked upstream: [nanograv/discovery#137](https://github.com/nanograv/discovery/issues/137).

### Conditional distributions

```python
mean, cov_factor = pl.conditional(params)   # GP coefficient conditional
# sample_conditional when available on the route
```

Variable-GP timing reconstruction (`makegp_timing(..., variable=True)`) works
on both backends.

## Sampled vs marginalized GPs

With several variable GPs on one pulsar:

```python
# Concatenate bases (one joint coefficient block) — usually what you want
pl = ds.PulsarLikelihood([y, noise, gp1, gp2], concat=True)

# Chain Woodburys; only the *last* variable GP keeps free coefficients unless
# you opt into the old accidental behaviour:
pl = ds.PulsarLikelihood(
    [y, noise, gp1, gp2],
    concat=False,
    marginalize_all_but_last=True,  # required when ≥2 variable GPs
)
```

Without `marginalize_all_but_last=True`, construction raises: earlier variable
GPs would be silently marginalized by index overwrite. That is intentional —
sampled-vs-marginalized treatment is explicit.

`model.summary()` reports a **`coefficients`** column describing how each block
appears to the *coefficient* frontend (`sampled (k)`, `marginalized`,
`projected`, …), derived from the assembled kernel’s index — not from GP type
alone. See [Model summary](guide/model_summary).

## Decentering and transport

Decentering reparameterizes GP coefficients so the sampler walks a better-conditioned
space. On the graph backend this is a free-standing object, not an opaque closure.

### Simple sugar

```python
al = ds.ArrayLikelihood(
    psls,
    commongp=commongp,
    globalgp=globalgp,   # optional; uses CURN-style conditioner view
    decenter=True,
)
```

Requirements for `decenter=True`:

- A `commongp` must be present.
- White noise must be **fixed** at construction (frozen reference noise). If EFAC
  etc. are free, construction raises with a clear message — build an explicit
  `Transport` instead (below).
- Mutually exclusive with `transport=...`.

When `extsignals=` is also set, `decenter=True` subtracts those deterministic
signals from the centering residual (the Fourier coefficients are centered on
the ExtSignal-subtracted data). The Jacobian does not depend on ExtSignal
parameters. An explicit `transport=` is caller-owned and is not rewritten
from `extsignals`. `transport=` is metamath-only (`likelihood.py` has no
such argument).

### Explicit transport

```python
from discovery import transport as tr

npsr = len(psls)
per = []
for i, psl in enumerate(psls):
    blocks = [tr.gp_block(commongp, psr_slot=i)]
    if globalgp is not None:
        blocks.append(tr.globalgp_curn_block(globalgp, i, npsr))
    per.append(tr.Transport(
        blocks,
        reference_noise=tr.reference_noise_frozen(psl.N, params0={}),
        reference_residual=psl.y,
        center=True,
        center_extsignals=extsignals,   # same list on every pulsar
        psr_slot=i,
    ))
t = tr.ArrayTransport(per)
al = ds.ArrayLikelihood(psls, commongp=commongp, transport=t,
                        extsignals=extsignals)
```

`ArrayTransport` batches ExtSignal centering when every per-pulsar `Transport`
was built with the same `center_extsignals` list and `psr_slot=i`. It still
rejects `softclip`. Explicit `transport=` is metamath-only.

### Varying white noise: class-tracked reference

`reference_noise_frozen` pins the white noise the chart whitens against. When
EFAC/EQUAD/ECORR are sampled, a frozen chart gives the sampler a funnel
(condition numbers of 10²–10⁵ on IPTA-style pulsars). `class_tracking` bakes
the same reference *and* the exact Gram of a class-quantized white-noise
model that follows the white-noise parameters: EFAC exactly, ECORR exactly,
small backends exactly, EQUAD elsewhere to the spread of TOA errors inside
0.2-dex classes. Cost: tens of `k×k` matrices plus one `(n_dense × k)` and one
`(n_epoch × k)` contraction per step — a few ms per pulsar on CPU, against
tens of ms for the exact Gram.

```python
ref = tr.class_tracking(psl.white_noise_kernel, params0=noisedict, toaerrs=psr.toaerrs)
t = tr.Transport(blocks, reference_noise=ref, reference_residual=psl.y, center=True)
```

or, for the sugar, `ds.ArrayLikelihood(psls, commongp=..., decenter=True,
decenter_params0=noisedict)`. ECORR may be given either as
`makenoise_measurement(..., ecorr=True)` or as a fixed `makegp_ecorr`
component; both are folded into the same Sherman–Morrison form
(`PulsarLikelihood.white_noise_kernel`). Free ECORR hyperparameters need the
former; a `makegp_ecorr` without a noisedict is a sampled-amplitude GP in
`clogL` and is a transport *block*, not white noise. `params0` must be a
sensible white-noise point (empirical-Bayes dictionary, MPE), not `toaerrs`:
the chart is exact there and degrades smoothly away from it.
`transport.diagnostics(params, noise_solve=...)` reports the whitened metric
eigenvalues against a live solve; `diagnostics()["tracking"]` reports the
class/dense-row counts and the bake-point digest. The chart is a coordinate
choice: it never changes the posterior, only the sampler's efficiency.
`make_packed_clogL` requires frozen noise and refuses a tracked model.

Helpers:

| API | Role |
|---|---|
| `tr.gp_block(gp, psr_slot)` | Diagonal GP prior → exact conditioner precision |
| `tr.globalgp_curn_block(globalgp, psr_slot, npsr)` | Dense global prior → per-pulsar inverse-marginal-variance conditioner |
| `tr.array_block(F, index, conditioner_precision, name=...)` | Caller-owned basis (e.g. timing); precision is mandatory |
| `tr.reference_noise(psr)` / `tr.reference_noise_frozen(kernel, params0)` | Freeze $N_0$ for the transport bake |
| `tr.class_tracking(kernel, params0, toaerrs=...)` | White-noise-tracking reference for sampled EFAC/EQUAD/ECORR (`PulsarLikelihood.white_noise_kernel`) |
| `Transport` / `ArrayTransport` | Map $\xi\mapsto q$ with log Jacobian |
| `MarginalTransport` / `marginal_transport(...)` | Live-kernel decentering of one external block against marginalized $C(\eta)$ |
| `t.fingerprint()` | Stable structural digest (for run manifests) |
| `t.reference_noise_quadratic(v)` / `t.reference_noise_standard_deviation()` | Frozen $N_0$ probes |

Failure semantics (user-visible):

- Construction and `validate(params)` raise on bad shapes / non-PD conditioners.
- Runtime `apply` under JAX is NaN-propagating (no silent floors on prior precision).

Recipes that wrap common decenter patterns live in `discovery.recipes`
(`decenter_intrinsic_rn`, `decenter_intrinsic_rn_global_hd`,
`decenter_extsignal_cw`, `decenter_extsignal_cw_global_hd`).

## Single precision (float32) and reference+delta

GPUs prefer `float32`, but PTA log-likelihoods subtract large numbers. Discovery
offers opt-in mitigations that leave the default `float64` path unchanged:

1. **Final combine in float64** — expensive work can run in float32; the scalar
   assembly of quadratic and log-det pieces is promoted.
2. **Reference + delta** — freeze GP prior covariances at a reference and evaluate
   $\ln L = \ln L_{\mathrm{ref}} + \Delta\ln L$ so float32 only holds an $O(1)$
   increment:

   ```python
   al = ds.ArrayLikelihood(..., reference=params_ref)
   ```

3. **Timing-model projection** — `makegp_timing(..., project=True)` uses an exact
   flat-prior projection instead of a huge-variance improper prior (float32-safe).

4. **Baked constants stay float64** — the transport (`decenter=True` /
   `transport=`) bakes $G_0 = W^T N_0^{-1} W$, $b_0$ and any ExtSignal $E_0$
   once at construction, and takes conditioner precisions $\phi^{-1}$, in
   float64 whenever x64 is on (`transport.bake_dtype()`), independent of
   `config(working=float32)`. A float32 solve through the timing-model
   Woodbury leaves $G_0$ indefinite at the $10^{-5}$ level (worse under TF32
   GPU matmul), so $G_0 + \mathrm{diag}(\phi^{-1})$ goes indefinite for legal
   hyperparameters and NUTS sees NaN log-densities; a float32 $\phi^{-1}$ is
   also quantized and has an overflowing second derivative. Construction now
   raises if the baked Gram is materially indefinite. Boundary code can request
   the same via `metamatrix.func(graph, working=jnp.float64)`.

Details and numbers: [Single-precision tutorial](advanced/single_precision).
Design notes (ADRs, research math): [docs/design/single_precision/](design/single_precision/README).
Limitations worth knowing:

- Fused reference+delta for the HD (global) path is the best-supported case.
- Single-level (CURN/IRN-only) refdelta routing is more limited; see the developer
  guide if you hit missing twins.

## Extra signal factories useful with the graph path

### Unit-normal coefficient GP

```python
gp = ds.makegp_standard_normal(psr, F)   # c ~ N(0, I), proper prior
```

Unlike `makegp_improper` (huge constant variance), this retains a real log
determinant and does not project or renormalize columns.

### Pivot-amplitude power law

Sample amplitude at a sensitivity-weighted pivot frequency instead of $1/\mathrm{yr}$:

```python
from discovery.signals import (
    make_powerlaw_pivot,
    sensitivity_weighted_pivot_frequency,
    fourier_sensitivity_weights,
    reference_log10_amplitude,
)

# f_pivot from frozen noise + Fourier basis, or pass log10_f_pivot explicitly
psd = make_powerlaw_pivot(log10_f_pivot=..., components=nmodes)
gp = ds.makegp_fourier(psr, psd, nmodes, Tspan=T)
```

Public amplitude name is `log10_A_pivot`. Decode amplitude at $1/\mathrm{yr}$ with
`reference_log10_amplitude`.

## Inspecting models

```python
print(pl.summary())           # table including coefficients column
# kernel tree / signal reprs available via summary helpers
```

## Recipes and tests as documentation

`discovery.recipes` is the importable model zoo used by:

- the [cookbook](tutorials/cookbook_models),
- `tests/metamatrix/` parity tests (matrix vs metamath).

If a recipe builds and the parity suite passes for that topology, the graph path
is certified for that topology.

## Known limitations (user-facing)

| Topic | Status |
|---|---|
| Dual backends | Both `matrix` and `metamath` selectable; default `metamath`. |
| Multi-ExtSignal `clogL` cross form | Incomplete cross-terms; use `clogl_form="residual"` ([#137](https://github.com/nanograv/discovery/issues/137)). |
| `cglogL` | Not currently runnable on either backend (missing optional deps + incomplete API). Do not rely on it. |
| Chained multi-GP `clogL` on matrix route | Builds with `marginalize_all_but_last=True` but coefficient likelihood evaluates reliably only under `kernels='metamath'`. |
| Ragged array transports | Not supported; all pulsars in an `ArrayTransport` must share coefficient dimension. |
| Pickling transports | Not supported as a stable API; use `fingerprint()` for identity in run metadata. |

## Quick reference

```python
import discovery as ds

ds.config()                          # 'metamath'
ds.config(kernels='matrix')          # legacy path
ds.config(kernels='metamath')        # graph path (default)

al = ds.ArrayLikelihood(
    psls,
    commongp=commongp,
    globalgp=globalgp,
    decenter=False,
    transport=None,
    clogl_form="auto",
    reference=None,                  # optional float32 ref+delta
    extsignals=None,
)

al.logL / al.clogL / al.conditional
al.clogl_form_resolved
pl.summary()
```

For architecture, parity design, graph house rules, and how to extend metamath,
continue with [Metamatrix developer guide](metamatrix_dev).
