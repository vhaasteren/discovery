# Packed coefficient likelihood

The public `ArrayLikelihood.clogL` is a named dictionary: every hyperparameter
and every per-pulsar coefficient vector is its own key. That is the general
path, and it remains the source of truth.

For decentered, frozen-noise (cross-form) array models, an opt-in **packed**
path evaluates the *same* log-density from two arrays:

- `theta` — one flat hyperparameter vector (`discovery.params.Params`)
- `xi` — one rectangular coefficient array of shape `(npsr, k)`

JAX then sees two dynamic leaves instead of hundreds. The numerical value
matches `model.clogL` (checked by `PackedClogL.oracle`). Nothing about the
likelihood algebra changes.

## When to use it

Use the packed path when you are sampling or differentiating a decentered
`clogL` and want one coefficient buffer plus one hyperparameter vector.
Keep `model.clogL` for inspection, residual-form models, live white noise,
and anything the eligibility check rejects.

```python
packed = model.make_packed_clogL()
value, physical_coefficients = packed(theta, xi)
```

`physical_coefficients` is the decentered coefficient array `c` of shape
`(npsr, k)`, the same object the named path returns as the staged
coefficient block.

Convert a named dict either way:

```python
theta, xi = packed.pack(params)
params = packed.unpack(theta, xi)
```

## How evaluation works

`packed(theta, xi)` unpacks to the named dict and calls `model.clogL`.
`oracle` is the same path. The numerical value is identical by
construction; the difference is the JIT ABI (two leaves instead of one
named dict).

The named graph already uses the structured prior and batched
conditioner internally. Packing does not change that algebra.

## NumPyro

```python
from discovery.samplers import numpyro as ds_numpyro

numpyro_model = ds_numpyro.makemodel_packed(model, priordict=PRIORDICT)
```

Sites are a Uniform `theta` vector and a standard-normal `xi` array.
`clogL` is already the transformed joint, so the Normal quadratic is
cancelled with a mandatory `xi_base_correction` factor
(`+0.5 * sum(xi * xi)`). Without that cancellation the target would
count \(\exp(-\xi^2/2)\) twice.

Convert a chain with `numpyro_model.to_df(samples)` (or
`packed.samples_to_df`). Columns are named scalars, including
per-component coefficient fields.

## Eligibility

`make_packed_clogL()` raises `PackedClogLUnsupported` rather than falling
back silently. Inspect first:

```python
from discovery.packed import packed_clogl_diagnostics

packed_clogl_diagnostics(model)
# {"eligible": True, "form": "cross", "xi_shape": (67, 88), ...}
# or {"eligible": False, "reasons": [...]}
```

v1 requires all of:

- metamath `ArrayLikelihood` with a single diagonal Fourier `commongp`;
- resolved `clogl_form == "cross"` (frozen measurement noise);
- a centered `ArrayTransport` (`decenter=True` or an explicit transport);
- equal per-pulsar coefficient row widths;
- no class-tracked white noise (`decenter_params0` / `class_tracking`);
- no user transform after the transport, no `reference` / `refdelta`,
  no live pulsar delays, no GP prior means;
- if a `globalgp` is present, it must carry a separable Fourier prior
  (one ORF × diagonal spectrum). Non-SPD or asymmetric ORFs stay on
  the dense named path;
- ExtSignals, if any, must have a fixed per-pulsar basis (no live /
  evolving TOA-space CW).

Residual form, free EFAC/EQUAD/ECORR, class-tracked white noise, and
live-basis deterministic signals stay on `model.clogL`.

## Layout

`theta` uses `discovery.params.Params` / `make_layout`. Pass
`template_params` to `make_packed_clogL` when an unsuffixed name is an
array rather than a scalar.

`xi` rows follow `Transport.index` on each pulsar, concatenated in that
local order. Do not index `xi` with source-GP slices from the original
signal objects; those need not match the assembled row.

```python
packed.theta_names     # hyperparameter names in layout order
packed.xi_names        # per-row coefficient field names
packed.xi_shape        # (npsr, k)
packed.diagnostics()
```

## Related

- Named `clogL` and `clogl_form`: [Metamatrix user guide](../metamatrix.md)
- `Params` single-leaf container: `discovery.params`
- Decentering recipes: `discovery.recipes.decenter_intrinsic_rn`,
  `decenter_intrinsic_rn_global_hd`, `decenter_extsignal_cw_global_hd`
