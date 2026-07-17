# BUG: `clogL` cross form omits inter-ExtSignal cross-terms (≥2 ExtSignals)

**Severity:** MEDIUM (narrow: affects only ≥2 non-orthogonal ExtSignals; usually
numerically swamped in practice)
**Status:** Open — not fixed. Documented here for a maintainer decision.
**Found:** during the PR1–PR6 review of `feature_metamatrix_cleanup.md`.
**Not introduced by that feature** — the defect is in the pre-existing cross-form
graph; the new residual form is the one that is *correct*.

## Summary

`ArrayLikelihood.clogL` has two algebraically-intended-identical forms
(`clogl_form="cross"` vs `"residual"`, auto-selected). They are **not** identical
when **two or more `ExtSignal`s with non-orthogonal bases** are supplied: the
cross form silently drops the cross-terms *between* distinct ExtSignals, while
the residual form includes them. The residual form is the mathematically correct
joint log-likelihood; the cross form has a latent omission.

Because `clogl_form="auto"` routes to `residual` when any per-pulsar noise solve
has free parameters and to `cross` otherwise, the **same model can yield
different `clogL` values depending only on whether its white noise is fixed or
free** — an internal inconsistency.

## The two forms

For a data model `y = F c + Σ_e Fext_e ccw_e + noise`, the exact joint
log-likelihood at fixed coefficients uses the full residual

```
r = y − F c − Σ_e Fext_e ccw_e
logp ⊃ −½ rᵀ N⁻¹ r
```

Expanding `−½ rᵀ N⁻¹ r` produces, among others, the cross-terms

```
− Σ_{e≠e'}  ccw_eᵀ Fext_eᵀ N⁻¹ Fext_{e'} ccw_{e'}      (inter-ExtSignal)
```

- **`vectorresidualcomponent`** (`src/discovery/metamath.py:860`, added by PR4)
  forms `r` and does one `N⁻¹` solve, so it **includes** these terms. Correct.
- **`vectorgpcomponent`** (`src/discovery/metamath.py:784`, pre-existing) loops
  over ExtSignals independently (`for ccw, Fcw_list in zip(...)` at
  `metamath.py:841`), adding only each ExtSignal's own data term, its cross with
  the GP coefficients `c`, and its self term. It has **no `e≠e'` term**, so it
  **omits** the inter-ExtSignal cross-terms.

## Reproduction

Independently confirmed two ways during review:

1. **Code analysis** (definitive): the cross-form loop at `metamath.py:841-857`
   contains no term coupling ExtSignal `e` to ExtSignal `e'`; the residual form
   at `metamath.py:891-898` subtracts all ExtSignals into one `r` before the
   solve, so it does.
2. **Numerical** (synthetic, 2 ExtSignals with non-orthogonal bases and nonzero
   coefficients): `cross` and `residual` differ by exactly
   `−w₁ᵀ G₁ᵀ N⁻¹ G₂ w₂`.

Note: in a *realistic* model the difference is often invisible. With an improper
timing prior the `clogL` scale is ~1e18, so an O(1)…O(1e2) inter-ExtSignal
cross-term falls below the float64 rounding floor (~1e18·1e-16 ≈ 1e2) and the two
forms compare equal. The bug is real but its practical magnitude depends on the
model's dynamic range and on how non-orthogonal the ExtSignal bases are under
`N⁻¹`.

## Impact / scope

- **0 or 1 ExtSignal:** no effect — the two forms are exact (there is no `e≠e'`
  pair). This is the overwhelmingly common configuration (a single
  `makecw_extsignal` CW).
- **≥2 ExtSignals:** `auto` routing can give different `clogL` between the
  fixed-WN (`cross`) and free-WN (`residual`) configurations of the same model.
  For a multi-resolvable-source analysis (e.g. two CWs) the sampled posterior
  would differ between the two routes.

## Affected code

- `src/discovery/metamath.py:784` `vectorgpcomponent` — the ExtSignal loop at
  `:841-857` omits inter-ExtSignal cross-terms.
- `src/discovery/metamath.py:860` `vectorresidualcomponent` — correct; includes
  them.
- `src/discovery/likelihood_metamath.py` `ArrayLikelihood.clogL` /
  `clogl_form_resolved` — the `auto` router that can switch between the two.

## Recommended fix (maintainer decision)

Make the cross form agree with the (correct) residual form by adding the
inter-ExtSignal cross-terms to `vectorgpcomponent`: for each unordered pair
`(e, e')`, subtract `ccw_eᵀ (Fext_eᵀ N⁻¹ Fext_{e'}) ccw_{e'}`. This restores
`cross == residual` for all ExtSignal counts.

Caveats for whoever takes this:
- `vectorgpcomponent` is on the **certified matrix-vs-metamath parity path**;
  changing it must be re-checked against that suite. (The legacy matrix cross
  form has the same omission, so a true fix touches both, or the parity oracle
  must be updated to the corrected value.)
- The spec's blanket "algebraically identical" claim (§4, §4.6 test 2) and the
  single-ExtSignal coverage of `tests/metamatrix/test_residual_clogl.py::
  test_form_equivalence_with_extsignal` should be extended to a ≥2-ExtSignal
  case once fixed, to lock the equivalence.

Until fixed, prefer `clogl_form="residual"` for any analysis with ≥2 ExtSignals.
