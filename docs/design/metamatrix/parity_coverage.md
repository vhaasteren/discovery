# Parity coverage (matrix ↔ metamath)

Historical coverage audit: every kernel constructor that production code or
the example/tutorial notebooks emit should be exercised by a parity route
(or documented as out of scope). Living enforcement is `tests/metamatrix/`
and `discovery.recipes`.

Scope of "used": `signals.py` / `measurement_noise.py` / `deterministic.py`
call sites, plus the builders invoked by `examples/*.ipynb` and
`docs/tutorials/*.ipynb` (incl. `cw_extsignal_example.ipynb`).

## Coverage table

Column meanings:
- **Emitted by** — the `signals.py` / `measurement_noise.py` builder whose call
  constructs this kernel.
- **metamath-mapped?** — whether the `_kernels` factory resolves the constructor
  to a metamath class in metamath mode (`_METAMATH` entry), or it's a collapsed
  factory function that dispatches directly.
- **Covering model (test builder)** — the model-builder function in
  `tests/metamatrix/test_*.py` that assembles a model exercising this
  constructor (on real pulsar data). Each such model is run through **all three
  routes** in `_routes.build_routes` — `matrix` (oracle), `mh_patched`,
  `mh_native` — and their logL / conditional / clogL are asserted equal. That
  three-way equality is the "parity"; the builder is just the vehicle.

| Constructor | Emitted by | metamath-mapped? | Covering model (test builder) | Status |
|---|---|---|---|---|
| `NoiseMatrix1D` (collapsed) | `makenoise_measurement*` | factory fn | `measurement_white`, `measurement_simple` | ✅ |
| `NoiseMatrixSM` (collapsed) | `makenoise_measurement(ecorr=True)` | factory fn | `ecorr_sm+timing` | ✅ |
| `NoiseMatrix1D_novar`/`_var` | `makegp_ecorr*`, `makegp_timing`, `makegp_improper` | ✅ | `measurement+ecorr_gp`, `…timing_svd`, `variable_timing` | ✅ |
| `NoiseMatrix12D_var` (1D) | `makegp_fourier` (1D PSD) | ✅→`NoiseMatrix12D` | `full_rn`, `multi_vgp` | ✅ |
| `NoiseMatrix2D_var` (2D) | `makegp_fftcov`/`avgcov`/`intcov` | ✅ | **`fftcov_2d` (added)** | ✅ closed |
| `NoiseMatrix2D_novar` | `makegp_fourier_variance` (fixed) | ✅ (added) | **`fourier_variance_fixed`** | ✅ closed (Phase 4a) |
| `VectorNoiseMatrix12D_var` | `makecommongp_fourier` | ✅ | `intrinsic_rn`, `decenter+intrinsic_rn` | ✅ (closed Phase 1) |
| `CompoundDelay` | `makedelay`/`makedelay_binary` | ✅ | **`delay` (added)** | ✅ closed |
| `WoodburyKernel`, `CompoundGP`, `VectorCompoundGP`, `VectorWoodburyKernel_varP` | likelihood layer | ✅ | all model builders | ✅ |
| `CompoundGlobalGP` | globalgp-as-list (HD+monopole) | ✅ (Phase 4b) | **`global_compound`** | ✅ closed (Phase 4b) |
| `ExtSignal` (CW) | `makecw_extsignal` | ✅ (marker in `utils`) | `extsignal_cw`, `decenter_extsignal_cw`, `decenter_extsignal_cw_global_hd` (`test_clogL_new_features`) | ✅ |

PSD/ORF helpers (`powerlaw`, `freespectrum`, `brokenpowerlaw`,
`make_combined_crn`, `makepowerlaw_crn`, `hd_orf`, …) are prior functions, not
kernel constructors — they ride on the GP builders above.

## Gaps found and disposition

- **A — 2D variable noise (`NoiseMatrix2D_var`).** Used for real by
  `makegp_fftcov` in `os_example` / `numpyro_example`; no 1D-PSD parity row
  reached it. **Closed:** added `test_pulsar.py::_fftcov_2d` (passes matrix /
  mh_patched / mh_native).
- **B — delay (`CompoundDelay`).** Used by the CW examples via `makedelay`; no
  parity row touched a delay. **Closed:** added `test_pulsar.py::_delay`
  (parameter-free deterministic delay; passes all routes).
- **C — fixed 2D GP prior (`NoiseMatrix2D_novar`).** Emitted by
  `makegp_fourier_variance` when the variance matrix is supplied (no notebook
  uses this builder). The **kernel** maps (`NoiseMatrix2D_novar → mh.NoiseMatrix2D`),
  but the **likelihood path** is unsupported in metamath: an all-constant 2D GP
  prior reaches `metamath.CompoundGP._build_mixed_logprior`, which requires
  `gp.index` — a marginalized constant GP has none.
  **Closed:** `metamath.CompoundGP` now builds a real combined dense
  `Phi` (block-diagonal, promoting 1D blocks) for a mixed but *marginalized*
  compound, gating the coefficient-log-prior branch to the vector/decentered
  path. `test_pulsar.py::test_logL[fourier_variance_fixed]` passes (xfail removed).
- **D — `CompoundGlobalGP`.** The Phase-2 factory fallthrough (globalgp passed
  as a *list*, e.g. HD + monopole). Originally lived only in `matrix.py` and
  built `matrix.NoiseMatrix1D_var` directly, so it couldn't consume metamath GP
  priors. **Closed:** relocated to `signals.CompoundGlobalGP`,
  backend-agnostic (factory + `utils.GlobalVariableGP`, reading only
  mode-neutral `gp.Phi.getN`/`.getN.params`/`gp.Phi_inv`). Both `likelihood.py`
  and `likelihood_metamath.py` now call it; `matrix.CompoundGlobalGP` is dead
  code that goes with `matrix.py` when the legacy path is removed. Routed by `test_global.py`'s
  `global_compound` (logL + conditional).

## Integration check — `cw_extsignal_example.ipynb`

Pre-change CW+decenter numbers omitted; they assumed centering that did not
subtract the ExtSignal. See `test_clogL_new_features` ids
`decenter+extsignal_cw` and `decenter+extsignal_cw+global_hd`. (NUTS not run.)

## Result

All production / notebook-driven kernel constructors are now parity-routed.
The two carry-overs (C `NoiseMatrix2D_novar`, D `CompoundGlobalGP`) were closed
later — see the updated dispositions above; there are no remaining
`xfail`s. Every constructor the matrix oracle covers is now matched by the
metamath path, so the oracle may be deleted when the legacy path is removed.
