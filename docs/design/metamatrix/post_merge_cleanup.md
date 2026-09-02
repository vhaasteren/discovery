# Post-merge cleanup (Phase 3)

Guarded-but-not-deleted surfaces from the dual-path merge PR. Removals happen
in one reviewable, almost-all-red PR on `main` after the merge. No behavior
change except where a guard already covers the API (keep the
`NotImplementedError` as the method body) or the listed typed-error tweak.

Gate: full suite green, `git diff --stat` net-negative, zero changes under
`matrix.py` / `likelihood.py`.

## Likelihood frontends — keep the guard, delete the dead body

| Item | Anchor |
|---|---|
| `cglogL` body | `likelihood_metamath.py` (`ArrayLikelihood.cglogL`) |
| `plogL` globalgp branch | `likelihood_metamath.py` (`GlobalLikelihood.plogL`) |
| Two unreachable `make_kernelterms` else-branches | `GlobalLikelihood.logL`, `ArrayLikelihood.logL` |

These raise `NotImplementedError` pointing here. After cleanup the guard *is*
the method body (~254 lines of dead code under the raise go away).

## Delete outright (not likelihood frontends — no guards)

| Item | Anchor |
|---|---|
| `woodburyfast`, `noiseallsolve`, Italian-comment derivation block | `metamath.py` |
| `stacksolve` global + branch | `metamath.py` |
| Unreachable `globalwoodbury` else (≈298–311) | `metamath.py` |
| Unreachable `GlobalWoodburyKernel` else (≈1316–1318) | `metamath.py` |
| Unused `make_joint_solve`, `vectorwoodburysolve` + `VectorWoodburyKernel.make_solve` | `metamath.py` |
| Dead `lP` / `lS` / `Nmy` unpackings (≈209/218/252/264/265/587) | `metamath.py` |
| `sample_graph` (`func(graph, jit=False)` → TypeError) | `metamatrix.py` |
| `GraphBuilder.eval` (unreachable string-eval, mis-ordered zip) | `metamatrix.py` |
| `GraphBuilder.solve` (NameError — `Sym.solve` is the real API) | `metamatrix.py` |
| Commented fold block (≈232–262), `from unicodedata import name`, `import ast` | `metamatrix.py` |
| Dead `apply_patches` / `restore_patches`; stale "used by `ds.config`" docstring | `_kernel_switch.py` |

## Move to tests/

| Item | From |
|---|---|
| `dense_coefficient_logprior_legacy` | `metamath.py:1450` |
| `_legacy_globalgp_curn_precision` | `transport.py:274` |

## Small behavior change — call out in the PR body

`Sym.__radd__` / `Sym.__rmul__` numpy-operand fallback currently raises
`AttributeError: 'ndarray' has no 'name'`. Replace with a typed `TypeError`
naming the operand (`metamatrix.py` ≈719–729).
