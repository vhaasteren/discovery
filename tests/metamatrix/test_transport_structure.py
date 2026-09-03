"""Structural invariants of the transport: wiring, not numerics.

The numerical parity tests in ``test_transport.py`` build transports over
7,758-19,571-residual pulsars, factor Choleskys and differentiate through
JAX. They answer "is the arithmetic right?", which is a different and much
more expensive question than "is the object wired together correctly?".

A rename, a refactor, or a signature change can only break the second one,
and this module answers it on a synthetic 40-TOA fixture with no real data,
no gradients and no Cholesky worth the name -- in well under a second. It is
the first thing to run after any structural edit.

Every assertion here corresponds to a way the object has actually been broken:
a resolved keyword silently rebound to an integer by a same-named local, a
``dimension`` that stopped being an int, and a ``_b0`` that went ``None``
because a stale attribute read ``False``. None of those raised; all of them
produced plausible numbers. These checks make them loud.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import discovery as ds  # noqa: E402
from discovery import transport as tr  # noqa: E402

N_TOA = 40


@pytest.fixture(autouse=True)
def _metamath():
    """Set and restore the kernel mode.

    ``ds.config(kernels=...)`` is process-global. A test that sets it and
    leaves it set makes every later test order-dependent, which is how a
    suite acquires failures that vanish when run alone.
    """
    previous = ds._kernels.get_mode() if hasattr(ds, "_kernels") else None
    ds.config(kernels="metamath")
    yield
    if previous is not None:
        ds.config(kernels=previous)


class _UnitRef:
    """Reference noise N0 = I: the cheapest thing that satisfies the seam."""

    description = "unit"
    params = ()

    def solve(self, W):
        return np.asarray(W, dtype=float), None


def _blocks(ks=(3, 2)):
    rng = np.random.default_rng(0)
    return [
        tr.array_block(
            rng.standard_normal((N_TOA, k)),
            {f"c{i}": slice(0, k)},
            1.0,
            name=f"b{i}",
        )
        for i, k in enumerate(ks)
    ]


def _transport(origin="conditional_mode", ks=(3, 2)):
    rng = np.random.default_rng(1)
    return tr.Transport(
        _blocks(ks),
        reference_noise=_UnitRef(),
        reference_residual=rng.standard_normal(N_TOA),
        origin=origin,
    )


# --- the resolved keyword survives construction ----------------------------


@pytest.mark.parametrize("origin", ["conditional_mode", "zero"])
def test_origin_is_stored_as_the_resolved_string(origin):
    """The single check that catches a same-named local clobbering the kwarg."""
    t = _transport(origin)
    assert isinstance(t.origin, str), f"origin is {type(t.origin).__name__}"
    assert t.origin in tr.ORIGINS
    assert t.origin == origin


def test_unknown_origin_is_refused_at_the_boundary():
    with pytest.raises(ValueError, match="origin must be one of"):
        _transport("middle")


def test_center_keyword_is_gone():
    """No alias. Two names for one concept in one function body is how the
    resolved value got shadowed in the first place."""
    with pytest.raises(TypeError):
        tr.Transport(
            _blocks(), reference_noise=_UnitRef(),
            reference_residual=np.zeros(N_TOA), center=True,
        )
    assert not hasattr(_transport(), "center")


# --- dimensions and the block index ----------------------------------------


def test_dimension_is_an_int_equal_to_the_column_count():
    """`dimension` is the block-column cursor; it must not pick up a keyword."""
    t = _transport(ks=(3, 2, 4))
    assert isinstance(t.dimension, (int, np.integer)), type(t.dimension)
    assert t.dimension == 9


def test_block_index_slices_are_contiguous_and_ordered():
    t = _transport(ks=(3, 2, 4))
    spans = [t.index[f"c{i}"] for i in range(3)]
    assert [(s.start, s.stop) for s in spans] == [(0, 3), (3, 5), (5, 9)]


# --- the origin actually reaches the bake ----------------------------------


def test_mode_origin_bakes_a_translation_and_zero_does_not():
    """`_b0` is the whole point of the origin; a stale read made it None."""
    assert _transport("conditional_mode")._b0 is not None
    assert _transport("zero")._b0 is None


def test_mode_origin_requires_a_reference_residual():
    with pytest.raises(ValueError, match="requires reference_residual"):
        tr.Transport(_blocks(), reference_noise=_UnitRef(),
                     origin="conditional_mode")


# --- the serialized surface -------------------------------------------------


def test_diagnostics_reports_the_origin_and_no_stale_key():
    d = _transport().diagnostics()
    assert d["origin"] == "conditional_mode"
    assert "center" not in d
    assert d["dimension"] == 5
    assert [b["name"] for b in d["blocks"]] == ["b0", "b1"]


def test_fingerprint_is_stable_and_distinguishes_origins():
    assert _transport().fingerprint() == _transport().fingerprint()
    assert _transport("conditional_mode").fingerprint() != _transport("zero").fingerprint()


# --- ArrayTransport aggregates the origin -----------------------------------


def test_array_transport_takes_the_common_origin():
    at = tr.ArrayTransport([_transport(), _transport()])
    assert at.origin == "conditional_mode"
    assert at._b0 is not None
    assert not hasattr(at, "center")


def test_array_transport_refuses_a_mixed_origin():
    with pytest.raises(ValueError, match="all-or-none origin"):
        tr.ArrayTransport([_transport("conditional_mode"), _transport("zero")])
