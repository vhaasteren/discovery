"""Packed ``clogL`` boundary: two arrays ``theta`` and ``xi``.

The named dictionary used by the general graph remains the source of truth
for diagnostics and the parity oracle. This module packs non-coefficient
parameters into one flat vector and coefficient inputs into one rectangular
``(npsr, k)`` array so the compiled kernel sees two dynamic buffers.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .params import Params, make_layout
from .transport import ArrayTransport
from .metamath import NoiseMatrix1D


def pack_with_layout(params, layout):
    """Concatenate named values in a fixed ``make_layout`` order."""
    from . import utils as kh

    blocks = []
    for name, start, stop, _shape in layout:
        value = kh.jnp.asarray(params[name]).reshape(-1)
        if value.size != stop - start:
            raise ValueError(
                f"{name!r} has flattened size {value.size}; "
                f"layout requires {stop - start}"
            )
        blocks.append(value)
    return kh.jnp.concatenate(blocks)


@dataclass(frozen=True)
class CoefficientArrayLayout:
    """Row-local coefficient slices for a rectangular ``(npsr, k)`` array."""

    rows: tuple
    shape: tuple

    @classmethod
    def build(cls, transports):
        rows = []
        row_widths = []
        for transport in transports:
            offset = 0
            row = []
            for name, local_slice in transport.index.items():
                if local_slice.start != offset:
                    raise ValueError(
                        f"non-contiguous transport index for {name!r}: "
                        f"expected start {offset}, got {local_slice.start}"
                    )
                row.append((name, local_slice))
                offset = local_slice.stop
            rows.append(tuple(row))
            row_widths.append(offset)
        if len(set(row_widths)) != 1:
            raise ValueError(
                f"packed clogL requires equal row widths; got {row_widths}"
            )
        return cls(tuple(rows), (len(rows), row_widths[0]))

    @property
    def names(self):
        return tuple(tuple(name for name, _ in row) for row in self.rows)

    def unpack(self, xi):
        if xi.shape != self.shape:
            raise ValueError(
                f"xi has shape {xi.shape}; expected {self.shape}"
            )
        return {
            name: xi[row_number, local_slice]
            for row_number, row in enumerate(self.rows)
            for name, local_slice in row
        }

    def pack(self, params):
        from . import utils as kh

        packed_rows = []
        for row in self.rows:
            blocks = []
            for name, part in row:
                value = kh.jnp.asarray(params[name]).reshape(-1)
                width = part.stop - part.start
                if value.size != width:
                    raise ValueError(
                        f"{name!r} has flattened size {value.size}; "
                        f"coefficient layout requires {width}"
                    )
                blocks.append(value)
            packed_rows.append(kh.jnp.concatenate(blocks))
        return kh.jnp.stack(packed_rows)


def _theta_names(model, transport, coefficient_layout):
    callables = [
        transport._conditioner_precision,
        model.commongp.Phi.getN,
    ]
    global_prior = getattr(model.globalgp, "separable_prior", None)
    if global_prior is not None:
        callables.append(global_prior.spectrum)
    callables.extend(ext.coeffs for ext in (model.extsignals or []))

    names = set().union(
        *(set(getattr(func, "params", ())) for func in callables)
    )
    coefficient_names = {
        name for row in coefficient_layout.rows for name, _slice in row
    }
    overlap = names & coefficient_names
    if overlap:
        raise ValueError(
            f"parameters used as both theta and xi: {sorted(overlap)}"
        )
    return tuple(sorted(names))


class PackedClogLUnsupported(ValueError):
    """Raised when a model does not satisfy the packed ``clogL`` contract."""


def packed_clogl_ineligibility(model):
    """Return every reason ``model`` cannot use the packed ``clogL`` path."""
    reasons = []

    if model.commongp is None:
        reasons.append("a commongp coefficient assembly is required")
    if getattr(model, "clogl_form_resolved", None) != "cross":
        reasons.append("clogL must resolve to cross form (frozen noise)")
    if not model.decenter and model.transport is None:
        reasons.append("an ArrayTransport/decenter transform is required")
    if model.transform is not None:
        reasons.append("user transforms after ArrayTransport are not supported")
    if model.reference is not None:
        reasons.append("reference/refdelta models are not supported")
    if any(getattr(psl, "delay", None) for psl in model.psls):
        reasons.append("live PulsarLikelihood delays are not supported")
    for extsignal in model.extsignals or []:
        if len(extsignal.Fs) != len(model.psls):
            reasons.append(
                f"ExtSignal {extsignal.name!r} basis count does not "
                "match the pulsars"
            )
        elif any(callable(F) or not hasattr(F, "shape")
                 for F in extsignal.Fs):
            reasons.append(
                f"ExtSignal {extsignal.name!r} has a live/non-array basis; "
                "v1 requires fixed per-pulsar bases"
            )
    if model.commongp is not None:
        if isinstance(model.commongp, (list, tuple)):
            reasons.append("v1 requires one diagonal Fourier commongp")
        elif not isinstance(model.commongp.Phi, NoiseMatrix1D):
            reasons.append("v1 requires a diagonal common-GP covariance")
        if getattr(model.commongp, "means", None) is not None:
            reasons.append("v1 does not support GP prior means")

    try:
        _vsm, ys = model._coefficient_assembly
        transport = (
            model.transport
            if model.transport is not None
            else model._build_decenter_transport(ys)
        )
        if not isinstance(transport, ArrayTransport):
            raise TypeError("the transform is not an ArrayTransport")
        layout = CoefficientArrayLayout.build(transport.transports)
    except (AttributeError, TypeError, ValueError) as error:
        reasons.append(f"coefficient layout is not rectangular: {error}")
    else:
        if layout.shape[0] != len(model.psls):
            reasons.append("coefficient rows do not match pulsar count")
        if not transport.center:
            reasons.append("the ArrayTransport must be centered")

    if model.globalgp is not None:
        if getattr(model.globalgp, "means", None) is not None:
            reasons.append("v1 does not support global GP prior means")
        if getattr(model.globalgp, "separable_prior", None) is None:
            reasons.append(
                "globalgp has no separable Fourier prior; use general clogL"
            )

    return reasons


class PackedClogL:
    """Opt-in packed evaluation of a decentered cross-form ``clogL``.

    Same log-density as ``model.clogL``, with two JIT inputs: a flat
    hyperparameter vector ``theta`` and a rectangular coefficient array
    ``xi`` of shape ``(npsr, k)``. ``__call__`` runs the fused kernel
    (frozen grams, no graph walk). ``oracle`` unpacks and calls
    ``model.clogL`` for parity.

    Raises ``PackedClogLUnsupported`` when the model is ineligible.
    User guide: ``docs/advanced/packed_clogl.md``.
    """

    def __init__(self, model, template_params=None):
        reasons = packed_clogl_ineligibility(model)
        if reasons:
            raise PackedClogLUnsupported(
                "packed clogL is unavailable:\n- " + "\n- ".join(reasons)
            )

        self.model = model
        self._named = None
        _vsm, ys = model._coefficient_assembly
        self.transport = (
            model.transport
            if model.transport is not None
            else model._build_decenter_transport(ys)
        )
        self.coefficients = CoefficientArrayLayout.build(
            self.transport.transports
        )
        theta_names = _theta_names(
            model, self.transport, self.coefficients
        )
        self.theta_layout, self.theta_size = make_layout(
            theta_names, template=template_params
        )
        from .fast_clogl import build_fused_clogl
        self.kernel = build_fused_clogl(
            model, self.transport, self.theta_layout, self.coefficients
        )

    @property
    def named(self):
        if self._named is None:
            self._named = self.model.clogL
            expected = set(self.theta_names) | {
                name
                for row in self.coefficients.rows
                for name, _slice in row
            }
            if set(self._named.params) != expected:
                raise ValueError(
                    "packed parameter discovery differs from model.clogL: "
                    f"missing={sorted(set(self._named.params) - expected)}, "
                    f"extra={sorted(expected - set(self._named.params))}"
                )
        return self._named

    @property
    def theta_names(self):
        return tuple(entry[0] for entry in self.theta_layout)

    @property
    def xi_names(self):
        return self.coefficients.names

    @property
    def xi_shape(self):
        return self.coefficients.shape

    @property
    def dtype(self):
        from . import utils as kh
        return kh.working_dtype()

    def unpack(self, theta, xi):
        """Expand ``(theta, xi)`` to the named dict used by ``model.clogL``."""
        if theta.shape != (self.theta_size,):
            raise ValueError(
                f"theta has shape {theta.shape}; "
                f"expected ({self.theta_size},)"
            )
        params = dict(Params(theta, self.theta_layout))
        params.update(self.coefficients.unpack(xi))
        return params

    def pack(self, params):
        """Pack a named parameter dict into ``(theta, xi)``."""
        theta = pack_with_layout(params, self.theta_layout)
        return theta, self.coefficients.pack(params)

    def __call__(self, theta, xi):
        """Evaluate the fused kernel. Returns ``(logp, physical_coefficients)``."""
        return self.kernel(theta, xi)

    def oracle(self, theta, xi):
        """Named-graph parity check: ``model.clogL(self.unpack(theta, xi))``."""
        return self.named(self.unpack(theta, xi))

    def diagnostics(self):
        """Eligibility and layout report for an already-built packed object."""
        global_prior = getattr(self.model.globalgp, "separable_prior", None)
        return {
            "eligible": True,
            "form": self.model.clogl_form_resolved,
            "npsr": self.coefficients.shape[0],
            "theta_size": self.theta_size,
            "xi_shape": self.coefficients.shape,
            "global_prior": (
                "separable_fourier" if global_prior is not None else None
            ),
            "global_width": (
                global_prior.width if global_prior is not None else None
            ),
            "extsignals": [
                {
                    "name": ext.name,
                    "width": int(np.asarray(ext.Fs[0]).shape[1]),
                }
                for ext in (self.model.extsignals or [])
            ],
            "transport_conditioner": "batched",
            "jit_input_leaves": 2,
        }

    def samples_to_df(self, chain):
        """Expand packed ``theta`` / ``xi`` draws to named scalar columns."""
        theta = np.asarray(chain["theta"])
        xi = np.asarray(chain["xi"])
        columns = {}

        for name, start, stop, shape in self.theta_layout:
            width = stop - start
            if not shape:
                columns[name] = theta[:, start]
            else:
                root = name[:name.rfind("(")]
                for j in range(width):
                    columns[f"{root}[{j}]"] = theta[:, start + j]

        for row_number, row in enumerate(self.coefficients.rows):
            for name, part in row:
                root = name[:name.rfind("(")] if name.endswith(")") else name
                for j in range(part.stop - part.start):
                    columns[f"{root}[{j}]"] = xi[:, row_number, part.start + j]

        return pd.DataFrame(columns).sort_index(axis=1)


def packed_clogl_diagnostics(model, template_params=None):
    """Report packed-path eligibility without raising on unsupported models."""
    reasons = packed_clogl_ineligibility(model)
    if reasons:
        return {"eligible": False, "reasons": reasons}
    packed = PackedClogL(model, template_params=template_params)
    return packed.diagnostics()
