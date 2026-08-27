"""Direct fused evaluation of a packed decentered cross-form ``clogL``.

Construction materializes the same frozen data products used by the general
graph. Evaluation is a single JAX body: it does not call ``model.clogL``.
"""

from dataclasses import dataclass

import numpy as np

from . import metamatrix as mm
from . import utils as kh
from .params import Params
from .structured import separable_contrib


@dataclass(frozen=True)
class BlockSlices:
    rn: slice
    gw: object  # slice or None


@dataclass(frozen=True)
class FusedClogLConstants:
    G0: object
    b0: object
    ext_centering: tuple
    FtNmF: object
    NmFty: object
    ytNmy: object
    ldN: object
    ext_grams: tuple
    block_slices: BlockSlices
    rn_spectrum: object
    global_prior: object
    conditioner: object


def _validated_block_slices(transport, has_global):
    expected = 2 if has_global else 1
    rows = [tuple(t._block_slice.items()) for t in transport.transports]
    if not rows or any(row != rows[0] for row in rows[1:]):
        raise ValueError("transport block names/slices differ across pulsars")
    if len(rows[0]) != expected:
        raise ValueError(
            f"expected {expected} ordered GP blocks; got {len(rows[0])}"
        )
    slices = [part for _name, part in rows[0]]
    return BlockSlices(
        rn=slices[0],
        gw=slices[1] if has_global else None,
    )


def _common_spectrum(gp, npsr, width):
    getN = gp.Phi.getN

    def spectrum(params):
        phi = kh.jnp.asarray(getN(params))
        if phi.shape == (width,):
            phi = kh.jnp.broadcast_to(phi, (npsr, width))
        if phi.shape != (npsr, width):
            raise ValueError(
                f"common GP spectrum has shape {phi.shape}; "
                f"expected ({npsr}, {width})"
            )
        return phi

    spectrum.params = list(getN.params)
    return spectrum


def _constant_basis(F):
    from .transport import _eval_basis
    return np.asarray(_eval_basis(F), dtype=np.float64)


def _materialize_cross_terms(model, vsm, ys):
    FtNmF = []
    NmFty = []
    ytNmy = []
    ldN = []
    solved = []

    for N, F, y in zip(vsm.Ns, vsm.Fs, ys):
        F = _constant_basis(F)
        y = np.asarray(y, dtype=np.float64)
        solve = mm.func(N.make_solve)
        Nmy, logdet_N = solve(y, params={})
        NmF, _ = solve(F, params={})
        solved.append((solve, NmF))
        FtNmF.append(F.T @ np.asarray(NmF))
        NmFty.append(np.asarray(NmF).T @ y)
        ytNmy.append(y @ np.asarray(Nmy))
        ldN.append(logdet_N)

    ext_grams = []
    for extsignal in model.extsignals or []:
        FcwNmy = []
        FtNmFcw = []
        FcwtNmFcw = []
        for i, (Fcw, y, F) in enumerate(zip(extsignal.Fs, ys, vsm.Fs)):
            F = _constant_basis(F)
            Fcw = _constant_basis(Fcw)
            y = np.asarray(y, dtype=np.float64)
            solve, NmF = solved[i]
            NmFcw, _ = solve(Fcw, params={})
            FcwNmy.append(np.asarray(NmFcw).T @ y)
            FtNmFcw.append(F.T @ np.asarray(NmFcw))
            FcwtNmFcw.append(Fcw.T @ np.asarray(NmFcw))
        ext_grams.append((
            extsignal.coeffs,
            kh.jnparray(FcwNmy),
            kh.jnparray(FtNmFcw),
            kh.jnparray(FcwtNmFcw),
        ))

    return (
        kh.jnparray(FtNmF),
        kh.jnparray(NmFty),
        kh.jnp.sum(kh.jnparray(ytNmy)),
        kh.jnp.sum(kh.jnparray(ldN)),
        tuple(ext_grams),
    )


def _fused_eval(params, xi, const):
    k = xi.shape[1]
    i1, i2 = kh.jnp.diag_indices(k, ndim=2)
    pinv = const.conditioner(params)
    cf = kh.jsp.linalg.cho_factor(
        const.G0.at[:, i1, i2].add(pinv),
        lower=True,
    )
    c = kh.jsp.linalg.solve_triangular(
        cf[0], xi, trans=1, lower=cf[1]
    )
    ldJ = -kh.jnp.logdet(cf[0][:, i1, i2])

    rhs = const.b0
    for E0, coeffs in const.ext_centering:
        ccw = coeffs(params)
        rhs = rhs - kh.jnp.einsum("ijk,ik->ij", E0, ccw)
    c = c + kh.jsp.linalg.cho_solve(cf, rhs)

    logp = (
        -0.5 * const.ytNmy
        + kh.jnp.sum(c * const.NmFty)
        - 0.5 * kh.jnp.einsum("ij,ijk,ik->", c, const.FtNmF, c)
        - 0.5 * const.ldN
        + ldJ
    )

    rn = c[:, const.block_slices.rn]
    phi_rn = const.rn_spectrum(params)
    logp = logp + (
        -0.5 * kh.jnp.sum(rn * rn / phi_rn)
        - 0.5 * kh.jnp.sum(kh.jnp.log(kh.jnp.abs(phi_rn)))
    )

    if const.global_prior is not None:
        gw = c[:, const.block_slices.gw]
        phi = const.global_prior.spectrum(params)
        logp = logp + separable_contrib(
            gw,
            phi,
            const.global_prior.orf_cholesky,
            const.global_prior.orf_logdet,
        )

    for coeffs, FcwNmy, FtNmFcw, FcwtNmFcw in const.ext_grams:
        ccw = coeffs(params)
        logp = logp + kh.jnp.sum(ccw * FcwNmy)
        logp = logp - kh.jnp.einsum("ij,ijk,ik->", c, FtNmFcw, ccw)
        logp = logp - 0.5 * kh.jnp.einsum(
            "ij,ijk,ik->", ccw, FcwtNmFcw, ccw
        )

    return logp, c


def build_fused_clogl(model, transport, theta_layout, coefficient_layout):
    import jax

    vsm, ys = model._coefficient_assembly
    if not transport.center or transport._b0 is None:
        raise ValueError("fused clogL requires a centered ArrayTransport")
    FtNmF, NmFty, ytNmy, ldN, ext_grams = _materialize_cross_terms(
        model, vsm, ys
    )
    slices = _validated_block_slices(
        transport, has_global=model.globalgp is not None
    )
    constants = FusedClogLConstants(
        G0=transport._G0,
        b0=transport._b0,
        ext_centering=tuple(
            (E0, coeffs) for E0, coeffs, _ in transport._extsignals
        ),
        FtNmF=FtNmF,
        NmFty=NmFty,
        ytNmy=ytNmy,
        ldN=ldN,
        ext_grams=ext_grams,
        block_slices=slices,
        rn_spectrum=_common_spectrum(
            model.commongp,
            transport.npsr,
            slices.rn.stop - slices.rn.start,
        ),
        global_prior=getattr(model.globalgp, "separable_prior", None),
        conditioner=transport._conditioner_precision,
    )

    def evaluate(theta, xi):
        params = Params(theta, theta_layout)
        return _fused_eval(params, xi, constants)

    return jax.jit(evaluate)
