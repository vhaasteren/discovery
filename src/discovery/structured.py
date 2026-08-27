"""Separable Fourier-frequency priors for global GPs.

A single ORF and a diagonal Fourier spectrum have covariance
``Gamma kron diag(phi)`` in pulsar-major order. This module stores the
constant ORF factor once and evaluates the coefficient log-density without
materializing the dense Kronecker product.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import utils as kh


def diagonal_fourier_covariance(func):
    """Mark a PSD factory whose contract is independent Fourier variances.

    The returned callable must evaluate to a 1-D array of length ``k`` (one
    variance per Fourier coefficient). Custom priors opt in by using this
    decorator or by setting the same ``fourier_covariance = "diagonal"``
    attribute. Untagged priors keep the dense covariance path.
    """
    func.fourier_covariance = "diagonal"
    return func


def separable_contrib(c, phi, orf_cholesky, orf_logdet):
    """Gaussian log-density of a pulsar-major ``(npsr, width)`` coefficient block.

    ``c`` is whitened against the constant ORF Cholesky; frequency columns
    are independent given ``phi``.
    """
    whitened = kh.jsp.linalg.solve_triangular(orf_cholesky, c, lower=True)
    quad = kh.jnp.sum(whitened * whitened / phi[None, :])
    logdet = c.shape[1] * orf_logdet + c.shape[0] * kh.jnp.sum(kh.jnp.log(phi))
    return -0.5 * (quad + logdet)


@dataclass(frozen=True)
class SeparableFourierPrior:
    """``Gamma`` (pulsar) kron ``diag(spectrum)``, in pulsar-major ordering."""

    orf_cholesky: object
    orf_logdet: float
    orf_diagonal: object
    spectrum: Callable
    npsr: int
    width: int

    @classmethod
    def build(cls, orf, spectrum, width):
        orf = np.asarray(orf, dtype=np.float64)
        if orf.ndim != 2 or orf.shape[0] != orf.shape[1]:
            raise ValueError("separable ORF must be square")
        if not np.allclose(orf, orf.T, rtol=1e-13, atol=1e-15):
            raise ValueError("separable ORF must be symmetric")
        sign, logdet = np.linalg.slogdet(orf)
        if sign <= 0:
            raise ValueError("separable ORF must be positive definite")
        chol = np.linalg.cholesky(orf)
        return cls(
            orf_cholesky=kh.to_working(chol),
            orf_logdet=float(logdet),
            orf_diagonal=kh.to_working(np.diag(orf)),
            spectrum=spectrum,
            npsr=orf.shape[0],
            width=int(width),
        )

    @property
    def params(self):
        return list(self.spectrum.params)

    def marginal_precision(self, params):
        phi = kh.jnp.asarray(self.spectrum(params))
        return 1.0 / (self.orf_diagonal[:, None] * phi[None, :])
