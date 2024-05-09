#!/usr/bin/env python3
"""Tests for discovery likelihood"""

import operator
from functools import reduce
from pathlib import Path

import discovery as ds
import jax
import pytest
from enterprise_extensions import models


class TestLikelihood:
    @pytest.mark.integration
    def test_compare_enterprise(self):
        # The directory containing the pulsar feather files should be parallel to the tests directory
        data_dir = Path(__file__).resolve().parent.parent / "data"

        # Choose two pulsars for reproducibility
        psr_files = [
            data_dir / "v1p1_de440_pint_bipm2019-B1855+09.feather",
            data_dir / "v1p1_de440_pint_bipm2019-B1953+29.feather",
        ]

        # Construct a list of Pulsar objects
        psrs = [ds.Pulsar.read_feather(psr) for psr in psr_files]

        # Get the timespan
        tspan = ds.getspan(psrs)

        # Construct the discovery global likelihood for CURN
        gl = ds.GlobalLikelihood(
            (
                ds.PulsarLikelihood(
                    [
                        psrs[ii].residuals,
                        ds.makenoise_measurement(psrs[ii], psrs[ii].noisedict),
                        ds.makegp_ecorr(psrs[ii], psrs[ii].noisedict),
                        ds.makegp_timing(psrs[ii]),
                        ds.makegp_fourier(psrs[ii], ds.powerlaw, 30, T=tspan, name="red_noise"),
                        ds.makegp_fourier(
                            psrs[ii], ds.powerlaw, 14, T=tspan, common=["gw_log10_A", "gw_gamma"], name="gw"
                        ),
                    ]
                )
                for ii in range(len(psrs))
            )
        )

        # Get the jitted discovery log-likelihood
        jlogl = jax.jit(gl.logL)

        # Construct an enterprise PTA object
        pta = models.model_2a(
            psrs,
            noisedict=reduce(operator.or_, [psr.noisedict for psr in psrs], {}),
            components=30,
            n_gwbfreqs=14,
            Tspan=tspan,
            tm_marg=True,
            tm_svd=True,
        )

        # Get parameters to feed likelihood
        initial_position = ds.prior.sample_uniform(gl.logL.params)

        # Find the difference between enterprise and discovery likelihoods
        ll_difference = pta.get_lnlikelihood(initial_position) - jlogl(initial_position)

        # There is a constant offset of ~ -52.4
        offset = -52.4

        # Choose the absolute tolerance
        atol = 0.1

        assert jax.numpy.abs(ll_difference - offset) <= atol
