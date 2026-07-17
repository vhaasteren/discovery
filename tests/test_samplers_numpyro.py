"""`discovery.samplers.numpyro` — kwarg routing and checkpoint-runner hardening.

The old `makesampler_nuts` iterated `kwargs.items()` and tested each `(key,
value)` tuple for membership in an argname list, so the test was always false
and every kwarg was silently discarded (D14). These tests pin the routing.
"""
import pickle

import pandas as pd
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import numpyro  # noqa: E402
from numpyro import distributions as dist  # noqa: E402
from numpyro import infer  # noqa: E402

from discovery.samplers import numpyro as ds_numpyro  # noqa: E402


@pytest.fixture
def toy_model():
    """A minimal NumPyro model carrying the `to_df` attribute that
    `makesampler_nuts` / `run_nuts_with_checkpoints` rely on."""
    def numpyro_model():
        numpyro.sample("x", dist.Normal(0.0, 1.0))

    numpyro_model.to_df = lambda chain: pd.DataFrame({"x": chain["x"]})
    return numpyro_model


class TestMakesamplerNuts:
    def test_defaults_reach_nuts_and_mcmc(self, toy_model):
        sampler = ds_numpyro.makesampler_nuts(toy_model)

        assert sampler.sampler._max_tree_depth == 8
        assert sampler.sampler._dense_mass is False
        assert sampler.sampler._forward_mode_differentiation is False
        assert sampler.sampler._target_accept_prob == 0.8

        assert sampler.num_warmup == 512
        assert sampler.num_samples == 1024
        assert sampler.num_chains == 1
        assert sampler.chain_method == "vectorized"

    def test_nuts_override_is_honored(self, toy_model):
        """The regression: `max_tree_depth=6` used to be dropped silently."""
        sampler = ds_numpyro.makesampler_nuts(toy_model, max_tree_depth=6)

        assert sampler.sampler._max_tree_depth == 6

    def test_mcmc_override_is_honored(self, toy_model):
        sampler = ds_numpyro.makesampler_nuts(toy_model, num_warmup=7,
                                              progress_bar=False)

        assert sampler.num_warmup == 7
        assert sampler.progress_bar is False

    def test_unknown_kwarg_raises_typeerror_naming_it(self, toy_model):
        with pytest.raises(TypeError, match="max_tree_dpeth"):
            ds_numpyro.makesampler_nuts(toy_model, max_tree_dpeth=6)

    def test_potential_fn_raises_because_a_model_is_supplied(self, toy_model):
        """`potential_fn` is in the NUTS signature but is mutually exclusive
        with the positional model this factory always passes."""
        with pytest.raises(TypeError, match="potential_fn"):
            ds_numpyro.makesampler_nuts(toy_model, potential_fn=lambda z: 0.0)

    def test_to_df_delegates_to_the_model(self, toy_model):
        sampler = ds_numpyro.makesampler_nuts(toy_model, num_warmup=2,
                                              num_samples=3, progress_bar=False)
        sampler.run(jax.random.PRNGKey(0))

        df = sampler.to_df()

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["x"]
        assert len(df) == 3


class TestEnsureSamplerToDf:
    def test_noop_when_already_present(self, toy_model):
        sampler = ds_numpyro.makesampler_nuts(toy_model)
        before = sampler.to_df

        ds_numpyro._ensure_sampler_to_df(sampler)

        assert sampler.to_df is before

    def test_attaches_from_sampler_sampler_model(self, toy_model):
        """A raw MCMC around a to_df-carrying model gets the attachment back."""
        sampler = infer.MCMC(infer.NUTS(toy_model), num_warmup=2, num_samples=3,
                             progress_bar=False)
        assert not hasattr(sampler, "to_df")

        ds_numpyro._ensure_sampler_to_df(sampler)
        sampler.run(jax.random.PRNGKey(0))

        assert list(sampler.to_df().columns) == ["x"]

    def test_raises_attributeerror_without_a_to_df_model(self):
        def bare_model():
            numpyro.sample("x", dist.Normal(0.0, 1.0))

        sampler = infer.MCMC(infer.NUTS(bare_model), num_warmup=2,
                             num_samples=3, progress_bar=False)

        with pytest.raises(AttributeError, match="has no to_df"):
            ds_numpyro._ensure_sampler_to_df(sampler)


class TestRunNutsWithCheckpoints:
    def test_creates_missing_nested_outdir_and_appends_across_rounds(
            self, toy_model, tmp_path):
        sampler = ds_numpyro.makesampler_nuts(toy_model, num_warmup=4,
                                              num_samples=4, progress_bar=False)
        outdir = tmp_path / "deeply" / "nested" / "chains"
        assert not outdir.exists()

        ds_numpyro.run_nuts_with_checkpoints(sampler, 2, jax.random.PRNGKey(0),
                                             outdir=outdir)

        assert outdir.is_dir()
        # Two checkpoint rounds of 2 samples each, appended into one frame.
        df = pd.read_feather(outdir / "numpyro-samples.feather")
        assert len(df) == 4

        # The checkpoint holds sampler state, not model objects.
        with (outdir / "numpyro-checkpoint.pickle").open("rb") as f:
            pickle.load(f)

    def test_accepts_a_path_outdir_that_does_not_exist(self, toy_model, tmp_path):
        """The old body only created `outdir` when it was *not* already a Path,
        so passing a Path to a missing directory failed."""
        sampler = ds_numpyro.makesampler_nuts(toy_model, num_warmup=2,
                                              num_samples=2, progress_bar=False)
        outdir = tmp_path / "as_path"

        ds_numpyro.run_nuts_with_checkpoints(sampler, 2, jax.random.PRNGKey(0),
                                             outdir=outdir)

        assert (outdir / "numpyro-samples.feather").is_file()
