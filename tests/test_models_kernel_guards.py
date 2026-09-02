"""Guards on discovery.models builders that construct matrix.* kernels directly."""
import pytest

import discovery as ds
from discovery.models import epta


@pytest.mark.parametrize(
    "builder, kwargs",
    [
        (epta.makemodel_singlepulsar, {"psr": None}),
        (epta.makemodel_curn, {"psrs": []}),
        (epta.makemodel_hd, {"psrs": []}),
    ],
    ids=["singlepulsar", "curn", "hd"],
)
def test_epta_builders_require_matrix(builder, kwargs):
    saved = ds.config()
    ds.config(kernels="metamath")
    try:
        with pytest.raises(NotImplementedError, match=r"ds\.config"):
            builder(**kwargs)
    finally:
        ds.config(kernels=saved)


def test_outlier_builder_requires_matrix():
    pytest.importorskip("numpyro")
    from discovery.models.nanograv_single_pulsar_outlier import (
        make_outlier_likelihood,
    )

    saved = ds.config()
    ds.config(kernels="metamath")
    try:
        with pytest.raises(NotImplementedError, match=r"ds\.config"):
            make_outlier_likelihood(None)
    finally:
        ds.config(kernels=saved)
