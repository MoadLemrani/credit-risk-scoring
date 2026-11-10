import pytest

from creditrisk import train


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory):
    """Train a tiny model once (3k-row sample, small ensembles) so the API can be tested end to end."""
    out = tmp_path_factory.mktemp("artifacts")
    train.run(out, quick=True, use_mlflow=False)
    return out
