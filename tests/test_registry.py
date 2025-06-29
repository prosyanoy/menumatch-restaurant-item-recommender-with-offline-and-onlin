import boto3
import numpy as np
import pytest
from moto import mock_aws

from menumatch.models import Batch
from menumatch.registry import ModelRegistry
from menumatch.store import LocalStore, S3Store


def _store_roundtrip(store):
    store.put_bytes("a/b.txt", b"hi")
    assert store.exists("a/b.txt") and not store.exists("a/c.txt")
    assert store.get_bytes("a/b.txt") == b"hi"
    assert store.list_keys("a/") == ["a/b.txt"]
    with pytest.raises(KeyError):
        store.get_bytes("nope")


def test_local_store(tmp_path):
    _store_roundtrip(LocalStore(tmp_path))
    with pytest.raises(ValueError):
        LocalStore(tmp_path).put_bytes("../escape", b"x")


@mock_aws
def test_s3_store_against_moto():
    _store_roundtrip(S3Store("bkt", client=boto3.client("s3", region_name="us-east-1")))


@mock_aws
def test_registry_register_promote_load_on_s3(catalog, models, fs):
    from menumatch.evaluation import evaluate

    store = S3Store("bkt", client=boto3.client("s3", region_name="us-east-1"))
    reg = ModelRegistry(store, "t")
    assert reg.production() is None and reg.load_production() is None
    m = models["cooccurrence"]
    metrics = evaluate(m, fs.test, n_boot=0)
    v1 = reg.register(m, metrics)
    v2 = reg.register(m, metrics)
    assert (v1, v2) == ("v0001", "v0002")
    assert reg.promote("cooccurrence", v2, metrics)
    worse = {**metrics, "ndcg@5": metrics["ndcg@5"] - 0.1}
    assert not reg.promote("cooccurrence", v1, worse)  # regression gate
    assert reg.production()["version"] == v2
    assert reg.promote("cooccurrence", v1, worse, force=True)

    loaded = reg.load("cooccurrence", v2)
    batch = Batch.from_examples(fs.test.head(30))
    assert np.allclose(loaded.model.score(batch), m.score(batch))
    assert loaded.metrics["ndcg@5"] == pytest.approx(metrics["ndcg@5"])
