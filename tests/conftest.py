import pytest

from menumatch.config import Settings
from menumatch.features import build_features
from menumatch.generator import generate_to_dir
from menumatch.models import Catalog, CooccurrenceRecommender, PopularityRecommender, TwoTowerRecommender


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    generate_to_dir(d, seed=3, n_days=40, orders_per_day=120)
    return d


@pytest.fixture(scope="session")
def fs(raw_dir, tmp_path_factory):
    return build_features(raw_dir, tmp_path_factory.mktemp("db") / "t.duckdb")


@pytest.fixture(scope="session")
def catalog(fs):
    return Catalog(fs.items)


@pytest.fixture(scope="session")
def models(fs, catalog):
    return {
        "popularity": PopularityRecommender(catalog).fit(fs),
        "cooccurrence": CooccurrenceRecommender(catalog).fit(fs),
        "two_tower": TwoTowerRecommender(catalog, epochs=8, patience=8).fit(fs),
    }


@pytest.fixture
def small_settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", seed=5, n_days=30, orders_per_day=80, sim_sessions=3000,
                    store="local", local_store_dir=tmp_path / "art", s3_endpoint=None)
