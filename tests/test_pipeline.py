import json

from fastapi.testclient import TestClient

from menumatch import pipeline
from menumatch.generator import World
from menumatch.registry import ModelRegistry
from menumatch.serving.app import create_app
from menumatch.simulation import simulate_ab
from menumatch.store import get_store


def test_end_to_end_pipeline_registers_and_serves(small_settings):
    out = pipeline.run_all(small_settings, "2025-01-01T00:00:00+00:00")  # airflow-style run id
    run_dir = small_settings.run_dir("2025-01-01T00:00:00+00:00")
    assert ":" not in run_dir.name and (run_dir / "metrics.json").exists() and (run_dir / "online.json").exists()
    assert set(out["offline"]["models"]) == {"popularity", "cooccurrence", "two_tower"}
    assert out["registry"]["promoted"]

    store = get_store(small_settings)
    prod = ModelRegistry(store, small_settings.registry_name).production()
    assert prod["kind"] == out["registry"]["best"]
    assert json.loads(store.get_bytes(f"{small_settings.registry_name}/models/two_tower/v0001/metrics.json"))["metrics"]

    # a second run registers v0002 and the API serves the promoted model from the registry
    pipeline.run_all(small_settings, "second")
    assert ModelRegistry(store, small_settings.registry_name).next_version("two_tower") == "v0003"
    with TestClient(create_app(settings=small_settings)) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/recommend", json={"restaurant_id": 0, "cart_item_ids": [0], "k": 3})
        assert r.status_code == 200 and len(r.json()["recommendations"]) == 3


def test_online_simulation_is_deterministic_and_informative(small_settings):
    pipeline.generate(small_settings, "s")
    pipeline.features(small_settings, "s")
    pipeline.train(small_settings, "s", two_tower_params={"epochs": 4})
    world = World.load(small_settings.run_dir("s") / "raw" / "world.json")
    models = pipeline._load_trained(small_settings.run_dir("s"))
    a = simulate_ab(world, models, "popularity", 4000, seed=1)
    b = simulate_ab(world, models, "popularity", 4000, seed=1)
    assert a == b
    assert sum(arm["sessions"] for arm in a["arms"].values()) == 4000
    assert a["arms"]["cooccurrence"]["conversion_lift_rel"] > 0
    assert "conversion_p_value" in a["arms"]["two_tower"] and "popularity" in a["arms"]
