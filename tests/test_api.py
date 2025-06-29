import pytest
from fastapi.testclient import TestClient

from menumatch.registry import LoadedModel
from menumatch.serving.app import create_app


@pytest.fixture
def client(models, catalog):
    loaded = LoadedModel(models["two_tower"], catalog, "two_tower", "v0001", {})
    with TestClient(create_app(loaded)) as c:
        yield c


def _ids(catalog, restaurant, category):
    it = catalog.items
    return it[(it.restaurant_id == restaurant) & (it.category == category)].item_id.tolist()


def test_recommend_happy_path(client, catalog):
    main = _ids(catalog, 1, "main")[0]
    r = client.post("/recommend", json={"restaurant_id": 1, "cart_item_ids": [main], "k": 4, "hour": 19, "is_weekend": True})
    assert r.status_code == 200
    body = r.json()
    recs = body["recommendations"]
    assert len(recs) == 4 and body["model_kind"] == "two_tower"
    assert all(catalog.restaurant[x["item_id"]] == 1 and x["category"] != "main" for x in recs)
    assert main not in [x["item_id"] for x in recs]
    scores = [x["score"] for x in recs]
    assert scores == sorted(scores, reverse=True)


def test_recommend_validation(client, catalog):
    other = _ids(catalog, 2, "main")[0]
    assert client.post("/recommend", json={"restaurant_id": 99}).status_code == 404
    assert client.post("/recommend", json={"restaurant_id": 1, "cart_item_ids": [other]}).status_code == 422
    assert client.post("/recommend", json={"restaurant_id": 1, "k": 0}).status_code == 422
    assert client.post("/recommend", json={"restaurant_id": 1, "hour": 30}).status_code == 422


def test_empty_cart_and_cart_items_excluded(client, catalog):
    drinks = _ids(catalog, 0, "drink")
    r = client.post("/recommend", json={"restaurant_id": 0, "cart_item_ids": drinks, "k": 20})
    ids = {x["item_id"] for x in r.json()["recommendations"]}
    assert not ids & set(drinks)
    assert client.post("/recommend", json={"restaurant_id": 0}).status_code == 200


def test_health_and_menu(client):
    assert client.get("/health").json()["status"] == "ok"
    menu = client.get("/restaurants/0/menu").json()
    assert {m["category"] for m in menu} == {"main", "side", "drink", "dessert"}
    assert client.get("/restaurants/42/menu").status_code == 404


def test_no_model_returns_503(tmp_path):
    from menumatch.config import Settings
    s = Settings(store="local", local_store_dir=tmp_path)
    with TestClient(create_app(settings=s)) as c:
        assert c.get("/health").json()["status"] == "no_model"
        assert c.post("/recommend", json={"restaurant_id": 0}).status_code == 503
