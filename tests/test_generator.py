import numpy as np

from menumatch.generator import World, generate_orders


def test_world_and_orders_are_deterministic():
    w1, w2 = World.build(1, 5), World.build(1, 5)
    o1, i1 = generate_orders(w1, 30, 1)
    o2, i2 = generate_orders(w2, 30, 1)
    assert o1.equals(o2) and i1.equals(i2)
    o3, _ = generate_orders(World.build(2, 5), 30, 2)
    assert not o1.equals(o3)


def test_orders_have_mains_and_valid_menus():
    w = World.build(1, 6)
    orders, oi = generate_orders(w, 40, 1)
    items = w.items.set_index("item_id")
    merged = oi.merge(orders, on="order_id")
    # every item belongs to the restaurant of its order
    assert (items.loc[merged.item_id, "restaurant_id"].to_numpy() == merged.restaurant_id.to_numpy()).all()
    cats = items.loc[oi.item_id, "category"].to_numpy()
    has_main = oi.assign(main=cats == "main").groupby("order_id").main.any()
    assert has_main.all() and len(has_main) == len(orders)
    # at most one add-on per category per order
    addons = oi.assign(cat=cats)[cats != "main"]
    assert addons.groupby(["order_id", "cat"]).size().max() == 1


def test_addon_probs_are_valid_and_context_dependent():
    w = World.build(1, 10)
    mains = w.items[(w.items.restaurant_id == 0) & (w.items.category == "main")].item_id.tolist()[:1]
    p_lunch = w.addon_probs(0, mains, 1, 0, 0.5)
    p_dinner = w.addon_probs(0, mains, 3, 1, 0.5)
    assert (p_lunch >= 0).all() and p_lunch[w.items.category == "main"].sum() == 0
    for cat in ("side", "drink", "dessert"):
        idx = w.items[(w.items.restaurant_id == 0) & (w.items.category == cat)].item_id.to_numpy()
        assert 0 < p_lunch[idx].sum() < 1
    dessert = w.items[(w.items.restaurant_id == 0) & (w.items.category == "dessert")].item_id.to_numpy()
    assert p_dinner[dessert].sum() > p_lunch[dessert].sum()


def test_world_roundtrip(tmp_path):
    w = World.build(4, 7)
    w.save(tmp_path / "w.json")
    w2 = World.load(tmp_path / "w.json")
    assert np.allclose(w.z, w2.z) and w.items.equals(w2.items)
