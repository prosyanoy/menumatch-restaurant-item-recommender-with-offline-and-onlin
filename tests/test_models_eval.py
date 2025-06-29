import numpy as np
import pytest

from menumatch.evaluation import coverage, evaluate, rank_metrics
from menumatch.models import Batch, TwoTowerRecommender, candidate_mask


def test_rank_metrics_by_hand():
    recs = np.array([[5, 6, 7], [1, 2, 3], [9, 8, 7]])
    targets = np.array([5, 3, 4])
    recall, ndcg = rank_metrics(recs, targets, 3)
    assert recall.tolist() == [1, 1, 0]
    assert ndcg == pytest.approx([1.0, 1 / np.log2(4), 0.0])
    recall1, _ = rank_metrics(recs, targets, 1)
    assert recall1.tolist() == [1, 0, 0]


def test_coverage_ignores_padding():
    recs = np.array([[0, 1], [1, -1]])
    assert coverage(recs, 4, 2) == 0.5


def test_candidates_only_same_restaurant_addons_not_in_cart(fs, catalog, models):
    batch = Batch.from_examples(fs.test.head(200))
    mask = candidate_mask(catalog, batch)
    for name, m in models.items():
        recs = m.recommend(batch, 5)
        for row, cart in enumerate(batch.carts):
            for item in recs[row]:
                assert mask[row, item], (name, row, item)
                assert item not in cart
            assert len(set(recs[row])) == 5


def test_model_ordering_on_test_split(fs, models):
    res = {k: evaluate(m, fs.test, n_boot=0) for k, m in models.items()}
    assert res["cooccurrence"]["ndcg@5"] > res["popularity"]["ndcg@5"]
    assert res["two_tower"]["ndcg@5"] > res["popularity"]["ndcg@5"]
    for r in res.values():
        assert r["recall@1"] <= r["recall@3"] <= r["recall@5"] <= 1


def test_evaluate_reports_ci(fs, models):
    r = evaluate(models["popularity"], fs.test, ks=(3,), n_boot=50)
    lo, hi = r["ndcg@3_ci"]
    assert lo <= r["ndcg@3"] <= hi


def test_popularity_matches_sql_counts(fs, models):
    top = models["popularity"].counts
    row = fs.item_stats.iloc[0]
    assert top[int(row.item_id)] == row.n_orders


def test_two_tower_trains_and_roundtrips(fs, catalog, models):
    m = models["two_tower"]
    assert m.history[-1]["train_loss"] < m.history[0]["train_loss"]
    weights, meta = m.to_bytes()
    import io, json
    with np.load(io.BytesIO(weights)) as npz:
        arrays = {k: npz[k] for k in npz.files}
    clone = TwoTowerRecommender.from_state(arrays, json.loads(meta), catalog)
    batch = Batch.from_examples(fs.test.head(50))
    assert np.allclose(m.score(batch), clone.score(batch), atol=1e-6)


def test_cooccurrence_handles_empty_cart(catalog, models):
    batch = Batch(np.array([0]), [np.array([], dtype=np.int64)], np.array([1]), np.array([0]))
    recs = models["cooccurrence"].recommend(batch, 3)
    assert (recs >= 0).all()
