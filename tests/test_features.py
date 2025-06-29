import duckdb
import pandas as pd

from menumatch.context import daypart_of
from menumatch.features import build_features


def test_sql_daypart_matches_python():
    con = duckdb.connect()
    con.execute("CREATE TABLE orders AS SELECT 1 AS order_id, 0 AS restaurant_id, "
                "make_timestamp(2025, 1, 6, h, 0, 0) AS ts FROM range(24) t(h)")
    from menumatch.features import run_sql
    from datetime import datetime
    run_sql(con, "orders_enriched", train_end=datetime(2030, 1, 1), val_end=datetime(2030, 1, 2))
    rows = con.execute("SELECT hour, daypart_idx FROM orders_enriched").fetchall()
    assert len(rows) == 24 and all(dp == daypart_of(h) for h, dp in rows)


def test_time_based_split_is_chronological(raw_dir, fs):
    orders = pd.read_parquet(raw_dir / "orders.parquet").set_index("order_id")
    ts = lambda df: orders.loc[df.order_id, "ts"]
    assert ts(fs.train).max() < fs.splits.train_end <= ts(fs.val).min()
    assert ts(fs.val).max() < fs.splits.val_end <= ts(fs.test).min()
    assert len(fs.train) and len(fs.val) and len(fs.test)


def test_eval_examples_are_clean(fs):
    items = fs.items.set_index("item_id")
    for df in (fs.val, fs.test):
        assert df.order_id.is_unique  # one held-out add-on per order
        assert (items.loc[df.target_item, "category"] != "main").all()
        assert not any(t in set(c) for t, c in zip(df.target_item, df.cart))
        assert all(len(c) >= 1 for c in df.cart)  # a main is always left in the cart


def test_no_leakage_from_future_orders(raw_dir, fs, tmp_path):
    """Features/train examples must not change when test-period data is altered."""
    alt = tmp_path / "alt"
    alt.mkdir()
    orders = pd.read_parquet(raw_dir / "orders.parquet")
    oi = pd.read_parquet(raw_dir / "order_items.parquet")
    keep = orders.ts < fs.splits.train_end
    # drop all future orders and add a flood of fake future orders for item 0
    orders[keep].to_parquet(alt / "orders.parquet")
    oi[oi.order_id.isin(orders[keep].order_id)].to_parquet(alt / "order_items.parquet")
    pd.read_parquet(raw_dir / "items.parquet").to_parquet(alt / "items.parquet")
    # keep the same cutoffs by anchoring the date range with one late order (no items)
    late = orders[~keep].iloc[[-1]]
    pd.concat([orders[keep], late]).to_parquet(alt / "orders.parquet")
    fs2 = build_features(alt, tmp_path / "alt.duckdb")
    assert fs2.splits == fs.splits
    pd.testing.assert_frame_equal(fs.item_stats, fs2.item_stats)
    pd.testing.assert_frame_equal(fs.item_pairs, fs2.item_pairs)
    pd.testing.assert_frame_equal(fs.train, fs2.train)
