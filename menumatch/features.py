"""Feature/example construction in DuckDB (SQL lives in ``menumatch/sql``)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

SQL_DIR = Path(__file__).parent / "sql"
SQL_STEPS = [
    "orders_enriched",
    "item_stats",
    "item_daypart_stats",
    "item_pairs",
    "examples_train",
    "examples_eval",
]


@dataclass
class Splits:
    train_end: datetime
    val_end: datetime

    def as_dict(self) -> dict:
        return {"train_end": self.train_end.isoformat(), "val_end": self.val_end.isoformat()}


@dataclass
class FeatureSet:
    """Everything the models/evaluator need, all derived from the train window."""

    items: pd.DataFrame
    item_stats: pd.DataFrame
    item_pairs: pd.DataFrame
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    splits: Splits


def compute_splits(con: duckdb.DuckDBPyConnection, train_frac: float, val_frac: float) -> Splits:
    """Day-aligned chronological cutoffs: train < train_end <= val < val_end <= test."""
    lo, hi = con.execute("SELECT min(ts), max(ts) FROM orders").fetchone()
    first_day = datetime(lo.year, lo.month, lo.day)
    n_days = (datetime(hi.year, hi.month, hi.day) - first_day).days + 1
    train_end = first_day + timedelta(days=int(n_days * train_frac))
    val_end = first_day + timedelta(days=int(n_days * (train_frac + val_frac)))
    return Splits(train_end, val_end)


def run_sql(con: duckdb.DuckDBPyConnection, name: str, **params) -> None:
    sql = (SQL_DIR / f"{name}.sql").read_text()
    used = {k: v for k, v in params.items() if f"${k}" in sql}
    con.execute(sql, used)


def build_features(
    raw_dir: Path, db_path: Path, train_frac: float = 0.7, val_frac: float = 0.1
) -> FeatureSet:
    """Load raw parquet into DuckDB, run the SQL feature pipeline, return the feature set."""
    raw_dir, db_path = Path(raw_dir), Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        for table in ("orders", "order_items", "items"):
            path = (raw_dir / f"{table}.parquet").as_posix().replace("'", "''")
            con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_parquet('{path}')")
        return build_features_from_connection(con, train_frac, val_frac)
    finally:
        con.close()


def build_features_from_connection(con: duckdb.DuckDBPyConnection, train_frac: float, val_frac: float) -> FeatureSet:
    splits = compute_splits(con, train_frac, val_frac)
    for step in SQL_STEPS:
        run_sql(con, step, train_end=splits.train_end, val_end=splits.val_end)
    ev = con.execute("SELECT * FROM examples_eval ORDER BY order_id").df()
    return FeatureSet(
        items=con.execute("SELECT * FROM items ORDER BY item_id").df(),
        item_stats=con.execute("SELECT * FROM item_stats ORDER BY item_id").df(),
        item_pairs=con.execute("SELECT * FROM item_pairs ORDER BY item_a, item_b").df(),
        train=con.execute("SELECT * FROM examples_train ORDER BY order_id, target_item").df(),
        val=ev[ev.split == "val"].reset_index(drop=True),
        test=ev[ev.split == "test"].reset_index(drop=True),
        splits=splits,
    )


def save_features(fs: FeatureSet, out_dir: Path) -> None:
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("items", "item_stats", "item_pairs", "train", "val", "test"):
        getattr(fs, name).to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "splits.json").write_text(json.dumps(fs.splits.as_dict()))


def load_features(in_dir: Path) -> FeatureSet:
    import json

    in_dir = Path(in_dir)
    s = json.loads((in_dir / "splits.json").read_text())
    frames = {n: pd.read_parquet(in_dir / f"{n}.parquet") for n in ("items", "item_stats", "item_pairs", "train", "val", "test")}
    return FeatureSet(
        splits=Splits(datetime.fromisoformat(s["train_end"]), datetime.fromisoformat(s["val_end"])), **frames
    )
