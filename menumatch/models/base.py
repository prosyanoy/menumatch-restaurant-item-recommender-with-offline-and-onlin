"""Common recommender interface, catalog and batch containers."""
from __future__ import annotations

import io
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..context import CATEGORIES
from ..features import FeatureSet


@dataclass
class Catalog:
    """Item metadata as dense arrays indexed by contiguous ``item_id``."""

    items: pd.DataFrame

    def __post_init__(self) -> None:
        self.items = self.items.sort_values("item_id").reset_index(drop=True)
        if not (self.items.item_id.to_numpy() == np.arange(len(self.items))).all():
            raise ValueError("item_id must be contiguous 0..N-1")
        self.n_items = len(self.items)
        self.restaurant = self.items.restaurant_id.to_numpy()
        self.n_restaurants = int(self.restaurant.max()) + 1
        self.category = self.items.category.map(CATEGORIES.index).to_numpy()
        self.price = self.items.price.to_numpy(dtype=float)
        self.is_addon = self.category != CATEGORIES.index("main")

    def to_json(self) -> str:
        return self.items[["item_id", "restaurant_id", "restaurant_name", "name", "category", "price"]].to_json(orient="records")

    @classmethod
    def from_json(cls, s: str) -> "Catalog":
        return cls(pd.read_json(io.StringIO(s), orient="records"))


@dataclass
class Batch:
    restaurant: np.ndarray  # (B,)
    carts: list[np.ndarray]
    daypart: np.ndarray  # (B,)
    weekend: np.ndarray  # (B,)

    def __len__(self) -> int:
        return len(self.restaurant)

    @classmethod
    def from_examples(cls, df: pd.DataFrame) -> "Batch":
        return cls(
            restaurant=df.restaurant_id.to_numpy(dtype=np.int64, copy=True),
            carts=[np.asarray(c, dtype=np.int64) for c in df.cart],
            daypart=df.daypart_idx.to_numpy(dtype=np.int64, copy=True),
            weekend=df.is_weekend.to_numpy(dtype=np.int64, copy=True),
        )

    def take(self, idx: np.ndarray) -> "Batch":
        return Batch(self.restaurant[idx], [self.carts[i] for i in idx], self.daypart[idx], self.weekend[idx])


def candidate_mask(catalog: Catalog, batch: Batch) -> np.ndarray:
    """(B, N) bool: add-ons of the order's restaurant that are not already in the cart."""
    mask = catalog.is_addon[None, :] & (catalog.restaurant[None, :] == batch.restaurant[:, None])
    for row, cart in enumerate(batch.carts):
        mask[row, cart] = False
    return mask


class Recommender:
    """Interface: fit on a FeatureSet, score a Batch, (de)serialise without pickle."""

    kind = "base"

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    def fit(self, fs: FeatureSet) -> "Recommender":
        raise NotImplementedError

    def score(self, batch: Batch) -> np.ndarray:
        """Raw (B, N) scores; masking is applied in ``recommend``."""
        raise NotImplementedError

    def recommend(self, batch: Batch, k: int) -> np.ndarray:
        """Top-k item ids per row among valid candidates, (B, k); -1 pads short lists."""
        scores = np.where(candidate_mask(self.catalog, batch), self.score(batch), -np.inf)
        k_eff = min(k, scores.shape[1])
        top = np.argsort(-scores, axis=1, kind="stable")[:, :k_eff]
        top_scores = np.take_along_axis(scores, top, axis=1)
        top = np.where(np.isfinite(top_scores), top, -1)
        if k_eff < k:
            top = np.pad(top, ((0, 0), (0, k - k_eff)), constant_values=-1)
        return top

    # -- serialisation: arrays (npz) + json-able metadata --------------
    def get_state(self) -> tuple[dict[str, np.ndarray], dict]:
        raise NotImplementedError

    @classmethod
    def from_state(cls, arrays: dict[str, np.ndarray], meta: dict, catalog: Catalog) -> "Recommender":
        raise NotImplementedError

    def to_bytes(self) -> tuple[bytes, bytes]:
        arrays, meta = self.get_state()
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        return buf.getvalue(), json.dumps({"kind": self.kind, **meta}).encode()
