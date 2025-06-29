from __future__ import annotations

import numpy as np

from ..features import FeatureSet
from .base import Batch, Catalog, Recommender


class PopularityRecommender(Recommender):
    """Baseline: most-ordered add-ons of the restaurant in the training window."""

    kind = "popularity"

    def __init__(self, catalog: Catalog, counts: np.ndarray | None = None) -> None:
        super().__init__(catalog)
        self.counts = np.zeros(catalog.n_items) if counts is None else counts

    def fit(self, fs: FeatureSet) -> "PopularityRecommender":
        self.counts = np.zeros(self.catalog.n_items)
        self.counts[fs.item_stats.item_id.to_numpy()] = fs.item_stats.n_orders.to_numpy()
        return self

    def score(self, batch: Batch) -> np.ndarray:
        return np.tile(self.counts, (len(batch), 1))

    def get_state(self):
        return {"counts": self.counts}, {}

    @classmethod
    def from_state(cls, arrays, meta, catalog):
        return cls(catalog, arrays["counts"])
