from __future__ import annotations

import numpy as np

from ..features import FeatureSet
from .base import Batch, Catalog, Recommender


class CooccurrenceRecommender(Recommender):
    """Item-item graph model.

    Edge weight is the cosine-normalised basket co-occurrence
    ``n_ab / sqrt(n_a * n_b)``. A cart is scored by summing the edge weights from
    its items (1-hop) plus a damped 2-hop walk over the row-normalised graph, with a
    small popularity prior so empty/unseen carts still get a sensible ranking.
    """

    kind = "cooccurrence"

    def __init__(self, catalog: Catalog, sim: np.ndarray | None = None, pop: np.ndarray | None = None,
                 hop2_weight: float = 0.2, pop_weight: float = 0.05) -> None:
        super().__init__(catalog)
        n = catalog.n_items
        self.sim = np.zeros((n, n)) if sim is None else sim
        self.pop = np.zeros(n) if pop is None else pop
        self.hop2_weight = hop2_weight
        self.pop_weight = pop_weight
        self._graph = self._combine()

    def _combine(self) -> np.ndarray:
        row_sum = self.sim.sum(axis=1, keepdims=True)
        walk = np.divide(self.sim, row_sum, out=np.zeros_like(self.sim), where=row_sum > 0)
        return self.sim + self.hop2_weight * (walk @ walk)

    def fit(self, fs: FeatureSet) -> "CooccurrenceRecommender":
        n = self.catalog.n_items
        n_i = np.zeros(n)
        n_i[fs.item_stats.item_id.to_numpy()] = fs.item_stats.n_orders.to_numpy()
        pairs = fs.item_pairs
        counts = np.zeros((n, n))
        counts[pairs.item_a.to_numpy(), pairs.item_b.to_numpy()] = pairs.n_orders.to_numpy()
        denom = np.sqrt(np.outer(n_i, n_i))
        self.sim = np.divide(counts, denom, out=np.zeros_like(counts), where=denom > 0)
        self.pop = n_i / max(n_i.max(), 1.0)
        self._graph = self._combine()
        return self

    def score(self, batch: Batch) -> np.ndarray:
        out = np.tile(self.pop_weight * self.pop, (len(batch), 1))
        for row, cart in enumerate(batch.carts):
            if len(cart):
                out[row] += self._graph[cart].sum(axis=0)
        return out

    def get_state(self):
        return {"sim": self.sim, "pop": self.pop}, {"hop2_weight": self.hop2_weight, "pop_weight": self.pop_weight}

    @classmethod
    def from_state(cls, arrays, meta, catalog):
        return cls(catalog, arrays["sim"], arrays["pop"], meta["hop2_weight"], meta["pop_weight"])
