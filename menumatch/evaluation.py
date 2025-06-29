"""Offline evaluation: Recall@K, NDCG@K and catalog coverage@K."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models.base import Batch, Recommender

DEFAULT_KS = (1, 3, 5, 10)


def rank_metrics(recs: np.ndarray, targets: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-example (recall@k, ndcg@k) for a single relevant item per example.

    With one relevant item, Recall@K equals hit rate and NDCG@K = 1/log2(rank+2).
    """
    top = recs[:, :k]
    hit = top == targets[:, None]
    recall = hit.any(axis=1).astype(float)
    rank = hit.argmax(axis=1)
    ndcg = np.where(recall > 0, 1.0 / np.log2(rank + 2), 0.0)
    return recall, ndcg


def coverage(recs: np.ndarray, n_candidates: int, k: int) -> float:
    """Share of the add-on catalog that appears in at least one top-k list."""
    shown = np.unique(recs[:, :k])
    return float((shown >= 0).sum() / n_candidates)


def _bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(model: Recommender, examples: pd.DataFrame, ks=DEFAULT_KS, n_boot: int = 200, seed: int = 0) -> dict:
    """Evaluate ``model`` on held-out examples; returns a flat metrics dict."""
    batch = Batch.from_examples(examples)
    targets = examples.target_item.to_numpy()
    recs = model.recommend(batch, max(ks))
    n_addons = int(model.catalog.is_addon.sum())
    out: dict = {"n_examples": int(len(examples))}
    for k in ks:
        recall, ndcg = rank_metrics(recs, targets, k)
        out[f"recall@{k}"] = float(recall.mean())
        out[f"ndcg@{k}"] = float(ndcg.mean())
        out[f"coverage@{k}"] = coverage(recs, n_addons, k)
        if n_boot:
            out[f"recall@{k}_ci"] = _bootstrap_ci(recall, n_boot, seed)
            out[f"ndcg@{k}_ci"] = _bootstrap_ci(ndcg, n_boot, seed)
    return out
