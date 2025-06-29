"""Model registry on top of an ArtifactStore.

Layout (relative to the bucket root)::

    <name>/models/<kind>/v0001/{model.npz,meta.json,catalog.json,metrics.json}
    <name>/production.json     -> {"kind", "version", "metrics", ...}
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass

import numpy as np

from .models import MODEL_CLASSES, Catalog, Recommender
from .store import ArtifactStore

PROMOTION_METRIC = "ndcg@5"


@dataclass
class LoadedModel:
    model: Recommender
    catalog: Catalog
    kind: str
    version: str
    metrics: dict


class ModelRegistry:
    def __init__(self, store: ArtifactStore, name: str = "menumatch") -> None:
        self.store = store
        self.name = name

    def _prefix(self, kind: str, version: str) -> str:
        return f"{self.name}/models/{kind}/{version}"

    def next_version(self, kind: str) -> str:
        existing = {k.split("/")[3] for k in self.store.list_keys(f"{self.name}/models/{kind}/")}
        nums = [int(v[1:]) for v in existing if v.startswith("v") and v[1:].isdigit()]
        return f"v{max(nums, default=0) + 1:04d}"

    def register(self, model: Recommender, metrics: dict, extra: dict | None = None) -> str:
        version = self.next_version(model.kind)
        prefix = self._prefix(model.kind, version)
        weights, meta = model.to_bytes()
        self.store.put_bytes(f"{prefix}/model.npz", weights)
        self.store.put_bytes(f"{prefix}/meta.json", meta)
        self.store.put_bytes(f"{prefix}/catalog.json", model.catalog.to_json().encode())
        self.store.put_bytes(f"{prefix}/metrics.json", json.dumps({"metrics": metrics, **(extra or {})}).encode())
        return version

    def production(self) -> dict | None:
        key = f"{self.name}/production.json"
        return json.loads(self.store.get_bytes(key)) if self.store.exists(key) else None

    def promote(self, kind: str, version: str, metrics: dict, force: bool = False) -> bool:
        """Point ``production`` at this version unless it would regress the promotion metric."""
        current = self.production()
        if current and not force and metrics[PROMOTION_METRIC] < current["metrics"][PROMOTION_METRIC]:
            return False
        record = {"kind": kind, "version": version, "metrics": metrics}
        self.store.put_bytes(f"{self.name}/production.json", json.dumps(record).encode())
        return True

    def load(self, kind: str, version: str) -> LoadedModel:
        prefix = self._prefix(kind, version)
        catalog = Catalog.from_json(self.store.get_bytes(f"{prefix}/catalog.json").decode())
        meta = json.loads(self.store.get_bytes(f"{prefix}/meta.json"))
        with np.load(io.BytesIO(self.store.get_bytes(f"{prefix}/model.npz"))) as npz:
            arrays = {k: npz[k] for k in npz.files}
        model = MODEL_CLASSES[meta["kind"]].from_state(arrays, meta, catalog)
        metrics = json.loads(self.store.get_bytes(f"{prefix}/metrics.json")).get("metrics", {})
        return LoadedModel(model, catalog, kind, version, metrics)

    def load_production(self) -> LoadedModel | None:
        rec = self.production()
        return self.load(rec["kind"], rec["version"]) if rec else None
