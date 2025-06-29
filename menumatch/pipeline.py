"""Pipeline stages. Each stage reads/writes files under ``<data_dir>/runs/<run_id>/``
so they can run as separate Airflow tasks (or all in-process via ``run_all``)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import Settings
from .evaluation import evaluate
from .features import build_features, load_features, save_features
from .generator import World, generate_to_dir
from .models import MODEL_CLASSES, Catalog
from .registry import PROMOTION_METRIC, ModelRegistry
from .simulation import simulate_ab
from .store import LocalStore, get_store

log = logging.getLogger("menumatch")
MODEL_ORDER = ["popularity", "cooccurrence", "two_tower"]


def _trained_registry(run_dir: Path) -> ModelRegistry:
    return ModelRegistry(LocalStore(run_dir / "trained"), "run")


def _load_trained(run_dir: Path) -> dict:
    reg = _trained_registry(run_dir)
    return {kind: reg.load(kind, "v0001").model for kind in MODEL_ORDER}


def generate(settings: Settings, run_id: str = "latest") -> Path:
    run_dir = settings.run_dir(run_id)
    generate_to_dir(run_dir / "raw", settings.seed, settings.n_days, settings.orders_per_day, settings.n_restaurants)
    log.info("generated raw data in %s", run_dir / "raw")
    return run_dir


def features(settings: Settings, run_id: str = "latest") -> Path:
    run_dir = settings.run_dir(run_id)
    fs = build_features(run_dir / "raw", run_dir / "menumatch.duckdb", settings.train_frac, settings.val_frac)
    save_features(fs, run_dir / "features")
    log.info("features: train=%d val=%d test=%d examples", len(fs.train), len(fs.val), len(fs.test))
    return run_dir


def train(settings: Settings, run_id: str = "latest", two_tower_params: dict | None = None) -> Path:
    import shutil

    run_dir = settings.run_dir(run_id)
    fs = load_features(run_dir / "features")
    catalog = Catalog(fs.items)
    shutil.rmtree(run_dir / "trained", ignore_errors=True)
    reg = _trained_registry(run_dir)
    for kind in MODEL_ORDER:
        model = MODEL_CLASSES[kind](catalog, **(two_tower_params or {})) if kind == "two_tower" else MODEL_CLASSES[kind](catalog)
        model.fit(fs)
        reg.register(model, {})
        log.info("trained %s", kind)
    return run_dir


def evaluate_models(settings: Settings, run_id: str = "latest") -> dict:
    run_dir = settings.run_dir(run_id)
    fs = load_features(run_dir / "features")
    models = _load_trained(run_dir)
    report = {
        "splits": fs.splits.as_dict(),
        "split_sizes": {"train": len(fs.train), "val": len(fs.val), "test": len(fs.test)},
        "models": {kind: evaluate(m, fs.test) for kind, m in models.items()},
    }
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def simulate(settings: Settings, run_id: str = "latest") -> dict:
    run_dir = settings.run_dir(run_id)
    world = World.load(run_dir / "raw" / "world.json")
    report = simulate_ab(world, _load_trained(run_dir), control="popularity", n_sessions=settings.sim_sessions)
    (run_dir / "online.json").write_text(json.dumps(report, indent=2))
    return report


def register(settings: Settings, run_id: str = "latest", force: bool = False) -> dict:
    """Upload every trained model with its test metrics; promote the best one."""
    run_dir = settings.run_dir(run_id)
    report = json.loads((run_dir / "metrics.json").read_text())
    models = _load_trained(run_dir)
    registry = ModelRegistry(get_store(settings), settings.registry_name)
    versions = {}
    for kind, model in models.items():
        versions[kind] = registry.register(model, report["models"][kind], {"run_id": run_id, "splits": report["splits"]})
    best = max(models, key=lambda k: report["models"][k][PROMOTION_METRIC])
    promoted = registry.promote(best, versions[best], report["models"][best], force=force)
    result = {"versions": versions, "best": best, "promoted": promoted}
    log.info("registered %s; best=%s promoted=%s", versions, best, promoted)
    return result


def run_all(settings: Settings, run_id: str = "latest") -> dict:
    generate(settings, run_id)
    features(settings, run_id)
    train(settings, run_id)
    metrics = evaluate_models(settings, run_id)
    online = simulate(settings, run_id)
    reg = register(settings, run_id)
    return {"offline": metrics, "online": online, "registry": reg}
