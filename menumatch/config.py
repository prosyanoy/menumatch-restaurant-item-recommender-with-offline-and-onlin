"""Configuration read from environment variables with safe local defaults."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("MENUMATCH_DATA_DIR", "./data")))
    seed: int = field(default_factory=lambda: int(_env("MENUMATCH_SEED", "7")))
    n_days: int = field(default_factory=lambda: int(_env("MENUMATCH_N_DAYS", "120")))
    orders_per_day: int = field(default_factory=lambda: int(_env("MENUMATCH_ORDERS_PER_DAY", "300")))
    n_restaurants: int = field(default_factory=lambda: int(_env("MENUMATCH_N_RESTAURANTS", "5")))
    train_frac: float = 0.7
    val_frac: float = 0.1
    store: str = field(default_factory=lambda: _env("MENUMATCH_STORE", "local"))
    local_store_dir: Path = field(default_factory=lambda: Path(_env("MENUMATCH_LOCAL_STORE_DIR", "./artifacts")))
    s3_bucket: str = field(default_factory=lambda: _env("MENUMATCH_S3_BUCKET", "menumatch-artifacts"))
    s3_endpoint: str | None = field(default_factory=lambda: os.environ.get("MENUMATCH_S3_ENDPOINT") or None)
    registry_name: str = field(default_factory=lambda: _env("MENUMATCH_REGISTRY_NAME", "menumatch"))
    sim_sessions: int = field(default_factory=lambda: int(_env("MENUMATCH_SIM_SESSIONS", "30000")))

    def run_dir(self, run_id: str = "latest") -> Path:
        return self.data_dir / "runs" / sanitize_run_id(run_id)


def sanitize_run_id(run_id: str) -> str:
    """Airflow run ids contain ':' and '+'; make them filesystem/S3-key safe."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
