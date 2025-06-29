"""FastAPI service: POST /recommend returns add-on suggestions for a cart."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ..config import Settings
from ..context import daypart_of
from ..models import Batch
from ..registry import LoadedModel, ModelRegistry
from ..store import get_store


class RecommendRequest(BaseModel):
    restaurant_id: int
    cart_item_ids: list[int] = Field(default_factory=list)
    k: int = Field(3, ge=1, le=20)
    hour: int | None = Field(None, ge=0, le=23, description="Defaults to the current hour")
    is_weekend: bool | None = Field(None, description="Defaults to today")


class Recommendation(BaseModel):
    item_id: int
    name: str
    category: str
    price: float
    score: float


class RecommendResponse(BaseModel):
    restaurant_id: int
    model_kind: str
    model_version: str
    recommendations: list[Recommendation]


def create_app(loaded: LoadedModel | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the app. ``loaded`` can be injected (tests); otherwise the production
    model is pulled from the registry at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.loaded = loaded
        if app.state.loaded is None:
            cfg = settings or Settings()
            try:
                app.state.loaded = ModelRegistry(get_store(cfg), cfg.registry_name).load_production()
            except Exception:  # store unreachable -> serve 503 until a reload succeeds
                app.state.loaded = None
        yield

    app = FastAPI(title="MenuMatch", version="0.1.0", lifespan=lifespan)

    def get_loaded(request: Request) -> LoadedModel:
        if request.app.state.loaded is None:
            raise HTTPException(503, "No production model registered yet; run the pipeline first.")
        return request.app.state.loaded

    @app.get("/health")
    def health(request: Request) -> dict:
        m = request.app.state.loaded
        return {"status": "ok" if m else "no_model",
                "model_kind": m.kind if m else None, "model_version": m.version if m else None}

    @app.post("/reload")
    def reload(request: Request) -> dict:
        cfg = settings or Settings()
        request.app.state.loaded = ModelRegistry(get_store(cfg), cfg.registry_name).load_production()
        return health(request)

    @app.get("/restaurants/{restaurant_id}/menu")
    def menu(restaurant_id: int, request: Request) -> list[dict]:
        items = get_loaded(request).catalog.items
        sub = items[items.restaurant_id == restaurant_id]
        if sub.empty:
            raise HTTPException(404, f"Unknown restaurant {restaurant_id}")
        return sub[["item_id", "name", "category", "price"]].to_dict(orient="records")

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(req: RecommendRequest, request: Request) -> RecommendResponse:
        m = get_loaded(request)
        cat = m.catalog
        if req.restaurant_id not in set(cat.restaurant.tolist()):
            raise HTTPException(404, f"Unknown restaurant {req.restaurant_id}")
        cart = sorted(set(req.cart_item_ids))
        bad = [i for i in cart if not (0 <= i < cat.n_items) or cat.restaurant[i] != req.restaurant_id]
        if bad:
            raise HTTPException(422, f"Items {bad} are not on the menu of restaurant {req.restaurant_id}")
        now = datetime.now()
        hour = now.hour if req.hour is None else req.hour
        weekend = int(now.weekday() >= 5) if req.is_weekend is None else int(req.is_weekend)
        batch = Batch(np.array([req.restaurant_id]), [np.array(cart, dtype=np.int64)],
                      np.array([daypart_of(hour)]), np.array([weekend]))
        ids = m.model.recommend(batch, req.k)[0]
        scores = m.model.score(batch)[0]
        rows = cat.items.set_index("item_id")
        recs = [
            Recommendation(item_id=int(i), name=rows.at[i, "name"], category=rows.at[i, "category"],
                           price=float(rows.at[i, "price"]), score=float(scores[i]))
            for i in ids if i >= 0
        ]
        return RecommendResponse(restaurant_id=req.restaurant_id, model_kind=m.kind,
                                 model_version=m.version, recommendations=recs)

    return app


app = create_app()
