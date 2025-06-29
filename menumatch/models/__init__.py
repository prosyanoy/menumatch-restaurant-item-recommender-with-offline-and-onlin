from .base import Batch, Catalog, Recommender, candidate_mask
from .cooccurrence import CooccurrenceRecommender
from .popularity import PopularityRecommender
from .twotower import TwoTowerRecommender

MODEL_CLASSES = {
    cls.kind: cls for cls in (PopularityRecommender, CooccurrenceRecommender, TwoTowerRecommender)
}

__all__ = [
    "Batch", "Catalog", "Recommender", "candidate_mask", "MODEL_CLASSES",
    "PopularityRecommender", "CooccurrenceRecommender", "TwoTowerRecommender",
]
