"""Shared definitions of categories and time-of-day context.

``daypart_of`` is mirrored in ``sql/orders_enriched.sql``; a test keeps them in sync.
"""
from __future__ import annotations

from datetime import datetime

CATEGORIES = ["main", "side", "drink", "dessert"]
ADDON_CATEGORIES = ["side", "drink", "dessert"]
DAYPARTS = ["breakfast", "lunch", "afternoon", "dinner", "late_night"]


def daypart_of(hour: int) -> int:
    """Map an hour of day (0-23) to a daypart index."""
    if 5 <= hour <= 10:
        return 0
    if 11 <= hour <= 14:
        return 1
    if 15 <= hour <= 17:
        return 2
    if 18 <= hour <= 21:
        return 3
    return 4


def is_weekend(ts: datetime) -> int:
    return int(ts.weekday() >= 5)
