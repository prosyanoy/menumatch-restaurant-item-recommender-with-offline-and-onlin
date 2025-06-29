"""Seeded synthetic restaurant order generator.

A ``World`` holds the hidden ground truth: per-item latent vectors, popularity
biases, time-of-day preferences and a linear popularity trend. Orders are a
set of mains plus at most one add-on per category (side/drink/dessert); the
add-on is chosen by a softmax over (latent affinity with the mains in the cart
+ item bias + daypart preference + trend). The same ``addon_probs`` function is
used by the online simulation as the "true" user propensity.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .context import ADDON_CATEGORIES, CATEGORIES, daypart_of

START_DATE = datetime(2025, 1, 6)  # a Monday
LATENT_DIM = 4
AFFINITY_BETA = 2.0

MENU_TEMPLATES = {
    "Burger Barn": {
        "main": ["Classic Burger", "Cheeseburger", "Bacon Burger", "Veggie Burger", "Chicken Sandwich", "Breakfast Burger"],
        "side": ["Fries", "Onion Rings", "Coleslaw", "Sweet Potato Fries"],
        "drink": ["Cola", "Lemonade", "Chocolate Shake", "Iced Tea"],
        "dessert": ["Brownie", "Apple Pie", "Ice Cream Cup"],
    },
    "Slice House": {
        "main": ["Margherita", "Pepperoni", "Quattro Formaggi", "Veggie Pizza", "Calzone", "Hawaiian"],
        "side": ["Garlic Bread", "Caesar Salad", "Mozzarella Sticks", "Wings"],
        "drink": ["Cola", "Sparkling Water", "Red Wine", "Craft Beer"],
        "dessert": ["Tiramisu", "Cannoli", "Gelato"],
    },
    "Taco Libre": {
        "main": ["Carne Asada Burrito", "Chicken Tacos", "Fish Tacos", "Veggie Bowl", "Quesadilla", "Breakfast Burrito"],
        "side": ["Chips & Salsa", "Guacamole", "Elote", "Refried Beans"],
        "drink": ["Horchata", "Jarritos", "Agua Fresca", "Margarita"],
        "dessert": ["Churros", "Flan", "Tres Leches"],
    },
    "Sakura Sushi": {
        "main": ["Salmon Roll", "Dragon Roll", "Chirashi Bowl", "Ramen", "Teriyaki Chicken", "Tempura Set"],
        "side": ["Edamame", "Miso Soup", "Gyoza", "Seaweed Salad"],
        "drink": ["Green Tea", "Ramune", "Sake", "Iced Matcha"],
        "dessert": ["Mochi", "Dorayaki", "Matcha Ice Cream"],
    },
    "Corner Cafe": {
        "main": ["Avocado Toast", "Club Sandwich", "Chicken Wrap", "Quiche", "Grain Bowl", "Pancake Stack"],
        "side": ["Side Salad", "Soup of the Day", "Hash Browns", "Fruit Cup"],
        "drink": ["Latte", "Cold Brew", "Orange Juice", "Chai"],
        "dessert": ["Croissant", "Cheesecake", "Cookie"],
    },
}

# P(order includes an add-on of this category | daypart); weekend multipliers below.
ADDON_TAKE = {
    "side": [0.25, 0.55, 0.35, 0.55, 0.50],
    "drink": [0.55, 0.65, 0.45, 0.60, 0.50],
    "dessert": [0.05, 0.12, 0.15, 0.30, 0.20],
}
WEEKEND_MULT = {"side": 1.0, "drink": 1.1, "dessert": 1.4}
# Hour-of-day mixture: (weight, mean hour, std)
HOUR_MIX = [(0.12, 8.0, 1.0), (0.33, 12.5, 1.0), (0.10, 15.5, 1.0), (0.38, 19.0, 1.3), (0.07, 22.5, 1.0)]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


@dataclass
class World:
    seed: int
    n_days: int
    items: pd.DataFrame  # item_id, restaurant_id, restaurant_name, name, category, price
    z: np.ndarray  # (N, d) latent vectors
    bias: np.ndarray  # (N,)
    daypart_pref: np.ndarray  # (N, 5)
    trend: np.ndarray  # (N,)
    restaurant_weights: np.ndarray  # (R,)

    def __post_init__(self) -> None:
        # item ids per (restaurant, category), precomputed: this is on the hot path
        self._by_rc = {
            (int(r), c): g.item_id.to_numpy()
            for (r, c), g in self.items.groupby(["restaurant_id", "category"])
        }

    # -- construction ---------------------------------------------------
    @classmethod
    def build(cls, seed: int, n_days: int, n_restaurants: int = 5) -> "World":
        rng = np.random.default_rng(seed)
        names = list(MENU_TEMPLATES)[:n_restaurants]
        rows = []
        for rid, rname in enumerate(names):
            for cat in CATEGORIES:
                price_lo, price_hi = {"main": (9, 17), "side": (3, 7), "drink": (2, 6), "dessert": (4, 8)}[cat]
                for iname in MENU_TEMPLATES[rname][cat]:
                    price = round(float(rng.uniform(price_lo, price_hi)) * 4) / 4
                    rows.append((rid, rname, iname, cat, price))
        items = pd.DataFrame(rows, columns=["restaurant_id", "restaurant_name", "name", "category", "price"])
        items.insert(0, "item_id", np.arange(len(items)))
        n = len(items)
        return cls(
            seed=seed,
            n_days=n_days,
            items=items,
            z=rng.normal(size=(n, LATENT_DIM)),
            bias=rng.normal(scale=0.5, size=n),
            daypart_pref=rng.normal(scale=0.5, size=(n, 5)),
            trend=rng.uniform(-0.8, 0.8, size=n),
            restaurant_weights=_softmax(rng.normal(scale=0.5, size=len(names))),
        )

    # -- ground-truth behaviour ----------------------------------------
    def _logits(self, idx: np.ndarray, dp: int, progress: float) -> np.ndarray:
        return self.bias[idx] + self.daypart_pref[idx, dp] + self.trend[idx] * progress

    def sample_mains(self, rng: np.random.Generator, restaurant: int, dp: int, progress: float) -> list[int]:
        mains = self._by_rc[(restaurant, "main")]
        # mains react more strongly to time of day (e.g. breakfast items)
        logits = self.bias[mains] + 1.6 * self.daypart_pref[mains, dp] + self.trend[mains] * progress
        n_main = int(rng.choice([1, 2, 3], p=[0.65, 0.28, 0.07]))
        n_main = min(n_main, len(mains))
        return sorted(int(m) for m in rng.choice(mains, size=n_main, replace=False, p=_softmax(logits)))

    def addon_probs(self, restaurant: int, mains: list[int], dp: int, weekend: int, progress: float) -> np.ndarray:
        """P(customer adds item j as the add-on of its category), length N, 0 for non add-ons."""
        out = np.zeros(len(self.items))
        zm = self.z[mains]
        for cat in ADDON_CATEGORIES:
            idx = self._by_rc[(restaurant, cat)]
            affinity = AFFINITY_BETA * (self.z[idx] @ zm.T).mean(axis=1) / np.sqrt(LATENT_DIM)
            p_cat = min(0.95, ADDON_TAKE[cat][dp] * (WEEKEND_MULT[cat] if weekend else 1.0))
            out[idx] = p_cat * _softmax(affinity + self._logits(idx, dp, progress))
        return out

    def sample_context(self, rng: np.random.Generator) -> tuple[int, int]:
        """Draw (restaurant, hour) for one order."""
        restaurant = int(rng.choice(len(self.restaurant_weights), p=self.restaurant_weights))
        comp = rng.choice(len(HOUR_MIX), p=[w for w, _, _ in HOUR_MIX])
        _, mu, sd = HOUR_MIX[comp]
        return restaurant, int(round(rng.normal(mu, sd))) % 24

    # -- (de)serialisation ---------------------------------------------
    def save(self, path: Path) -> None:
        payload = {
            "seed": self.seed,
            "n_days": self.n_days,
            "items": self.items.to_dict(orient="list"),
            "z": self.z.tolist(),
            "bias": self.bias.tolist(),
            "daypart_pref": self.daypart_pref.tolist(),
            "trend": self.trend.tolist(),
            "restaurant_weights": self.restaurant_weights.tolist(),
        }
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> "World":
        p = json.loads(Path(path).read_text())
        return cls(
            seed=p["seed"],
            n_days=p["n_days"],
            items=pd.DataFrame(p["items"]),
            z=np.array(p["z"]),
            bias=np.array(p["bias"]),
            daypart_pref=np.array(p["daypart_pref"]),
            trend=np.array(p["trend"]),
            restaurant_weights=np.array(p["restaurant_weights"]),
        )


def generate_orders(world: World, orders_per_day: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate ``world.n_days`` of orders. Returns (orders, order_items)."""
    rng = np.random.default_rng(seed + 1)
    order_rows, item_rows = [], []
    order_id = 0
    for day in range(world.n_days):
        date = START_DATE + timedelta(days=day)
        weekend = int(date.weekday() >= 5)
        progress = day / world.n_days
        for _ in range(int(rng.poisson(orders_per_day * (1.25 if weekend else 1.0)))):
            restaurant, hour = world.sample_context(rng)
            dp = daypart_of(hour)
            ts = date + timedelta(hours=hour, minutes=int(rng.integers(0, 60)), seconds=int(rng.integers(0, 60)))
            mains = world.sample_mains(rng, restaurant, dp, progress)
            basket = {m: 1 + int(rng.random() < 0.1) for m in mains}
            probs = world.addon_probs(restaurant, mains, dp, weekend, progress)
            for cat in ADDON_CATEGORIES:
                idx = world._by_rc[(restaurant, cat)]
                p_cat = probs[idx].sum()
                if rng.random() < p_cat:
                    basket[int(rng.choice(idx, p=probs[idx] / p_cat))] = 1 + int(rng.random() < 0.05)
            order_rows.append((order_id, restaurant, ts))
            item_rows.extend((order_id, item, qty) for item, qty in sorted(basket.items()))
            order_id += 1
    orders = pd.DataFrame(order_rows, columns=["order_id", "restaurant_id", "ts"])
    order_items = pd.DataFrame(item_rows, columns=["order_id", "item_id", "quantity"])
    return orders, order_items


def generate_to_dir(out_dir: Path, seed: int, n_days: int, orders_per_day: int, n_restaurants: int = 5) -> World:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    world = World.build(seed, n_days, n_restaurants)
    orders, order_items = generate_orders(world, orders_per_day, seed)
    orders.to_parquet(out_dir / "orders.parquet", index=False)
    order_items.to_parquet(out_dir / "order_items.parquet", index=False)
    world.items.to_parquet(out_dir / "items.parquet", index=False)
    world.save(out_dir / "world.json")
    return world
