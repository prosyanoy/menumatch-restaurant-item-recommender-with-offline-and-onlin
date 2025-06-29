"""Simulated online A/B/n test.

Fresh checkout sessions are drawn from the *ground-truth* world for the days
right after the data window (so the trend keeps moving). Sessions are randomly
assigned to arms; each arm shows its top-K add-ons and the simulated customer
accepts item j with the true propensity ``World.addon_probs``. Reported:
conversion (>=1 accepted add-on), add-on revenue per session, lift vs. control
with a two-proportion z-test and a Welch CI on revenue.
"""
from __future__ import annotations

import math

import numpy as np

from .context import daypart_of
from .generator import START_DATE, World
from .models import Batch, Recommender


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def simulate_ab(world: World, arms: dict[str, Recommender], control: str, n_sessions: int,
                k: int = 3, seed: int = 123, sim_days: int = 14) -> dict:
    rng = np.random.default_rng(seed)
    names = list(arms)
    price = world.items.price.to_numpy()
    sessions = []
    for _ in range(n_sessions):
        day = world.n_days + int(rng.integers(0, sim_days))
        restaurant, hour = world.sample_context(rng)
        dp = daypart_of(hour)
        progress = day / world.n_days
        weekend = int((START_DATE.weekday() + day) % 7 >= 5)
        mains = world.sample_mains(rng, restaurant, dp, progress)
        probs = world.addon_probs(restaurant, mains, dp, weekend, progress)
        sessions.append((restaurant, mains, dp, weekend, probs))
    arm_of = rng.integers(0, len(names), size=n_sessions)  # randomised assignment
    batch = Batch(
        restaurant=np.array([s[0] for s in sessions]),
        carts=[np.array(s[1], dtype=np.int64) for s in sessions],
        daypart=np.array([s[2] for s in sessions]),
        weekend=np.array([s[3] for s in sessions]),
    )
    draws = rng.random((n_sessions, k))  # common random numbers across arms
    stats = {}
    for a, name in enumerate(names):
        rows = np.flatnonzero(arm_of == a)
        recs = arms[name].recommend(batch.take(rows), k)
        accepted_rev = np.zeros(len(rows))
        converted = np.zeros(len(rows))
        for j, row in enumerate(rows):
            for pos in range(k):
                item = recs[j, pos]
                if item >= 0 and draws[row, pos] < sessions[row][4][item]:
                    accepted_rev[j] += price[item]
                    converted[j] = 1.0
        stats[name] = {"conv": converted, "rev": accepted_rev}

    report = {"k": k, "n_sessions": n_sessions, "control": control, "arms": {}}
    c = stats[control]
    for name, s in stats.items():
        n = len(s["conv"])
        p = s["conv"].mean()
        arm = {
            "sessions": n,
            "conversion": float(p),
            "revenue_per_session": float(s["rev"].mean()),
        }
        if name != control:
            n_c, p_c = len(c["conv"]), c["conv"].mean()
            pooled = (s["conv"].sum() + c["conv"].sum()) / (n + n_c)
            se = math.sqrt(pooled * (1 - pooled) * (1 / n + 1 / n_c)) or 1e-12
            z = (p - p_c) / se
            d_rev = s["rev"].mean() - c["rev"].mean()
            se_rev = math.sqrt(s["rev"].var(ddof=1) / n + c["rev"].var(ddof=1) / n_c)
            arm.update({
                "conversion_lift_rel": float((p - p_c) / p_c) if p_c else None,
                "conversion_p_value": float(2 * _norm_sf(abs(z))),
                "revenue_diff": float(d_rev),
                "revenue_diff_ci95": [float(d_rev - 1.96 * se_rev), float(d_rev + 1.96 * se_rev)],
                "revenue_lift_rel": float(d_rev / c["rev"].mean()) if c["rev"].mean() else None,
            })
        report["arms"][name] = arm
    return report
