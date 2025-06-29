"""Measure end-to-end /recommend latency over HTTP against a running server.

    python scripts/bench_latency.py --url http://localhost:8000 --n 2000
"""
import argparse
import json
import random
import statistics
import time
import urllib.request


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=100)
    args = ap.parse_args()
    rng = random.Random(0)
    mains = {}
    for rid in range(5):
        menu = json.load(urllib.request.urlopen(f"{args.url}/restaurants/{rid}/menu"))
        mains[rid] = [m["item_id"] for m in menu if m["category"] == "main"]

    def call() -> float:
        rid = rng.randrange(5)
        body = json.dumps({"restaurant_id": rid, "cart_item_ids": rng.sample(mains[rid], rng.randint(1, 2)),
                           "k": 10, "hour": rng.randrange(24)}).encode()
        req = urllib.request.Request(f"{args.url}/recommend", body, {"content-type": "application/json"})
        t = time.perf_counter()
        urllib.request.urlopen(req).read()
        return (time.perf_counter() - t) * 1000

    for _ in range(args.warmup):
        call()
    lat = sorted(call() for _ in range(args.n))
    q = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]
    print(json.dumps({"n": args.n, "p50_ms": round(q(0.5), 2), "p95_ms": round(q(0.95), 2),
                      "p99_ms": round(q(0.99), 2), "max_ms": round(lat[-1], 2),
                      "mean_ms": round(statistics.mean(lat), 2)}))


if __name__ == "__main__":
    main()
