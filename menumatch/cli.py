"""``python -m menumatch.cli <stage> [--run-id ID]``"""
from __future__ import annotations

import argparse
import json
import logging

from . import pipeline
from .config import Settings

STAGES = {
    "generate": pipeline.generate,
    "features": pipeline.features,
    "train": pipeline.train,
    "evaluate": pipeline.evaluate_models,
    "simulate": pipeline.simulate,
    "register": pipeline.register,
    "all": pipeline.run_all,
}


def format_offline(report: dict) -> str:
    cols = ["recall@1", "recall@5", "ndcg@5", "ndcg@10", "coverage@5"]
    lines = [f"{'model':<14}" + "".join(f"{c:>12}" for c in cols)]
    for kind, m in report["models"].items():
        lines.append(f"{kind:<14}" + "".join(f"{m[c]:>12.4f}" for c in cols))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="menumatch")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--run-id", default="latest")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    result = STAGES[args.stage](Settings(), args.run_id)
    if args.stage in ("evaluate",):
        print(format_offline(result))
    elif args.stage == "simulate":
        print(json.dumps(result["arms"], indent=2))
    elif args.stage == "all":
        print(format_offline(result["offline"]))
        print(json.dumps(result["online"]["arms"], indent=2))
        print(json.dumps(result["registry"], indent=2))


if __name__ == "__main__":
    main()
