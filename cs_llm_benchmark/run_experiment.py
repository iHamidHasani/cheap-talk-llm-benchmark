"""
Driver: collect 12,000 sender messages plus 60 comprehension diagnostics.

Usage:
    python -m cs_llm_benchmark.run_experiment \
        --output_dir runs/main --provider_set default

Pass `--provider_set stub` to run a fast offline pipeline check that does
not call any API.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from . import config, senders


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--provider_set",
                    choices=["default", "path_x", "path_y", "gpt4o", "others", "stub"],
                    default="default")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    if args.provider_set == "stub":
        models = tuple(config.ModelSpec(n, "stub", n) for n in
                       ("stub-A", "stub-B", "stub-C", "stub-D"))
    else:
        models = config.BASKETS[args.provider_set]

    states = senders.seeded_states()

    plan = senders.CollectionPlan(
        models=list(models),
        biases=config.BIASES,
        frames=config.FRAMES,
        states=states,
        output_path=args.output_dir / "messages.jsonl",
        max_retries=args.retries,
    )

    print(f"[collect] models={len(models)} biases={len(config.BIASES)} "
          f"frames={len(config.FRAMES)} states={len(states)} "
          f"=> {len(models) * len(config.BIASES) * len(config.FRAMES) * len(states)} calls")
    senders.run_collection(plan)

    print(f"[comprehension] writing 60 diagnostic prompts")
    senders.run_comprehension(models, args.output_dir / "comprehension.jsonl",
                              max_retries=args.retries)
    print("[done]")


if __name__ == "__main__":
    main()
