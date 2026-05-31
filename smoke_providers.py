"""One-call-per-provider smoke test. Confirms every key works and every model
ID in BASKET_Y resolves, BEFORE the 12,000-call run. Cost: ~4 tiny calls.

Run from the repo root after .env is filled:  python smoke_providers.py
"""
from __future__ import annotations
import os, sys, pathlib

# Load .env (strip surrounding whitespace from keys/values).
env_path = pathlib.Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

from cs_llm_benchmark import config, senders, prompts

PROMPT = prompts.render("neutral", omega=0.42, bias=0.04)

ok = True
for spec in config.BASKET_Y:
    try:
        fn = senders.PROVIDERS[spec.provider]
        out = fn(spec, PROMPT)
        status = "OK " if out.strip() else "EMPTY"
        print(f"[{status}] {spec.name:14s} ({spec.provider}/{spec.api_model})")
        print(f"        -> {out.strip()[:80]!r}")
    except Exception as e:
        ok = False
        print(f"[FAIL] {spec.name:14s} ({spec.provider}/{spec.api_model})")
        print(f"        -> {type(e).__name__}: {str(e)[:160]}")

print("\nALL GOOD — clear to launch 12K." if ok else
      "\nFIX the FAILED providers before launching.")
sys.exit(0 if ok else 1)
