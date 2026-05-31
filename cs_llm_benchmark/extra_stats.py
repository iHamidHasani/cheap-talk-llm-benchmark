"""
Reviewer-requested supplementary statistics (GPT-5.5 review, 2026-05):
  1. fraction of messages containing a parseable number (overall + per model)
  2. parsed-number-direct decoder informativeness by bias (vs hybrid/embedding/knn)
  3. linear-exaggeration fit: report = a + s*omega, by bias (pooled) and per model
  4. H1 slope of I_norm on bias with a STATE-CLUSTERED bootstrap CI
  5. H2 payoff-vs-honesty contrast: mean + 95% bootstrap CI
  6. over-revelation pooled by bias EXCLUDING llama

Run:  python -m cs_llm_benchmark.extra_stats --input_dir results_preview/full4model
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

from . import oracle, metrics

NUM = re.compile(r"[-+]?\d*\.\d+|\d+")

def parse_number(msg: str):
    m = NUM.findall(msg or "")
    if not m:
        return None
    try:
        v = float(m[0])
    except ValueError:
        return None
    return v

def load(input_dir: Path):
    return [json.loads(l) for l in (input_dir / "messages.jsonl").open() if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path, required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()
    recs = load(args.input_dir)
    rng = np.random.default_rng(20260531)

    biases = sorted({r["bias"] for r in recs})
    models = sorted({r["model"] for r in recs})

    # ---- 1. parseable-number fraction ----
    print("\n## 1. Fraction of messages with a parseable number")
    frac_all = np.mean([parse_number(r["message"]) is not None for r in recs])
    print(f"  overall: {frac_all:.3f}")
    for m in models:
        sub = [r for r in recs if r["model"] == m]
        f = np.mean([parse_number(r["message"]) is not None for r in sub])
        print(f"  {m:16s}: {f:.3f}")

    # ---- 2. parsed-number-direct decoder I_norm by bias (pooled) ----
    print("\n## 2. Parsed-number-direct decoder: I_norm by bias (pooled over models)")
    for b in biases:
        sub = [r for r in recs if r["bias"] == b]
        om = np.array([r["omega"] for r in sub])
        ah = np.array([parse_number(r["message"]) if parse_number(r["message"]) is not None
                       else 0.5 for r in sub])
        inorm = metrics.normalized_mi(om, ah, bins=20)
        print(f"  b={b:.2f}: parsed-direct I_norm={inorm:.3f}  (oracle {oracle.oracle_normalized_mi(b):.3f})")

    # ---- 3. linear-exaggeration fit report = a + s*omega ----
    def fit(sub):
        om = np.array([r["omega"] for r in sub])
        rep = np.array([parse_number(r["message"]) for r in sub], dtype=object)
        keep = np.array([v is not None for v in rep])
        om, rep = om[keep], rep[keep].astype(float)
        if len(om) < 5:
            return None
        s, a = np.polyfit(om, rep, 1)
        return s, a, len(om)

    print("\n## 3. Linear-exaggeration fit  report = intercept + slope*omega")
    print("   pooled by bias (all models):")
    for b in biases:
        r = fit([x for x in recs if x["bias"] == b])
        if r: print(f"  b={b:.2f}: slope={r[0]:.3f} intercept={r[1]:+.3f}  (intercept vs b: {r[1]-b:+.3f}) n={r[2]}")
    print("   per model (positive bias pooled):")
    for m in models:
        r = fit([x for x in recs if x["model"] == m and x["bias"] > 0])
        if r: print(f"  {m:16s}: slope={r[0]:.3f} intercept={r[1]:+.3f} n={r[2]}")

    # ---- helper: cell I_norm using parsed-direct decoder ----
    def cell_inorm(records):
        cells = {}
        keyed = defaultdict(list)
        for r in records:
            keyed[(r["model"], r["bias"], r["frame"])].append(r)
        for k, rs in keyed.items():
            om = np.array([r["omega"] for r in rs])
            ah = np.array([parse_number(r["message"]) if parse_number(r["message"]) is not None
                           else 0.5 for r in rs])
            cells[k] = metrics.normalized_mi(om, ah, bins=20)
        return cells

    # ---- 4. H1 slope of I_norm on bias with state-clustered bootstrap ----
    print("\n## 4. H1: slope of I_norm on bias, state-clustered bootstrap")
    pos = [r for r in recs if r["bias"] > 0]
    state_ids = sorted({r["state_index"] for r in pos})
    def slope_from(records):
        cells = cell_inorm(records)
        xs = np.array([k[1] for k in cells]); ys = np.array(list(cells.values()))
        s, _ = np.polyfit(xs, ys, 1); return s
    point = slope_from(pos)
    boots = []
    by_state = defaultdict(list)
    for r in pos: by_state[r["state_index"]].append(r)
    for _ in range(args.n_boot):
        samp_states = rng.choice(state_ids, size=len(state_ids), replace=True)
        samp = []
        for s in samp_states: samp.extend(by_state[s])
        boots.append(slope_from(samp))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    print(f"  slope={point:.3f}  95% state-clustered bootstrap CI [{lo:.3f}, {hi:.3f}]")

    # ---- 5. H2 payoff - honesty contrast, bootstrap CI ----
    print("\n## 5. H2: payoff-minus-honesty contrast in I_norm, state-clustered bootstrap")
    def contrast_from(records):
        cells = cell_inorm(records)
        diffs = []
        for m in models:
            for b in biases:
                if b == 0: continue
                p = cells.get((m, b, "payoff")); h = cells.get((m, b, "honesty"))
                if p is not None and h is not None: diffs.append(p - h)
        return np.mean(diffs)
    cpoint = contrast_from(recs)
    cb = []
    by_state_all = defaultdict(list)
    for r in recs: by_state_all[r["state_index"]].append(r)
    all_states = sorted(by_state_all)
    for _ in range(args.n_boot):
        ss = rng.choice(all_states, size=len(all_states), replace=True)
        samp = []
        for s in ss: samp.extend(by_state_all[s])
        cb.append(contrast_from(samp))
    clo, chi = np.quantile(cb, [0.025, 0.975])
    print(f"  contrast={cpoint:+.3f}  95% CI [{clo:+.3f}, {chi:+.3f}]")

    # ---- 6. over-revelation excluding llama ----
    print("\n## 6. Pooled I_norm by bias EXCLUDING llama (parsed-direct)")
    noll = [r for r in recs if "llama" not in r["model"]]
    for b in biases:
        sub = [r for r in noll if r["bias"] == b]
        om = np.array([r["omega"] for r in sub])
        ah = np.array([parse_number(r["message"]) if parse_number(r["message"]) is not None else 0.5 for r in sub])
        print(f"  b={b:.2f}: I_norm(no-llama)={metrics.normalized_mi(om, ah, bins=20):.3f}  oracle {oracle.oracle_normalized_mi(b):.3f}")

if __name__ == "__main__":
    main()
