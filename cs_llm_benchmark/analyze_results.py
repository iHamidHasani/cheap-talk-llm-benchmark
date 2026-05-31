"""
Driver: read messages.jsonl + comprehension.jsonl, run Algorithms 3-5 and
the hypothesis tests, and write every empirical table required by Section 7.

Outputs (CSV / JSON) into <output_dir>:
    cell_metrics.csv         one row per (model, bias, frame)
    per_row_predictions.csv  one row per query: omega, message, fold, a_hat
    bias_table.csv           5-row bias summary
    frame_contrasts_*.csv    payoff-vs-honesty contrasts
    model_summary.csv        4-row model summary
    robustness.csv           B=10, B=30, alternative decoder, drop-invalid
    validity.json            Table 3 diagnostics
    regressions.json         beta, SE, t for I_norm and N_hat
    reference_table.csv      Table 4 (oracle, regenerated)
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import (analysis, config, metrics, oracle, receiver, validity)


# --------------------------------------------------------------------------- #
# Helpers.                                                                     #
# --------------------------------------------------------------------------- #
def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    # Union of keys across rows (some rows may have extra fields like CIs).
    fields: List[str] = []
    seen: set = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


# --------------------------------------------------------------------------- #
# Core: build CellMetrics for every (model, bias, frame) and the per-row log.  #
# --------------------------------------------------------------------------- #
def build_cell_metrics(records: List[dict],
                       embedder=None,
                       drop_invalid: bool = False,
                       mi_bins_override: int | None = None,
                       regressor: str = "hybrid",
                       n_boot: int = 0,
                       collect_predictions: bool = False,
                       ) -> Tuple[List[dict], List[dict]]:
    if embedder is None:
        embedder = receiver.default_embedder()
    rows: List[dict] = []
    preds: List[dict] = []

    keyed: Dict[tuple, list] = defaultdict(list)
    for r in records:
        if drop_invalid and (r["parser_status"] != "ok" or r["api_error"]):
            continue
        keyed[(r["model"], r["bias"], r["frame"])].append(r)

    bins = mi_bins_override if mi_bins_override is not None else config.MI_BINS
    for (model, bias, frame), recs in sorted(keyed.items()):
        messages = [r["message"] for r in recs]
        omegas   = np.array([r["omega"] for r in recs], dtype=float)
        a_hat, fold_id = receiver.cross_fit_actions(
            messages, omegas, embedder=embedder,
            folds=config.RECEIVER_FOLDS, regressor=regressor)
        m = metrics.evaluate_cell(model, bias, frame, omegas, a_hat,
                                  n_boot=n_boot, bins=bins)
        row = metrics.cell_to_row(m)
        if mi_bins_override is not None:
            row["I_norm_bins"] = mi_bins_override
        if regressor != "ridge":
            row["regressor"] = regressor
        rows.append(row)

        if collect_predictions:
            for r, ah, fid in zip(recs, a_hat, fold_id):
                preds.append({
                    "model":    r["model"], "bias": r["bias"], "frame": r["frame"],
                    "state_index": r["state_index"], "omega": r["omega"],
                    "message":  r["message"],
                    "parser_status": r["parser_status"],
                    "fold":     int(fid),
                    "a_hat":    float(ah),
                    "a_hat_clipped": float(np.clip(ah, 0.0, 1.0)),
                })
    return rows, preds


# --------------------------------------------------------------------------- #
# Aggregations.                                                                #
# --------------------------------------------------------------------------- #
def bias_table(rows: List[dict]) -> List[dict]:
    out = []
    for b in sorted({r["bias"] for r in rows}):
        sub = [r for r in rows if r["bias"] == b]
        out.append({
            "bias":   b,
            "n_cells": len(sub),
            "mean_I_norm": float(np.mean([r["I_norm"] for r in sub])),
            "mean_N_hat":  float(np.mean([r["N_hat"] for r in sub])),
            "N_CS":        oracle.n_cs(b) if b > 0 else "Full",
            "I_norm_CS":   oracle.oracle_normalized_mi(b),
            "mean_L_R":    float(np.mean([r["L_R"] for r in sub])),
            "mean_L_S":    float(np.mean([r["L_S"] for r in sub])),
            "L_R_CS":      oracle.oracle_loss_receiver(b),
            "L_S_CS":      oracle.oracle_loss_sender(b),
            "frac_overreveal_N":  float(np.mean([r["over_reveal_N"]  for r in sub])),
            "frac_overreveal_MI": float(np.mean([r["over_reveal_mi"] for r in sub])),
        })
    return out


def model_summary(rows: List[dict]) -> List[dict]:
    out = []
    for m in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == m]
        slope_I = analysis.slope_on_bias(sub, "I_norm")
        slope_N = analysis.slope_on_bias(sub, "N_hat")
        out.append({
            "model": m,
            "n_cells": len(sub),
            "mean_I_norm": float(np.mean([r["I_norm"] for r in sub])),
            "mean_N_hat":  float(np.mean([r["N_hat"] for r in sub])),
            "slope_I_on_b": slope_I["slope"],
            "slope_N_on_b": slope_N["slope"],
            "frac_overreveal_N": float(np.mean([r["over_reveal_N"] for r in sub])),
        })
    return out


# --------------------------------------------------------------------------- #
# Robustness: B=10, B=30, alternative decoder, drop invalid.                   #
# --------------------------------------------------------------------------- #
def build_robustness(records: List[dict], embedder) -> List[dict]:
    robustness: List[dict] = []
    for B in (10, 30):
        cells_b, _ = build_cell_metrics(records, embedder=embedder,
                                        mi_bins_override=B)
        for row in bias_table(cells_b):
            row["spec"] = f"MI_bins={B}"
            robustness.append(row)
    # Alternative decoder: k-NN regressor on the same embeddings.
    cells_alt, _ = build_cell_metrics(records, embedder=embedder,
                                      regressor="knn")
    for row in bias_table(cells_alt):
        row["spec"] = "alt_decoder=knn"
        robustness.append(row)
    # Ablation: pure-embedding ridge (no numeric reading). Documents why the
    # hybrid decoder is necessary — embeddings alone cannot read number strings
    # and fail the b=0 truthfulness check.
    cells_emb, _ = build_cell_metrics(records, embedder=embedder,
                                      regressor="ridge")
    for row in bias_table(cells_emb):
        row["spec"] = "ablation_embedding_only"
        robustness.append(row)
    # Drop invalid outputs.
    cells_drop, _ = build_cell_metrics(records, embedder=embedder,
                                       drop_invalid=True)
    for row in bias_table(cells_drop):
        row["spec"] = "drop_invalid"
        robustness.append(row)
    return robustness


# --------------------------------------------------------------------------- #
# Main.                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir",  type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--n_boot", type=int, default=config.BOOTSTRAP_SAMPLES,
                    help="Bootstrap resamples per cell for L_R/L_S/I_norm CIs.")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    msg_path  = args.input_dir / "messages.jsonl"
    comp_path = args.input_dir / "comprehension.jsonl"
    records = load_jsonl(msg_path)
    print(f"[load] {len(records)} sender records")

    embedder = receiver.default_embedder()

    # 1) Primary cell metrics with bootstrap CIs and per-row predictions log.
    cells, preds = build_cell_metrics(records, embedder=embedder,
                                      n_boot=args.n_boot,
                                      collect_predictions=True)
    write_csv(cells, args.output_dir / "cell_metrics.csv")
    write_csv(preds, args.output_dir / "per_row_predictions.csv")

    # 2) Aggregations.
    write_csv(bias_table(cells),    args.output_dir / "bias_table.csv")
    write_csv(model_summary(cells), args.output_dir / "model_summary.csv")

    # 3) Frame contrasts (payoff vs honesty) on positive biases only.
    positive = [c for c in cells if c["bias"] > 0]
    write_csv(analysis.frame_contrasts(positive, "I_norm"),
              args.output_dir / "frame_contrasts_I.csv")
    write_csv(analysis.frame_contrasts(positive, "N_hat"),
              args.output_dir / "frame_contrasts_N.csv")

    # 4) Fixed-effects regressions Y = alpha_M + gamma_p + beta b + eps.
    reg_I = analysis.fixed_effects_regression(positive, "I_norm")
    reg_N = analysis.fixed_effects_regression(positive, "N_hat")
    (args.output_dir / "regressions.json").write_text(json.dumps({
        "I_norm": reg_I.__dict__, "N_hat": reg_N.__dict__,
    }, indent=2, default=str))

    # 5) Robustness (4 rows per paper Section 7).
    write_csv(build_robustness(records, embedder),
              args.output_dir / "robustness.csv")

    # 6) Validity diagnostics (Table 3).
    diag = [validity.valid_output_rate(records),
            validity.empty_output_rate(records),
            validity.format_violation_rate(records)]
    if comp_path.exists():
        diag.append(validity.comprehension_pass_rate(load_jsonl(comp_path)))
    b0 = [r for r in records if r["bias"] == 0.0 and r["frame"] == "neutral"
          and r["parser_status"] == "ok" and not r["api_error"]]
    if b0:
        r2 = receiver.zero_bias_r2([r["message"] for r in b0],
                                   np.array([r["omega"] for r in b0]),
                                   embedder=embedder)
        diag.append(validity.receiver_r2_check(r2))
    (args.output_dir / "validity.json").write_text(
        json.dumps([d.__dict__ for d in diag], indent=2))

    # 7) Oracle reference (Table 4).
    write_csv(oracle.reference_table(config.BIASES),
              args.output_dir / "reference_table.csv")

    print(f"[done] wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
