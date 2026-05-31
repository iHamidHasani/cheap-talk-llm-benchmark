# Cheap-Talk LLM Benchmark

A pre-specified Crawford–Sobel benchmark for measuring **LLM honesty under preference misalignment**. An LLM sender observes a state, has an objective biased away from the user's, and sends one message; an oracle computes the most-informative cheap-talk equilibrium as ground truth. (Design frozen before data collection; not externally timestamped, hence "pre-specified" rather than "pre-registered".)

Paper (working title): *Truthful AI Advisors: A Pre-Registered Benchmark for LLM Honesty Under Preference Misalignment.*

## Layout

```
cheap-talk-llm-benchmark/
├── cs_llm_benchmark/        # the experiment + analysis package
│   ├── config.py            # frozen, pre-specified design constants (Table 2)
│   ├── prompts.py           # sender + comprehension prompt templates (Appendix A)
│   ├── oracle.py            # Crawford–Sobel oracle (Alg 1, Prop 2, Table 4)
│   ├── senders.py           # provider adapters + collection loop (Alg 2)
│   ├── receiver.py          # hybrid receiver decoder: numeric read + embedding fallback (Alg 3)
│   ├── segmentation.py      # penalised monotone step-fit (Alg 4)
│   ├── metrics.py           # per-cell estimands + evaluation (Alg 5)
│   ├── analysis.py          # hypothesis tests, regressions, bootstrap
│   ├── validity.py          # Table 3 validity diagnostics
│   ├── run_experiment.py    # driver: collect 12,000 messages + 60 comprehension
│   ├── analyze_results.py   # driver: regenerate every empirical table
│   └── run.sh               # end-to-end orchestration
└── paper/
    └── main.tex             # manuscript (master copy; Overleaf is downstream)
```

## Quick start

```bash
# Offline pipeline check — no API calls, synthetic sender
PROVIDER_SET=stub ./cs_llm_benchmark/run.sh

# Full run — requires API keys (see below)
./cs_llm_benchmark/run.sh
```

## Required environment

The default 4-model basket (`config.DEFAULT_MODELS`) needs:

| Provider  | Env var             | Models |
|-----------|---------------------|--------|
| OpenAI    | `OPENAI_API_KEY`    | gpt-4o |
| Anthropic | `ANTHROPIC_API_KEY` | claude-sonnet |
| Google    | `GOOGLE_API_KEY`    | gemini-pro |
| Together  | `TOGETHER_API_KEY`  | llama-3.1-70b (OpenAI-compatible base_url) |

Estimated cost for the full 12,000-call run with the default basket: **≈ $4–10** (output is ~20 tokens/call). Swap to an all-budget basket (4o-mini + Haiku + Flash + 70B) for **≈ $1**.

## Design (frozen / pre-specified)

- 4 models × 5 biases {0, 0.01, 0.04, 0.08, 0.12} × 3 frames {neutral, payoff, honesty} × 200 states = **12,000 sender calls**
- 60 comprehension diagnostics (4 × 5 × 3)
- Decoding: temperature 0, seed locked, max 64 tokens
- Receiver decoder: **hybrid** — reads the sender's stated number when present (the estimand is "what a numerate receiver can infer"), falls back to ridge on frozen MiniLM embeddings for non-numeric prose; 5-fold cross-fitted. A pure-embedding decoder is retained as an ablation (`--regressor knn` / embedding-only), which mis-reads numeric messages and is reported for transparency.
- Default model basket: GPT-4o, Claude Sonnet 4.5, Gemini 2.5 Flash-Lite, Llama-3.3-70B (see `config.BASKET_Y`).
- Supplementary reviewer stats (decoder transparency, linear-exaggeration fit, state-clustered bootstrap, H2 CI, ex-Llama): `python -m cs_llm_benchmark.extra_stats --input_dir results_preview/full4model`.
- Over-revelation test, monotonicity regression, payoff-vs-honesty contrast

## Reproducibility

Every query is logged to `messages.jsonl` with prompt, raw output, parsed message, seed, model version, decoding params, parser status. The analysis driver regenerates **every table** from disk with no additional model calls. Collection is **idempotent** — re-running resumes from where it stopped.

## Workflow

This GitHub repo is the **source of truth** for both code and `paper/main.tex`. The Overleaf project is a downstream mirror of the tex only — edit `paper/main.tex` here, commit, then sync to Overleaf.
