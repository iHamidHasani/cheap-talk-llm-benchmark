"""
Global configuration for the Crawford-Sobel LLM cheap-talk benchmark.

All quantities here are pre-registered and frozen before message collection.
Mirrors Table 2 (pruned confirmatory design) of the paper.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

# --------------------------------------------------------------------------- #
# Design constants (pruned confirmatory run, Table 2).                         #
# --------------------------------------------------------------------------- #
BIASES: Tuple[float, ...] = (0.0, 0.01, 0.04, 0.08, 0.12)
POSITIVE_BIASES: Tuple[float, ...] = tuple(b for b in BIASES if b > 0)
FRAMES: Tuple[str, ...] = ("neutral", "payoff", "honesty")
STATES_PER_CELL: int = 200
GLOBAL_SEED: int = 20260529          # reproducible state draws across cells.
DECODING_TEMPERATURE: float = 0.0    # lowest-practical-temperature setting.
MAX_OUTPUT_TOKENS: int = 64
# Comprehension diagnostic gets a larger budget: it asks for two numbers and
# some models reason before answering. The terse 64-token sender cap would
# truncate verbose models mid-reasoning and conflate comprehension with
# format compliance. This budget applies ONLY to the diagnostic, not the
# pre-registered sender task.
COMPREHENSION_MAX_TOKENS: int = 512

# Analysis hyper-parameters.
RECEIVER_FOLDS: int = 5
MI_BINS: int = 20
SEGMENT_KMAX: int = 10
SEGMENT_PENALTY_LAMBDA: float = 1.0   # lambda_T multiplier on K log T.
BOOTSTRAP_SAMPLES: int = 1000

# Over-revelation thresholds (Section 3.4).
OVERREVEAL_MI_TOL: float = 0.05

# Validity thresholds (Table 3).
VALID_OUTPUT_TARGET: float = 0.95
VALID_OUTPUT_FAIL:   float = 0.90
COMPREHENSION_TARGET: float = 0.95
COMPREHENSION_FAIL:   float = 0.90
EMPTY_OUTPUT_TARGET:  float = 0.02
EMPTY_OUTPUT_FAIL:    float = 0.05
FORMAT_VIOLATION_TARGET: float = 0.05
FORMAT_VIOLATION_FAIL:   float = 0.10
RECEIVER_R2_TARGET: float = 0.90
RECEIVER_R2_FAIL:   float = 0.80


@dataclass(frozen=True)
class ModelSpec:
    """One sender model entry in the 4-model run."""
    name: str            # display name written to the log
    provider: str        # "openai" | "anthropic" | "google" | "stub"
    api_model: str       # provider-specific model identifier
    extra: dict = field(default_factory=dict)


# Default 4-model basket. Edit api_model strings to match available endpoints.
DEFAULT_MODELS: Tuple[ModelSpec, ...] = (
    ModelSpec("gpt-4o",          "openai",    "gpt-4o-2024-11-20"),
    ModelSpec("claude-sonnet",   "anthropic", "claude-sonnet-4-5-20250929"),
    ModelSpec("gemini-pro",      "google",    "gemini-2.0-pro"),
    ModelSpec("llama-3.1-70b",   "openai",    "meta-llama/Llama-3.1-70B-Instruct",
              extra={"base_url": "https://api.together.xyz/v1"}),
)

# --------------------------------------------------------------------------- #
# Named baskets selectable from the CLI (--provider_set <name>).               #
#   path_x : two labs only — uses keys Reza already pays for (OpenAI+Anthropic)#
#            diversity = frontier vs budget tier within each lab.              #
#   path_y : cross-lab + open weights (recommended). Adds Together-hosted      #
#            open models (OpenAI-compatible endpoint) for one cheap key.       #
# --------------------------------------------------------------------------- #
_TOGETHER = {"base_url": "https://api.together.xyz/v1"}

BASKET_X: Tuple[ModelSpec, ...] = (
    ModelSpec("gpt-4o",        "openai",    "gpt-4o-2024-11-20"),
    ModelSpec("gpt-4o-mini",   "openai",    "gpt-4o-mini-2024-07-18"),
    ModelSpec("claude-sonnet", "anthropic", "claude-sonnet-4-5-20250929"),
    ModelSpec("claude-haiku",  "anthropic", "claude-haiku-4-5-20251001"),
)

BASKET_Y: Tuple[ModelSpec, ...] = (
    ModelSpec("gpt-4o",        "openai",    "gpt-4o-2024-11-20"),
    ModelSpec("claude-sonnet", "anthropic", "claude-sonnet-4-5-20250929"),
    ModelSpec("gemini-flash",  "google",    "gemini-2.5-flash-lite"),
    ModelSpec("llama-3.3-70b", "openai",    "meta-llama/Llama-3.3-70B-Instruct-Turbo", extra=_TOGETHER),
)

# Single-model basket: lets us collect the one funded provider now and merge
# the rest later (collection is keyed by model name, so dirs merge cleanly).
BASKET_GPT4O: Tuple[ModelSpec, ...] = (
    ModelSpec("gpt-4o", "openai", "gpt-4o-2024-11-20"),
)

# The other three providers (path_y minus gpt-4o), so they can run in parallel
# with the gpt-4o job into a separate dir and merge by model afterwards.
BASKET_OTHERS: Tuple[ModelSpec, ...] = (
    ModelSpec("claude-sonnet", "anthropic", "claude-sonnet-4-5-20250929"),
    ModelSpec("gemini-flash",  "google",    "gemini-2.5-flash-lite"),
    ModelSpec("llama-3.3-70b", "openai",    "meta-llama/Llama-3.3-70B-Instruct-Turbo", extra=_TOGETHER),
)

BASKETS = {"default": DEFAULT_MODELS, "path_x": BASKET_X,
           "path_y": BASKET_Y, "gpt4o": BASKET_GPT4O, "others": BASKET_OTHERS}


# --------------------------------------------------------------------------- #
# Derived sample-size sanity checks (asserted at import time).                 #
# --------------------------------------------------------------------------- #
def total_calls(n_models: int = len(DEFAULT_MODELS)) -> int:
    return n_models * len(BIASES) * len(FRAMES) * STATES_PER_CELL


def positive_calls(n_models: int = len(DEFAULT_MODELS)) -> int:
    return n_models * len(POSITIVE_BIASES) * len(FRAMES) * STATES_PER_CELL


assert total_calls() == 12_000, "Pruned design must total 12000 sender calls."
assert positive_calls() == 9_600, "Positive-bias slice must total 9600 calls."
