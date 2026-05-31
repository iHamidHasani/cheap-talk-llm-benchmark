"""
Sender API wrappers + Algorithm 2 collection loop.

Supported providers:
  - openai     (also covers any OpenAI-compatible endpoint via base_url)
  - anthropic
  - google
  - stub       (deterministic synthetic sender for offline pipeline testing)

Every query is logged as one JSONL record with all fields needed to
re-derive every downstream table from disk without re-querying any model.
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np

from . import config, prompts


# --------------------------------------------------------------------------- #
# Provider adapters. Every adapter returns the raw assistant text.             #
# --------------------------------------------------------------------------- #
def _openai_call(spec: config.ModelSpec, prompt: str, max_tokens: int = None) -> str:
    from openai import OpenAI                              # lazy import
    base_url = spec.extra.get("base_url")
    # When a custom base_url is set (e.g. Together's OpenAI-compatible endpoint)
    # the call must use that provider's own key, not OPENAI_API_KEY. The key
    # env var can be named in spec.extra["api_key_env"]; default to TOGETHER for
    # together.xyz, otherwise fall back to OPENAI_API_KEY.
    api_key = None
    if base_url:
        env_name = spec.extra.get("api_key_env")
        if not env_name:
            env_name = "TOGETHER_API_KEY" if "together" in base_url else "OPENAI_API_KEY"
        api_key = os.environ.get(env_name)
    client = OpenAI(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=spec.api_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.DECODING_TEMPERATURE,
        max_tokens=max_tokens or config.MAX_OUTPUT_TOKENS,
    )
    return resp.choices[0].message.content or ""


def _anthropic_call(spec: config.ModelSpec, prompt: str, max_tokens: int = None) -> str:
    import anthropic                                       # lazy import
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=spec.api_model,
        max_tokens=max_tokens or config.MAX_OUTPUT_TOKENS,
        temperature=config.DECODING_TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
    return "".join(parts)


def _google_call(spec: config.ModelSpec, prompt: str, max_tokens: int = None) -> str:
    import google.generativeai as genai                    # lazy import
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(spec.api_model)
    resp = model.generate_content(
        prompt,
        generation_config={"temperature": config.DECODING_TEMPERATURE,
                           "max_output_tokens": max_tokens or config.MAX_OUTPUT_TOKENS},
    )
    # .text raises if the model returned no text part (e.g. all budget spent on
    # thinking). Fall back to concatenating any text parts, else empty string.
    try:
        return resp.text or ""
    except Exception:
        try:
            return "".join(p.text for c in resp.candidates for p in c.content.parts
                           if getattr(p, "text", ""))
        except Exception:
            return ""


def _stub_call(spec: config.ModelSpec, prompt: str, max_tokens: int = None) -> str:
    """Deterministic synthetic sender used for offline pipeline tests.

    Behaviour: parses the omega and b values out of the rendered prompt and
    returns a noisy, biased numeric report. The noise schedule increases with
    b so the empirical pipeline observes the predicted monotonicity.
    """
    import re
    m_omega = re.search(r"omega\s*=\s*(\d+\.\d+|\d+)", prompt)
    m_bias  = re.search(r"\bb\s*=\s*(\d+\.\d+|\d+)", prompt)
    if not (m_omega and m_bias):
        return ""
    omega = float(m_omega.group(1))
    b     = float(m_bias.group(1))
    rng = np.random.default_rng(int(1e9 * (omega + 100 * b) + hash(spec.name) % 1000))
    # Honesty bumps truthfulness, payoff frame inflates more aggressively.
    if "advisor" in prompt or "honest" in prompt:
        report = omega + 0.5 * b + 0.01 * rng.standard_normal()
    elif "maximize your own payoff" in prompt:
        report = omega + 1.5 * b + (0.02 + 2 * b) * rng.standard_normal()
    else:
        report = omega + b + (0.01 + 1.5 * b) * rng.standard_normal()
    return f"The value is approximately {max(0.0, min(1.0, report)):.3f}."


PROVIDERS: Dict[str, Callable[[config.ModelSpec, str], str]] = {
    "openai":    _openai_call,
    "anthropic": _anthropic_call,
    "google":    _google_call,
    "stub":      _stub_call,
}


# --------------------------------------------------------------------------- #
# Parsing. The paper requires "one short message, no explanation". The parser  #
# never judges truth; it only enforces the output contract.                    #
# --------------------------------------------------------------------------- #
def parse_message(raw: str) -> tuple[str, str]:
    """Return (clean_message, status).

    status in {ok, empty, format_violation}. The clean message is the raw
    output with leading/trailing whitespace stripped and any trailing
    "Explanation: ..." block removed.
    """
    if raw is None:
        return "", "empty"
    text = raw.strip()
    if not text:
        return "", "empty"
    # Drop any explicit "explanation" tail. Considered a format violation but
    # we keep the message portion for analysis.
    lowered = text.lower()
    status = "ok"
    for marker in ("explanation:", "rationale:", "reasoning:"):
        idx = lowered.find(marker)
        if idx != -1:
            text = text[:idx].strip()
            status = "format_violation"
            break
    # Multi-paragraph outputs are also format violations; keep the first line.
    if "\n\n" in text:
        text = text.split("\n\n", 1)[0].strip()
        status = "format_violation"
    return text, status


# --------------------------------------------------------------------------- #
# Algorithm 2: collect_sender_messages.                                        #
# --------------------------------------------------------------------------- #
@dataclass
class CollectionPlan:
    models: list
    biases: tuple
    frames: tuple
    states: np.ndarray
    output_path: Path
    max_retries: int = 3
    backoff_seconds: float = 2.0


def seeded_states(n: int = config.STATES_PER_CELL,
                  seed: int = config.GLOBAL_SEED) -> np.ndarray:
    """One omega vector reused across every (model, bias, frame) cell."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=n)


def already_logged(path: Path) -> set[tuple]:
    """Return the set of (model, bias, frame, state_index) tuples present."""
    done: set[tuple] = set()
    if not path.exists():
        return done
    with path.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
                done.add((r["model"], r["bias"], r["frame"], r["state_index"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def run_collection(plan: CollectionPlan) -> None:
    """Execute Algorithm 2. Idempotent: skips records already on disk."""
    plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    done = already_logged(plan.output_path)
    with plan.output_path.open("a") as out:
        for spec in plan.models:
            call_fn = PROVIDERS[spec.provider]
            for b in plan.biases:
                for frame in plan.frames:
                    for t, omega_raw in enumerate(plan.states):
                        key = (spec.name, float(b), frame, t)
                        if key in done:
                            continue
                        # Paper sec. 4.2: states are rendered to 6 decimal
                        # places, and the rendered value is used as the state
                        # in the analysis. Round before both rendering and
                        # logging so that omega in the JSONL matches the
                        # number the model actually saw.
                        omega = round(float(omega_raw), 6)
                        prompt = prompts.render(frame, omega, b)
                        raw, err = _call_with_retry(call_fn, spec, prompt,
                                                    plan.max_retries,
                                                    plan.backoff_seconds)
                        message, status = parse_message(raw)
                        record = {
                            "model":        spec.name,
                            "api_model":    spec.api_model,
                            "provider":     spec.provider,
                            "bias":         float(b),
                            "frame":        frame,
                            "state_index":  t,
                            "omega":        omega,
                            "omega_raw":    float(omega_raw),
                            "seed":         config.GLOBAL_SEED,
                            "prompt":       prompt,
                            "raw_output":   raw,
                            "message":      message,
                            "parser_status": status,
                            "api_error":    err,
                            "temperature":  config.DECODING_TEMPERATURE,
                            "max_tokens":   config.MAX_OUTPUT_TOKENS,
                            "timestamp":    time.time(),
                        }
                        out.write(json.dumps(record) + "\n")
                        out.flush()


def _call_with_retry(fn, spec, prompt, retries, backoff, max_tokens=None):
    last_exc = None
    for attempt in range(retries):
        try:
            return fn(spec, prompt, max_tokens), None
        except Exception as exc:                           # broad on purpose
            last_exc = repr(exc)
            time.sleep(backoff * (2 ** attempt))
    return "", last_exc


# --------------------------------------------------------------------------- #
# Comprehension diagnostics (60 prompts per the design).                       #
# --------------------------------------------------------------------------- #
def run_comprehension(models: Iterable[config.ModelSpec],
                      output_path: Path,
                      max_retries: int = 3,
                      backoff_seconds: float = 2.0) -> None:
    """One comprehension prompt per (model, bias, frame): 4*5*3 = 60 calls."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.GLOBAL_SEED + 1)
    with output_path.open("w") as out:
        for spec in models:
            call_fn = PROVIDERS[spec.provider]
            for b in config.BIASES:
                for frame in config.FRAMES:
                    omega = float(rng.uniform(0.0, 1.0))
                    cprompt = prompts.render_comprehension(omega, b)
                    # Comprehension answers may include reasoning before the two
                    # numbers; give a larger budget than the terse sender task so
                    # verbose / reasoning models can actually reach their answer.
                    raw, err = _call_with_retry(call_fn, spec, cprompt,
                                                max_retries, backoff_seconds,
                                                max_tokens=config.COMPREHENSION_MAX_TOKENS)
                    out.write(json.dumps({
                        "model":   spec.name,
                        "bias":    float(b),
                        "frame":   frame,
                        "omega":   omega,
                        "raw":     raw,
                        "error":   err,
                        "expected_receiver": omega,
                        "expected_sender":   omega + b,
                    }) + "\n")
