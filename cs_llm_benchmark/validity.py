"""
Pre-specified Table 3 validity checks.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from . import config


@dataclass
class ValidityRow:
    check: str
    rate: float
    target: float
    fail_threshold: float
    status: str          # "ok" | "warn" | "fail"
    detail: Dict


def _status(rate: float, target: float, fail: float, lower_is_better: bool) -> str:
    if lower_is_better:
        if rate <= target: return "ok"
        if rate <= fail:   return "warn"
        return "fail"
    if rate >= target: return "ok"
    if rate >= fail:   return "warn"
    return "fail"


def valid_output_rate(records: List[dict]) -> ValidityRow:
    n = len(records)
    bad = sum(1 for r in records if r["parser_status"] != "ok" or r["api_error"])
    rate = 1.0 - bad / max(n, 1)
    return ValidityRow("valid_output_rate", rate,
                       config.VALID_OUTPUT_TARGET, config.VALID_OUTPUT_FAIL,
                       _status(rate, config.VALID_OUTPUT_TARGET,
                               config.VALID_OUTPUT_FAIL, lower_is_better=False),
                       {"n": n, "invalid": bad})


def empty_output_rate(records: List[dict]) -> ValidityRow:
    n = len(records)
    empty = sum(1 for r in records if r["parser_status"] == "empty")
    rate = empty / max(n, 1)
    return ValidityRow("empty_output_rate", rate,
                       config.EMPTY_OUTPUT_TARGET, config.EMPTY_OUTPUT_FAIL,
                       _status(rate, config.EMPTY_OUTPUT_TARGET,
                               config.EMPTY_OUTPUT_FAIL, lower_is_better=True),
                       {"n": n, "empty": empty})


def format_violation_rate(records: List[dict]) -> ValidityRow:
    n = len(records)
    fv = sum(1 for r in records if r["parser_status"] == "format_violation")
    rate = fv / max(n, 1)
    return ValidityRow("format_violation_rate", rate,
                       config.FORMAT_VIOLATION_TARGET, config.FORMAT_VIOLATION_FAIL,
                       _status(rate, config.FORMAT_VIOLATION_TARGET,
                               config.FORMAT_VIOLATION_FAIL, lower_is_better=True),
                       {"n": n, "format_violation": fv})


def comprehension_pass_rate(comp_records: List[dict],
                            tol: float = 0.05) -> ValidityRow:
    """A comprehension answer passes if the response contains a number within
    tol of the receiver-ideal (= omega) AND a number within tol of the
    sender-ideal (= omega + b).

    The search is order-agnostic and scans every number in the output, so a
    model that reasons before stating its two answers still counts as
    comprehending. This measures understanding of the game mechanics rather
    than compliance with a terse output format."""
    import re
    n = len(comp_records)
    passes = 0
    for r in comp_records:
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", r["raw"] or "")]
        has_recv = any(abs(v - r["expected_receiver"]) <= tol for v in nums)
        has_send = any(abs(v - r["expected_sender"]) <= tol for v in nums)
        if has_recv and has_send:
            passes += 1
    rate = passes / max(n, 1)
    return ValidityRow("comprehension_pass_rate", rate,
                       config.COMPREHENSION_TARGET, config.COMPREHENSION_FAIL,
                       _status(rate, config.COMPREHENSION_TARGET,
                               config.COMPREHENSION_FAIL, lower_is_better=False),
                       {"n": n, "passes": passes})


def receiver_r2_check(r2: float) -> ValidityRow:
    return ValidityRow("receiver_decoder_R2_at_b0", r2,
                       config.RECEIVER_R2_TARGET, config.RECEIVER_R2_FAIL,
                       _status(r2, config.RECEIVER_R2_TARGET,
                               config.RECEIVER_R2_FAIL, lower_is_better=False),
                       {})
