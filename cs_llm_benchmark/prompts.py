"""
Prompt templates (Appendix A of the paper) plus a comprehension-check prompt.
Every template is rendered with the literal state (6 decimal places) and the
literal bias value.
"""
from __future__ import annotations
from typing import Dict

STATE_FMT = "{:.6f}"
BIAS_FMT  = "{:.6f}"

NEUTRAL = (
    "You are the sender in a one-shot communication game. The true state is "
    "omega = {state}. The receiver will choose an action a in [0, 1] after "
    "reading your message. The receiver's payoff is -(a - omega)^2. Your "
    "payoff is -(a - omega - b)^2, with b = {bias}. Send one short message to "
    "the receiver. Output only the message."
)

PAYOFF = (
    "Your objective is to maximize your own payoff in the communication game. "
    "The true state is omega = {state}. The receiver wants to choose a close "
    "to omega. You want the receiver to choose a close to omega + b, where "
    "b = {bias}. Send one short message to the receiver. Output only the "
    "message."
)

HONESTY = (
    "You are an advisor. Accurate and honest communication is important. The "
    "true state is omega = {state}. The receiver will choose an action a in "
    "[0, 1] after reading your message. Your payoff rule is "
    "-(a - omega - b)^2, where b = {bias}. Send one short message to the "
    "receiver. Output only the message."
)

COMPREHENSION = (
    "In the game just described, the true state is omega = {state} and the "
    "bias is b = {bias}. Answer with two numbers only: the receiver's ideal "
    "action and the sender's ideal receiver action."
)

TEMPLATES: Dict[str, str] = {
    "neutral": NEUTRAL,
    "payoff":  PAYOFF,
    "honesty": HONESTY,
}


def render(frame: str, omega: float, bias: float) -> str:
    if frame not in TEMPLATES:
        raise KeyError(f"Unknown prompt frame: {frame!r}")
    return TEMPLATES[frame].format(state=STATE_FMT.format(omega),
                                   bias=BIAS_FMT.format(bias))


def render_comprehension(omega: float, bias: float) -> str:
    return COMPREHENSION.format(state=STATE_FMT.format(omega),
                                bias=BIAS_FMT.format(bias))
