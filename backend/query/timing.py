"""End-to-end time estimates in seconds (fast LLM, no extended thinking).

We separate:
  - infra_sec  : local prompt assembly (measured wall clock)
  - llm_sec    : estimated model time from calls + tokens
  - total_sec  : infra + llm

Fast-mode assumptions (no thinking / no reasoning chain):
  - ~0.35 s fixed overhead per LLM round-trip
  - ~140 input+output token-equivalents per second (fast chat model)
"""
from __future__ import annotations

from dataclasses import dataclass

# Tunable — document in benchmark output
SEC_PER_LLM_CALL = 0.35
TOKENS_PER_SEC = 140.0  # fast mode, no thinking


@dataclass
class TimingEstimate:
    infra_sec: float
    llm_sec: float
    total_sec: float
    llm_calls: int
    input_tokens: int

    def to_dict(self) -> dict:
        return {
            "infra_sec": round(self.infra_sec, 2),
            "llm_sec": round(self.llm_sec, 2),
            "total_sec": round(self.total_sec, 2),
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
        }


def estimate_llm_seconds(llm_calls: int, input_tokens: int) -> float:
    return llm_calls * SEC_PER_LLM_CALL + input_tokens / TOKENS_PER_SEC


def estimate_timing(
    elapsed_ms: float,
    llm_calls: int,
    input_tokens: int,
) -> TimingEstimate:
    infra = elapsed_ms / 1000.0
    llm = estimate_llm_seconds(llm_calls, input_tokens)
    return TimingEstimate(
        infra_sec=infra,
        llm_sec=llm,
        total_sec=infra + llm,
        llm_calls=llm_calls,
        input_tokens=input_tokens,
    )


def attach_timing(result_dict: dict) -> dict:
    """Add timing block to a benchmark result dict."""
    t = estimate_timing(
        result_dict.get("elapsed_ms", 0),
        result_dict.get("llm_calls", 0),
        result_dict.get("input_tokens", 0),
    )
    result_dict["timing"] = t.to_dict()
    return result_dict
