from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.llm.base import TokenUsage


@dataclass
class TokenStats:
    """Accumulated stats for a single state."""
    cost_usd: float = 0.0
    duration_ms: int = 0
    calls: int = 0


class TokenTracker:
    """Tracks per-state usage across an FSM run."""

    def __init__(self) -> None:
        self.stats: dict[str, TokenStats] = {}

    def record(self, state_name: str, usage: TokenUsage) -> None:
        """Record usage for a state."""
        if state_name not in self.stats:
            self.stats[state_name] = TokenStats()
        s = self.stats[state_name]
        s.cost_usd += usage.cost_usd
        s.duration_ms += usage.duration_ms
        s.calls += 1

    def total_cost(self) -> float:
        """Total cost across all states."""
        return sum(s.cost_usd for s in self.stats.values())

    def total_calls(self) -> int:
        """Total LLM calls across all states."""
        return sum(s.calls for s in self.stats.values())

    def report(self) -> dict[str, Any]:
        """Generate a summary report."""
        per_state = {}
        for name, s in self.stats.items():
            per_state[name] = {
                "cost_usd": s.cost_usd,
                "duration_ms": s.duration_ms,
                "calls": s.calls,
            }
        return {
            "per_state": per_state,
            "total_cost_usd": self.total_cost(),
            "total_calls": self.total_calls(),
        }
