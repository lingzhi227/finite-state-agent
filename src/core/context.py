from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.monitoring.tracker import TokenStats


@dataclass
class ExecutionContext:
    """Tracks the full execution state of an FSM run."""

    task: str  # Original user input / task description
    current_state: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)  # Message history
    state_data: dict[str, Any] = field(default_factory=dict)  # Shared scratchpad
    turn_count: int = 0
    token_stats: dict[str, TokenStats] = field(default_factory=dict)  # Per-state

    def add_message(self, role: str, content: Any) -> None:
        """Append a message to the conversation history."""
        self.history.append({"role": role, "content": content})

    def get_messages(self) -> list[dict[str, Any]]:
        """Return the full message history."""
        return list(self.history)
