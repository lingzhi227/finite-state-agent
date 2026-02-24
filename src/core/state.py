from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine


@dataclass
class StateConfig:
    """Configuration for a single FSM state."""

    name: str
    prompt: str  # Per-state system prompt injected as system message
    transitions: dict[str, str]  # {target_state_name: "natural language condition"}
    allowed_tools: list[str] = field(default_factory=list)  # Tool whitelist
    is_terminal: bool = False
    max_turns: int = 10
    handler: Callable[..., Coroutine[Any, Any, Any]] | None = None  # Optional async handler
