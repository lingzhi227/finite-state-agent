"""High-level decorator API for defining FSM states."""

from __future__ import annotations

from src.core.machine import StateMachine

# Convenience: a default global machine instance
state_machine = StateMachine()
