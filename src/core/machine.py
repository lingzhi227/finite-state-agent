from __future__ import annotations

import logging
from typing import Any, Callable, Coroutine

from src.core.state import StateConfig
from src.core.transition import build_response_schema
from src.core.context import ExecutionContext
from src.llm.base import LLMProvider
from src.monitoring.tracker import TokenTracker

logger = logging.getLogger(__name__)


class CompilationError(Exception):
    """Raised when the FSM graph fails validation."""


class StateMachine:
    """FSM engine that orchestrates state transitions via LLM calls."""

    def __init__(self) -> None:
        self._states: dict[str, StateConfig] = {}
        self._initial_state: str | None = None
        self._compiled: bool = False
        self._tools: dict[str, Callable[..., Any]] = {}  # Registered tool implementations

    def add_state(self, config: StateConfig) -> None:
        """Register a state configuration."""
        self._states[config.name] = config
        self._compiled = False

    def set_initial_state(self, name: str) -> None:
        """Set which state the FSM starts in."""
        self._initial_state = name
        self._compiled = False

    def register_tool(self, name: str, func: Callable[..., Any]) -> None:
        """Register a tool implementation by name."""
        self._tools[name] = func

    def state(
        self,
        name: str,
        prompt: str,
        transitions: dict[str, str],
        allowed_tools: list[str] | None = None,
        is_terminal: bool = False,
        max_turns: int = 10,
    ) -> Callable:
        """Decorator to register a state with an optional handler."""
        def decorator(
            func: Callable[..., Coroutine[Any, Any, Any]],
        ) -> Callable[..., Coroutine[Any, Any, Any]]:
            config = StateConfig(
                name=name,
                prompt=prompt,
                transitions=transitions,
                allowed_tools=allowed_tools or [],
                is_terminal=is_terminal,
                max_turns=max_turns,
                handler=func,
            )
            self.add_state(config)
            return func

        return decorator

    def compile(self) -> None:
        """Validate the FSM graph.

        Checks:
        1. Initial state is set and exists
        2. All transition targets reference existing states
        3. At least one terminal state exists
        4. No orphan states (every non-initial state is reachable)
        """
        errors: list[str] = []

        # Check initial state
        if self._initial_state is None:
            errors.append("No initial state set. Call set_initial_state().")
        elif self._initial_state not in self._states:
            errors.append(
                f"Initial state '{self._initial_state}' not found in registered states."
            )

        # Check transition targets exist
        for state_name, config in self._states.items():
            for target in config.transitions:
                if target not in self._states:
                    errors.append(
                        f"State '{state_name}' has transition to unknown state '{target}'."
                    )

        # Check at least one terminal state
        terminals = [s for s, c in self._states.items() if c.is_terminal]
        if not terminals:
            errors.append("No terminal states defined. At least one state must be terminal.")

        # Check reachability (BFS from initial state)
        if self._initial_state and self._initial_state in self._states:
            reachable: set[str] = set()
            queue = [self._initial_state]
            while queue:
                current = queue.pop(0)
                if current in reachable:
                    continue
                reachable.add(current)
                state_cfg = self._states.get(current)
                if state_cfg:
                    for target in state_cfg.transitions:
                        if target not in reachable:
                            queue.append(target)

            orphans = set(self._states.keys()) - reachable
            if orphans:
                errors.append(
                    f"Unreachable states: {', '.join(sorted(orphans))}. "
                    "These states cannot be reached from the initial state."
                )

        if errors:
            raise CompilationError("FSM compilation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        self._compiled = True
        logger.info(
            "FSM compiled: %d states, %d terminals, initial='%s'",
            len(self._states),
            len(terminals),
            self._initial_state,
        )

    async def run(
        self,
        task: str,
        llm: LLMProvider,
        max_total_turns: int = 50,
    ) -> ExecutionContext:
        """Execute the FSM from initial state until a terminal state or max turns.

        Returns the final ExecutionContext with full history and token stats.
        """
        if not self._compiled:
            self.compile()

        assert self._initial_state is not None
        llm.reset_session()
        ctx = ExecutionContext(task=task, current_state=self._initial_state)
        tracker = TokenTracker()

        # Seed the conversation with the user's task
        ctx.add_message("user", task)

        total_turns = 0

        while total_turns < max_total_turns:
            state = self._states[ctx.current_state]
            logger.info("State: %s (turn %d)", state.name, total_turns + 1)

            # Terminal state: done
            if state.is_terminal:
                logger.info("Reached terminal state: %s", state.name)
                break

            # Per-state turn limit
            state_turns = sum(
                1 for _ in filter(
                    lambda m: m.get("_state") == state.name,
                    ctx.history,
                )
            )
            if state_turns >= state.max_turns:
                logger.warning(
                    "State '%s' exceeded max_turns (%d), forcing first transition.",
                    state.name,
                    state.max_turns,
                )
                first_target = next(iter(state.transitions))
                ctx.current_state = first_target
                continue

            # Build JSON schema for structured output
            response_schema = build_response_schema(state)

            # Call LLM
            response = await llm.call(
                messages=ctx.get_messages(),
                system=state.prompt,
                response_schema=response_schema,
            )

            # Record usage
            tracker.record(state.name, response.usage)
            total_turns += 1
            ctx.turn_count = total_turns

            # Parse structured response
            tool_input = response.tool_input
            if tool_input is None:
                logger.error("LLM did not return valid structured output")
                break

            thinking = tool_input.get("thinking", "")
            resp_text = tool_input.get("response", "")
            next_state = tool_input.get("next_state", "")
            tool_name = tool_input.get("tool_name")
            tool_args = tool_input.get("tool_args", {})

            logger.info("  thinking: %s", thinking[:100])
            logger.info("  response: %s", resp_text[:100])
            logger.info("  next_state: %s", next_state)

            # Record assistant response in history
            ctx.add_message("assistant", {
                "state": state.name,
                "thinking": thinking,
                "response": resp_text,
                "next_state": next_state,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "_state": state.name,
            })

            # Execute tool if requested and allowed
            if tool_name and tool_name != "none":
                if tool_name in state.allowed_tools and tool_name in self._tools:
                    try:
                        result = self._tools[tool_name](**tool_args)
                        tool_result_text = str(result)
                    except Exception as e:
                        tool_result_text = f"Tool error: {e}"
                    logger.info("  tool %s -> %s", tool_name, tool_result_text[:100])
                    ctx.add_message("user", f"[Tool Result: {tool_name}]\n{tool_result_text}")
                    ctx.state_data[f"last_tool_result_{state.name}"] = tool_result_text
                elif tool_name not in state.allowed_tools:
                    logger.warning("  tool '%s' not in allowed_tools for state '%s'", tool_name, state.name)
                    ctx.add_message(
                        "user",
                        f"[System] Tool '{tool_name}' is not allowed in state '{state.name}'. "
                        f"Allowed tools: {state.allowed_tools}",
                    )

            # Run handler if defined
            if state.handler:
                handler_result = await state.handler(ctx, tool_input)
                if isinstance(handler_result, str):
                    ctx.state_data[f"handler_result_{state.name}"] = handler_result

            # Transition
            if next_state not in self._states:
                logger.error("Invalid transition target: %s", next_state)
                break
            ctx.current_state = next_state

        # Finalize token stats
        ctx.token_stats = tracker.stats
        return ctx
