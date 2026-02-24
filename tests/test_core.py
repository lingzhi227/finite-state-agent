"""Unit tests for FSM core logic. No LLM calls."""

import pytest

from src.core.state import StateConfig
from src.core.transition import build_response_schema
from src.core.context import ExecutionContext
from src.core.machine import StateMachine, CompilationError
from src.monitoring.tracker import TokenTracker, TokenStats
from src.llm.base import TokenUsage


# --- StateConfig tests ---


class TestStateConfig:
    def test_basic_state(self):
        s = StateConfig(
            name="thinking",
            prompt="Think about the problem.",
            transitions={"answer": "When you have the answer"},
        )
        assert s.name == "thinking"
        assert s.is_terminal is False
        assert s.allowed_tools == []
        assert s.max_turns == 10

    def test_terminal_state(self):
        s = StateConfig(
            name="done",
            prompt="Final answer.",
            transitions={},
            is_terminal=True,
        )
        assert s.is_terminal is True
        assert s.transitions == {}

    def test_state_with_tools(self):
        s = StateConfig(
            name="tool_call",
            prompt="Use a tool.",
            transitions={"reflect": "After tool use"},
            allowed_tools=["calculator", "search"],
        )
        assert s.allowed_tools == ["calculator", "search"]


# --- Transition schema tests ---


class TestResponseSchema:
    def test_basic_schema(self):
        state = StateConfig(
            name="thinking",
            prompt="Think.",
            transitions={"answer": "When ready", "tool_call": "When need tool"},
        )
        schema = build_response_schema(state)
        assert schema["type"] == "object"

        props = schema["properties"]
        assert "thinking" in props
        assert "response" in props
        assert "next_state" in props
        assert set(props["next_state"]["enum"]) == {"answer", "tool_call"}

        # No tool fields when no allowed_tools
        assert "tool_name" not in props
        assert "tool_args" not in props

    def test_schema_with_tools(self):
        state = StateConfig(
            name="tool_call",
            prompt="Use tool.",
            transitions={"reflect": "After use"},
            allowed_tools=["calculator"],
        )
        schema = build_response_schema(state)
        props = schema["properties"]
        assert "tool_name" in props
        assert "tool_args" in props
        assert "calculator" in props["tool_name"]["enum"]
        assert "none" in props["tool_name"]["enum"]

    def test_schema_required_fields(self):
        state = StateConfig(
            name="s",
            prompt="p",
            transitions={"t": "c"},
            allowed_tools=["tool1"],
        )
        schema = build_response_schema(state)
        required = schema["required"]
        assert "thinking" in required
        assert "response" in required
        assert "next_state" in required
        assert "tool_name" in required
        assert "tool_args" in required

    def test_schema_is_valid_json_schema(self):
        """The schema should be a valid JSON Schema object."""
        state = StateConfig(
            name="test",
            prompt="test",
            transitions={"next": "go"},
        )
        schema = build_response_schema(state)
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema


# --- ExecutionContext tests ---


class TestExecutionContext:
    def test_init(self):
        ctx = ExecutionContext(task="solve 2+2")
        assert ctx.task == "solve 2+2"
        assert ctx.history == []
        assert ctx.turn_count == 0

    def test_add_message(self):
        ctx = ExecutionContext(task="test")
        ctx.add_message("user", "hello")
        ctx.add_message("assistant", "world")
        assert len(ctx.history) == 2
        assert ctx.history[0] == {"role": "user", "content": "hello"}

    def test_get_messages(self):
        ctx = ExecutionContext(task="test")
        ctx.add_message("user", "hi")
        msgs = ctx.get_messages()
        assert len(msgs) == 1
        # Ensure it's a copy
        msgs.append({"role": "assistant", "content": "bye"})
        assert len(ctx.history) == 1

    def test_state_data(self):
        ctx = ExecutionContext(task="test")
        ctx.state_data["key"] = "value"
        assert ctx.state_data["key"] == "value"


# --- StateMachine compilation tests ---


class TestStateMachineCompilation:
    def _simple_machine(self) -> StateMachine:
        """Create a valid 2-state machine."""
        m = StateMachine()
        m.add_state(StateConfig(
            name="start",
            prompt="Begin.",
            transitions={"end": "When done"},
        ))
        m.add_state(StateConfig(
            name="end",
            prompt="Done.",
            transitions={},
            is_terminal=True,
        ))
        m.set_initial_state("start")
        return m

    def test_valid_compilation(self):
        m = self._simple_machine()
        m.compile()  # Should not raise

    def test_no_initial_state(self):
        m = StateMachine()
        m.add_state(StateConfig(
            name="start", prompt="p", transitions={"end": "c"},
        ))
        m.add_state(StateConfig(
            name="end", prompt="p", transitions={}, is_terminal=True,
        ))
        with pytest.raises(CompilationError, match="No initial state"):
            m.compile()

    def test_missing_initial_state(self):
        m = StateMachine()
        m.add_state(StateConfig(
            name="start", prompt="p", transitions={}, is_terminal=True,
        ))
        m.set_initial_state("nonexistent")
        with pytest.raises(CompilationError, match="not found"):
            m.compile()

    def test_invalid_transition_target(self):
        m = StateMachine()
        m.add_state(StateConfig(
            name="start", prompt="p", transitions={"ghost": "never"},
        ))
        m.add_state(StateConfig(
            name="end", prompt="p", transitions={}, is_terminal=True,
        ))
        m.set_initial_state("start")
        with pytest.raises(CompilationError, match="unknown state 'ghost'"):
            m.compile()

    def test_no_terminal_state(self):
        m = StateMachine()
        m.add_state(StateConfig(
            name="a", prompt="p", transitions={"b": "go"},
        ))
        m.add_state(StateConfig(
            name="b", prompt="p", transitions={"a": "back"},
        ))
        m.set_initial_state("a")
        with pytest.raises(CompilationError, match="No terminal states"):
            m.compile()

    def test_orphan_state(self):
        m = StateMachine()
        m.add_state(StateConfig(
            name="start", prompt="p", transitions={"end": "done"},
        ))
        m.add_state(StateConfig(
            name="end", prompt="p", transitions={}, is_terminal=True,
        ))
        m.add_state(StateConfig(
            name="orphan", prompt="p", transitions={"end": "go"},
        ))
        m.set_initial_state("start")
        with pytest.raises(CompilationError, match="Unreachable states.*orphan"):
            m.compile()

    def test_three_state_chain(self):
        m = StateMachine()
        m.add_state(StateConfig(
            name="think", prompt="p",
            transitions={"act": "need action", "done": "have answer"},
        ))
        m.add_state(StateConfig(
            name="act", prompt="p",
            transitions={"think": "back to thinking"},
            allowed_tools=["calc"],
        ))
        m.add_state(StateConfig(
            name="done", prompt="p", transitions={}, is_terminal=True,
        ))
        m.set_initial_state("think")
        m.compile()  # Should not raise

    def test_decorator_api(self):
        m = StateMachine()

        @m.state(
            name="start",
            prompt="Begin.",
            transitions={"end": "When done"},
        )
        async def start_handler(ctx, response):
            return response

        @m.state(
            name="end",
            prompt="Done.",
            transitions={},
            is_terminal=True,
        )
        async def end_handler(ctx, response):
            return response

        m.set_initial_state("start")
        m.compile()  # Should not raise
        assert "start" in m._states
        assert m._states["start"].handler is start_handler


# --- TokenTracker tests ---


class TestTokenTracker:
    def test_record_and_report(self):
        tracker = TokenTracker()
        tracker.record("thinking", TokenUsage(cost_usd=0.001, duration_ms=500))
        tracker.record("thinking", TokenUsage(cost_usd=0.002, duration_ms=600))
        tracker.record("answer", TokenUsage(cost_usd=0.001, duration_ms=400))

        assert tracker.total_cost() == pytest.approx(0.004)
        assert tracker.total_calls() == 3

        report = tracker.report()
        assert report["per_state"]["thinking"]["calls"] == 2
        assert report["per_state"]["thinking"]["cost_usd"] == pytest.approx(0.003)
        assert report["per_state"]["answer"]["calls"] == 1

    def test_empty_tracker(self):
        tracker = TokenTracker()
        assert tracker.total_cost() == 0.0
        assert tracker.total_calls() == 0
        report = tracker.report()
        assert report["total_cost_usd"] == 0.0

    def test_token_stats(self):
        s = TokenStats(cost_usd=0.005, duration_ms=1000, calls=2)
        assert s.cost_usd == 0.005
        assert s.calls == 2


# --- Tool whitelist filtering test ---


class TestToolWhitelist:
    def test_allowed_tools_in_schema(self):
        state = StateConfig(
            name="tool_state",
            prompt="Use tools.",
            transitions={"next": "continue"},
            allowed_tools=["search", "calculator"],
        )
        schema = build_response_schema(state)
        tool_enum = schema["properties"]["tool_name"]["enum"]
        assert "search" in tool_enum
        assert "calculator" in tool_enum
        assert "none" in tool_enum
        # Not-allowed tool should not be in enum
        assert "dangerous_tool" not in tool_enum

    def test_no_tools_no_tool_fields(self):
        state = StateConfig(
            name="no_tools",
            prompt="No tools.",
            transitions={"next": "go"},
            allowed_tools=[],
        )
        schema = build_response_schema(state)
        assert "tool_name" not in schema["properties"]
