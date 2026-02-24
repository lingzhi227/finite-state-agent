"""Integration tests with real Claude CLI calls.

Run with: uv run pytest tests/test_claude.py -v
Requires: `claude` CLI installed and authenticated.
"""

import os
import shutil

import pytest

from src.core.state import StateConfig
from src.core.machine import StateMachine
from src.llm.claude import ClaudeProvider

# Skip all tests if claude CLI is not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("claude") is None,
        reason="claude CLI not installed",
    ),
    pytest.mark.skipif(
        bool(os.environ.get("CLAUDECODE")),
        reason="Cannot run claude CLI from within a Claude Code session",
    ),
]


@pytest.fixture
def claude():
    return ClaudeProvider(model="haiku", max_tokens=1024)


def _build_qa_machine() -> StateMachine:
    """Simple 2-state Q&A machine: thinking -> answer."""
    m = StateMachine()
    m.add_state(StateConfig(
        name="thinking",
        prompt=(
            "You are a helpful assistant. Analyze the user's question. "
            "Think step by step, then transition to 'answer' to give your final answer."
        ),
        transitions={"answer": "When you have formulated your answer"},
    ))
    m.add_state(StateConfig(
        name="answer",
        prompt="Provide the final answer.",
        transitions={},
        is_terminal=True,
    ))
    m.set_initial_state("thinking")
    return m


def _build_math_machine() -> StateMachine:
    """3-state machine with calculator tool: thinking -> tool_call -> reflect -> answer."""
    m = StateMachine()
    m.add_state(StateConfig(
        name="thinking",
        prompt=(
            "You are a math assistant. Analyze the math problem. "
            "If you need to compute something, transition to 'tool_call'. "
            "If you already know the answer, transition to 'answer'."
        ),
        transitions={
            "tool_call": "When you need to use the calculator",
            "answer": "When you already know the answer",
        },
    ))
    m.add_state(StateConfig(
        name="tool_call",
        prompt=(
            "You are in tool-calling mode. Use the calculator tool to compute. "
            "Set tool_name to 'calculator' and tool_args to {\"expression\": \"...\"}. "
            "Then transition to 'reflect' to review the result."
        ),
        transitions={"reflect": "After calling the tool"},
        allowed_tools=["calculator"],
    ))
    m.add_state(StateConfig(
        name="reflect",
        prompt=(
            "Review the tool result and formulate your final answer. "
            "Transition to 'answer' when ready."
        ),
        transitions={
            "answer": "When you have the final answer",
            "tool_call": "If you need another calculation",
        },
    ))
    m.add_state(StateConfig(
        name="answer",
        prompt="Provide the final answer.",
        transitions={},
        is_terminal=True,
    ))
    m.set_initial_state("thinking")

    # Register calculator tool
    def calculator(expression: str = "") -> str:
        try:
            allowed = set("0123456789+-*/.() ")
            if all(c in allowed for c in expression):
                return str(eval(expression))
            return f"Error: unsafe expression '{expression}'"
        except Exception as e:
            return f"Error: {e}"

    m.register_tool("calculator", calculator)
    return m


@pytest.mark.asyncio
async def test_simple_qa(claude):
    """Test 1: Simple Q&A — thinking -> answer (2 states, no tools)."""
    machine = _build_qa_machine()
    ctx = await machine.run("What is the capital of France?", llm=claude)

    assert ctx.current_state == "answer"
    assert ctx.turn_count >= 1
    assert len(ctx.history) >= 2  # At least user msg + assistant response

    # Check that a response was generated
    assistant_msgs = [m for m in ctx.history if m["role"] == "assistant"]
    assert len(assistant_msgs) >= 1
    last_response = assistant_msgs[-1]["content"]
    assert isinstance(last_response, dict)
    assert "response" in last_response


@pytest.mark.asyncio
async def test_math_with_calculator(claude):
    """Test 2: Math with calculator — thinking -> tool_call -> reflect -> answer."""
    machine = _build_math_machine()
    ctx = await machine.run("What is 347 * 923?", llm=claude)

    assert ctx.current_state == "answer"
    assert ctx.turn_count >= 1

    # Check history has content
    assistant_msgs = [m for m in ctx.history if m["role"] == "assistant"]
    assert len(assistant_msgs) >= 1


@pytest.mark.asyncio
async def test_cost_tracking(claude):
    """Test 3: Verify per-state cost is recorded."""
    machine = _build_qa_machine()
    ctx = await machine.run("What is 2 + 2?", llm=claude)

    # Token stats should be populated
    assert len(ctx.token_stats) > 0
    assert "thinking" in ctx.token_stats

    thinking_stats = ctx.token_stats["thinking"]
    assert thinking_stats.calls >= 1
    # Cost should be non-negative (may be 0 for free tier)
    assert thinking_stats.cost_usd >= 0.0


@pytest.mark.asyncio
async def test_tool_constraint(claude):
    """Test 4: Verify disallowed tools are rejected."""
    machine = _build_math_machine()

    from src.core.transition import build_response_schema

    # Verify thinking state schema has no tool fields
    thinking_state = machine._states["thinking"]
    schema = build_response_schema(thinking_state)
    assert "tool_name" not in schema["properties"]

    # Verify tool_call state schema has tool fields
    tool_state = machine._states["tool_call"]
    schema = build_response_schema(tool_state)
    assert "tool_name" in schema["properties"]
    assert "calculator" in schema["properties"]["tool_name"]["enum"]

    # Run the machine to verify it works end-to-end
    ctx = await machine.run("What is 15 * 28?", llm=claude)
    assert ctx.current_state == "answer"
