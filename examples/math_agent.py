"""Example: Math problem-solving agent with 3 states.

States: thinking -> tool_call -> reflect -> answer

Usage:
    uv run python examples/math_agent.py
    uv run python examples/math_agent.py "What is 123 * 456?"
"""

import asyncio
import json
import sys
from pathlib import Path

from src.core.state import StateConfig
from src.core.machine import StateMachine
from src.llm.claude import ClaudeProvider
from src.monitoring.tracker import TokenTracker

import logging
logging.disable(logging.CRITICAL)


def calculator(expression: str = "") -> str:
    """Safe calculator for basic arithmetic."""
    try:
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            return str(eval(expression))
        return f"Error: unsafe expression '{expression}'"
    except Exception as e:
        return f"Error: {e}"


def build_math_agent() -> StateMachine:
    machine = StateMachine()

    machine.add_state(StateConfig(
        name="thinking",
        prompt=(
            "You are a math problem-solving agent. Analyze the user's math problem.\n"
            "- Break the problem down into steps.\n"
            "- If you need to compute something, transition to 'tool_call'.\n"
            "- If you can answer directly, transition to 'answer'."
        ),
        transitions={
            "tool_call": "When you need to use the calculator for computation",
            "answer": "When you already know the answer without needing tools",
        },
    ))

    machine.add_state(StateConfig(
        name="tool_call",
        prompt=(
            "You are in calculation mode. Use the calculator tool.\n"
            "- Set tool_name to 'calculator'\n"
            "- Set tool_args to {\"expression\": \"<math expression>\"}\n"
            "- Then transition to 'reflect' to review the result."
        ),
        transitions={"reflect": "After performing the calculation"},
        allowed_tools=["calculator"],
    ))

    machine.add_state(StateConfig(
        name="reflect",
        prompt=(
            "Review the calculation result and decide next steps.\n"
            "- If you have the final answer, transition to 'answer'.\n"
            "- If you need more calculations, transition to 'tool_call'."
        ),
        transitions={
            "answer": "When you have the complete final answer",
            "tool_call": "When you need another calculation",
        },
    ))

    machine.add_state(StateConfig(
        name="answer",
        prompt="Provide the final answer clearly and concisely.",
        transitions={},
        is_terminal=True,
    ))

    machine.set_initial_state("thinking")
    machine.register_tool("calculator", calculator)
    return machine


def generate_report(ctx, problem: str) -> str:
    """Generate FSM.md execution report."""
    lines: list[str] = []
    w = lines.append

    w("# FSM Execution Report")
    w("")
    w(f"- input: `{problem}`")
    w(f"- final_state: `{ctx.current_state}`")
    w(f"- llm_calls: {ctx.turn_count}")
    w("")

    # State graph
    w("## Graph")
    w("")
    w("```")
    w("thinking --[tool_call]--> tool_call --[reflect]--> reflect --[answer]--> answer(T)")
    w("thinking --[answer]----> answer(T)")
    w("reflect  --[tool_call]--> tool_call")
    w("```")
    w("")

    # Trace
    w("## Trace")
    w("")

    step = 0
    for msg in ctx.history:
        role = msg["role"]
        content = msg["content"]

        if role == "user" and step == 0:
            step += 1
            continue

        if role == "assistant" and isinstance(content, dict):
            state = content.get("state", "?")
            next_state = content.get("next_state", "?")
            thinking = content.get("thinking", "").replace("\n", " ").strip()
            response = content.get("response", "").replace("\n", " ").strip()
            tool_name = content.get("tool_name")
            tool_args = content.get("tool_args", {})

            w(f"### Step {step}: `{state}` -> `{next_state}`")
            w("")
            if thinking:
                w(f"- thinking: {thinking}")
            if tool_name and tool_name != "none":
                w(f"- tool: `{tool_name}({json.dumps(tool_args)})`")
            if response:
                w(f"- response: {response}")
            w("")
            step += 1

        elif role == "user" and isinstance(content, str) and "[Tool Result" in content:
            parts = content.split("\n", 1)
            result = parts[1].strip() if len(parts) > 1 else parts[0]
            w(f"- tool_result: `{result}`")
            w("")

    # Cost table
    w("## Cost")
    w("")
    w("| state | calls | cost_usd | duration_ms |")
    w("|-------|------:|---------:|------------:|")

    tracker = TokenTracker()
    tracker.stats = ctx.token_stats
    report = tracker.report()

    for name, stats in report["per_state"].items():
        cost = f"{stats['cost_usd']:.6f}" if stats["cost_usd"] else "0"
        dur = str(stats["duration_ms"]) if stats["duration_ms"] else "0"
        w(f"| {name} | {stats['calls']} | {cost} | {dur} |")

    total_cost = report["total_cost_usd"]
    w(f"| **total** | **{report['total_calls']}** | **{total_cost:.6f}** | |")
    w("")

    return "\n".join(lines)


async def main():
    problem = (
        "A store sells apples at $1.50 each and oranges at $2.00 each. "
        "If I buy 12 apples and 8 oranges, and I have a 15% discount coupon, "
        "how much do I pay?"
    )

    if len(sys.argv) > 1:
        problem = " ".join(sys.argv[1:])

    machine = build_math_agent()
    llm = ClaudeProvider(model="haiku", max_tokens=1024)

    ctx = await machine.run(problem, llm=llm)

    report = generate_report(ctx, problem)

    # Write FSM.md
    out_path = Path(__file__).parent.parent / "FSM.md"
    out_path.write_text(report)

    # Also print to stdout
    print(report)
    print(f"--- written to {out_path} ---")


if __name__ == "__main__":
    asyncio.run(main())
