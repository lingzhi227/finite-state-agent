"""Salary projection agent — multi-tool, multi-cycle FSM demo.

5 states: analyze -> compute -> check -> (loop or) -> synthesize -> done
3 tools:  calc, compound, stats

Usage:
    uv run python examples/salary_analysis.py
"""

import asyncio
import json
import math
import sys
from datetime import datetime
from pathlib import Path

from src.core.state import StateConfig
from src.core.machine import StateMachine
from src.llm.claude import ClaudeProvider
from src.monitoring.tracker import TokenTracker

import logging
logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def calc(expression: str = "") -> str:
    """Basic arithmetic. Supports +, -, *, /, (), **."""
    try:
        allowed = set("0123456789+-*/.() e")
        if not all(c in allowed for c in expression.replace("**", "")):
            return f"ERROR: unsafe expression"
        return str(round(eval(expression), 2))
    except Exception as e:
        return f"ERROR: {e}"


def compound(base: float = 0, rate: float = 0, years: int = 1) -> str:
    """Compound growth. Returns year-by-year breakdown and total earned."""
    rows = []
    total = 0.0
    current = float(base)
    for y in range(1, int(years) + 1):
        if y > 1:
            current = current * (1 + float(rate))
        rows.append({"year": y, "salary": round(current, 2)})
        total += current
    return json.dumps({
        "yearly": rows,
        "final_year_salary": round(current, 2),
        "total_earned": round(total, 2),
    })


def stats(values: str = "[]") -> str:
    """Descriptive statistics on a JSON array of numbers."""
    try:
        nums = json.loads(values) if isinstance(values, str) else values
        nums = [float(x) for x in nums]
    except Exception as e:
        return f"ERROR: {e}"
    if not nums:
        return "ERROR: empty list"
    n = len(nums)
    mean = sum(nums) / n
    sorted_nums = sorted(nums)
    median = (
        sorted_nums[n // 2]
        if n % 2
        else (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
    )
    return json.dumps({
        "count": n,
        "mean": round(mean, 2),
        "median": round(median, 2),
        "min": round(min(nums), 2),
        "max": round(max(nums), 2),
        "range": round(max(nums) - min(nums), 2),
    })


# ---------------------------------------------------------------------------
# FSM definition
# ---------------------------------------------------------------------------

PROBLEM = (
    "A startup has 3 engineers.\n"
    "  - Engineer A: $95,000/yr, projected annual raise 5%\n"
    "  - Engineer B: $82,000/yr, projected annual raise 8%\n"
    "  - Engineer C: $71,000/yr, projected annual raise 12%\n\n"
    "Calculate:\n"
    "  1. Each engineer's salary in year 1 through year 4 (compound raises).\n"
    "  2. The total payroll (sum of all 3) in year 4.\n"
    "  3. Descriptive stats (mean, median, range) of the 3 year-4 salaries.\n"
    "  4. Which engineer's cumulative 4-year earnings are highest?\n"
)


def build_agent() -> StateMachine:
    m = StateMachine()

    m.add_state(StateConfig(
        name="analyze",
        prompt=(
            "You are a financial analysis agent. Read the problem carefully.\n"
            "Plan which calculations you need, in what order.\n"
            "Transition to 'compute' when you have a plan."
        ),
        transitions={
            "compute": "When you have a plan and are ready to start calculating",
        },
    ))

    m.add_state(StateConfig(
        name="compute",
        prompt=(
            "Execute ONE calculation step using the available tools.\n"
            "Tools:\n"
            "  - calc(expression): basic arithmetic, e.g. calc(\"95000+82000+71000\")\n"
            "  - compound(base, rate, years): compound salary growth, returns yearly breakdown\n"
            "    e.g. compound(95000, 0.05, 4)\n"
            "  - stats(values): descriptive stats on a JSON array, e.g. stats(\"[95000,82000,71000]\")\n\n"
            "Pick ONE tool call, then transition to 'check'."
        ),
        transitions={
            "check": "After making a tool call, go here to review the result",
        },
        allowed_tools=["calc", "compound", "stats"],
    ))

    m.add_state(StateConfig(
        name="check",
        prompt=(
            "Review the results so far. Decide:\n"
            "  - If more calculations are needed, transition to 'compute'.\n"
            "  - If ALL 4 questions are answered, transition to 'synthesize'.\n"
            "List what you still need to calculate."
        ),
        transitions={
            "compute": "When more calculations are still needed",
            "synthesize": "When all 4 questions have been fully answered with data",
        },
    ))

    m.add_state(StateConfig(
        name="synthesize",
        prompt=(
            "Compile all results into a clear, structured final answer.\n"
            "Address all 4 questions with exact numbers.\n"
            "Transition to 'done'."
        ),
        transitions={
            "done": "When the final answer is complete",
        },
    ))

    m.add_state(StateConfig(
        name="done",
        prompt="Terminal.",
        transitions={},
        is_terminal=True,
    ))

    m.set_initial_state("analyze")
    m.register_tool("calc", calc)
    m.register_tool("compound", compound)
    m.register_tool("stats", stats)

    return m


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(ctx, problem: str) -> str:
    lines: list[str] = []
    w = lines.append

    w("# FSM Execution Report")
    w("")
    w(f"generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"model: haiku | states: 5 | tools: calc, compound, stats")
    w("")

    w("## Problem")
    w("")
    w("```")
    for line in problem.strip().splitlines():
        w(line)
    w("```")
    w("")

    # Graph
    w("## State Graph")
    w("")
    w("```")
    w("analyze --> compute --> check --+--> compute  (loop)")
    w("                               +--> synthesize --> done(T)")
    w("```")
    w("")

    # Trace
    w("## Execution Trace")
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
            thinking = content.get("thinking", "").strip()
            response = content.get("response", "").strip()
            tool_name = content.get("tool_name")
            tool_args = content.get("tool_args", {})

            w(f"### step {step}: {state} -> {next_state}")
            w("")

            if thinking:
                w("**thinking**:")
                for tl in thinking.splitlines():
                    tl = tl.strip()
                    if tl:
                        w(f"> {tl}")
                w("")

            if tool_name and tool_name != "none":
                # Format tool call as code
                if tool_args:
                    args_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in tool_args.items())
                else:
                    args_str = ""
                w(f"**tool call**: `{tool_name}({args_str})`")
                w("")

            if response:
                w(f"**response**: {response}")
                w("")

            step += 1

        elif role == "user" and isinstance(content, str) and "[Tool Result" in content:
            parts = content.split("\n", 1)
            result = parts[1].strip() if len(parts) > 1 else parts[0]
            # Try to pretty-print JSON results
            try:
                parsed = json.loads(result)
                w("**tool result**:")
                w("```json")
                w(json.dumps(parsed, indent=2))
                w("```")
            except (json.JSONDecodeError, TypeError):
                w(f"**tool result**: `{result}`")
            w("")

    # Summary
    w("---")
    w("")
    w("## Summary")
    w("")
    w(f"| metric | value |")
    w(f"|--------|-------|")
    w(f"| final_state | `{ctx.current_state}` |")
    w(f"| llm_calls | {ctx.turn_count} |")
    w(f"| states_visited | {', '.join(s.get('content', {}).get('state', '?') for s in ctx.history if s.get('role') == 'assistant' and isinstance(s.get('content'), dict))} |")

    # Tool usage count
    tool_calls = []
    for msg in ctx.history:
        if msg["role"] == "assistant" and isinstance(msg["content"], dict):
            tn = msg["content"].get("tool_name")
            if tn and tn != "none":
                tool_calls.append(tn)
    from collections import Counter
    tc = Counter(tool_calls)
    tool_str = ", ".join(f"{k}x{v}" for k, v in tc.items()) if tc else "none"
    w(f"| tool_calls | {tool_str} |")

    # Cost
    tracker = TokenTracker()
    tracker.stats = ctx.token_stats
    report = tracker.report()

    w("")
    w("### Cost by State")
    w("")
    w("| state | calls | cost_usd | duration_ms |")
    w("|-------|------:|---------:|------------:|")
    for name, s in report["per_state"].items():
        cost = f"{s['cost_usd']:.6f}" if s["cost_usd"] else "0"
        dur = str(s["duration_ms"]) if s["duration_ms"] else "0"
        w(f"| {name} | {s['calls']} | {cost} | {dur} |")
    tc = report["total_cost_usd"]
    w(f"| **total** | **{report['total_calls']}** | **{tc:.6f}** | |")
    w("")

    return "\n".join(lines)


async def main():
    problem = PROBLEM
    if len(sys.argv) > 1:
        problem = " ".join(sys.argv[1:])

    machine = build_agent()
    llm = ClaudeProvider(model="haiku", max_tokens=1024)

    print(f"Running FSM agent ({5} states, {3} tools)...", flush=True)
    ctx = await machine.run(problem, llm=llm)

    report = generate_report(ctx, problem)

    out_path = Path(__file__).parent.parent / "FSM.md"
    out_path.write_text(report)

    print(report)
    print(f"\n--- written to {out_path} ---")


if __name__ == "__main__":
    asyncio.run(main())
