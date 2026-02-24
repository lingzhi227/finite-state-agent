from __future__ import annotations

from typing import Any

from src.core.state import StateConfig


def build_response_schema(state: StateConfig) -> dict[str, Any]:
    """Build a JSON Schema for structured state transitions.

    Used with `claude --json-schema` to enforce structured output.
    The LLM must output:
    - thinking: internal reasoning
    - response: user-facing response text
    - next_state: one of the valid transition targets (enum)

    For states with allowed_tools, also includes:
    - tool_name: name of tool to call (or "none")
    - tool_args: arguments for the tool
    """
    transition_targets = list(state.transitions.keys())

    # Build description from transition conditions
    conditions = "\n".join(
        f"- \"{target}\": {condition}"
        for target, condition in state.transitions.items()
    )

    properties: dict[str, Any] = {
        "thinking": {
            "type": "string",
            "description": "Your internal reasoning about what to do next.",
        },
        "response": {
            "type": "string",
            "description": "Your response text to the user or the result of your work.",
        },
        "next_state": {
            "type": "string",
            "enum": transition_targets,
            "description": f"Which state to transition to. Choose based on:\n{conditions}",
        },
    }
    required = ["thinking", "response", "next_state"]

    if state.allowed_tools:
        tool_names = state.allowed_tools + ["none"]
        properties["tool_name"] = {
            "type": "string",
            "enum": tool_names,
            "description": (
                "Name of the tool to call, or 'none' if no tool is needed. "
                f"Available tools: {', '.join(state.allowed_tools)}"
            ),
        }
        properties["tool_args"] = {
            "type": "object",
            "description": "Arguments to pass to the selected tool. Empty if tool_name is 'none'.",
        }
        required.extend(["tool_name", "tool_args"])

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


# Keep backward compat alias
build_state_response_tool = build_response_schema
