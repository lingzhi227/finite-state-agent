# finite-state-agent

FSM-based agent control framework. Each state has its own system prompt, tool whitelist, and transition rules. The LLM is forced to output structured JSON at every step, making state transitions deterministic and traceable.

## Why

Existing agent frameworks give the LLM a bag of tools and hope for the best. This leads to unpredictable control flow, wasted tokens on irrelevant tools, and no way to audit what happened.

finite-state-agent constrains the LLM at each step:
- **What it can do** — tool whitelist per state
- **Where it can go** — explicit transition targets as enum
- **What it must output** — JSON schema enforced via system prompt

The result: an agent whose execution is a readable state machine trace.

## Architecture

```
src/
  core/
    state.py        StateConfig dataclass (prompt, transitions, allowed_tools)
    transition.py   Builds JSON schema from state config
    context.py      ExecutionContext (history, shared data, token stats)
    machine.py      StateMachine (compile, run — the FSM engine)
  llm/
    base.py         LLMProvider ABC
    claude.py       ClaudeProvider (claude CLI, single-session --resume)
  monitoring/
    tracker.py      Per-state cost and call tracking
  api.py            Decorator convenience
```

## How it works

```
          +----------+     +----------+     +---------+
  task -> | analyze  | --> | compute  | --> |  check  | --+
          +----------+     +----------+     +---------+   |
                                ^               |         |
                                +-- need more --+         |
                                                          v
                                               +------------+     +------+
                                               | synthesize | --> | done |
                                               +------------+     +------+
```

Each non-terminal state:
1. Builds a JSON schema from `transitions` + `allowed_tools`
2. Calls Claude CLI with per-state instructions + schema
3. Parses the structured response: `{thinking, response, next_state, tool_name?, tool_args?}`
4. Executes tool if requested and allowed
5. Transitions to `next_state`
6. Records cost per state

All calls within one FSM run share a single Claude session (`--resume`), so the LLM has full conversation context.

## Setup

```bash
git clone https://github.com/lingzhi227/finite-state-agent.git
cd finite-state-agent
uv sync --extra dev
```

Requires: [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated.

## Usage

### Define states

```python
from src.core.state import StateConfig
from src.core.machine import StateMachine
from src.llm.claude import ClaudeProvider

machine = StateMachine()

machine.add_state(StateConfig(
    name="thinking",
    prompt="Analyze the problem. Transition to 'compute' if you need a tool.",
    transitions={
        "compute": "When you need to calculate something",
        "answer": "When you already know the answer",
    },
))

machine.add_state(StateConfig(
    name="compute",
    prompt="Use the calculator tool, then transition to 'thinking'.",
    transitions={"thinking": "After getting the result"},
    allowed_tools=["calculator"],
))

machine.add_state(StateConfig(
    name="answer",
    prompt="Final answer.",
    transitions={},
    is_terminal=True,
))

machine.set_initial_state("thinking")
machine.register_tool("calculator", lambda expression="": str(eval(expression)))
```

### Or use the decorator API

```python
machine = StateMachine()

@machine.state(
    name="thinking",
    prompt="Analyze the problem.",
    transitions={"answer": "When ready"},
)
async def on_thinking(ctx, response):
    return response

machine.set_initial_state("thinking")
```

### Run

```python
import asyncio

async def main():
    llm = ClaudeProvider(model="haiku")
    ctx = await machine.run("What is 347 * 923?", llm=llm)
    print(ctx.current_state)   # "answer"
    print(ctx.turn_count)      # number of LLM calls
    print(ctx.token_stats)     # per-state cost breakdown

asyncio.run(main())
```

### FSM compilation

`compile()` validates the graph before execution:
- Initial state exists
- All transition targets exist
- At least one terminal state
- No unreachable (orphan) states

```python
machine.compile()  # raises CompilationError on invalid graph
```

## Examples

**Simple math agent** — 3 states, 1 tool:
```bash
uv run python examples/math_agent.py
```

**Salary projection** — 5 states, 3 tools (`calc`, `compound`, `stats`), multi-cycle compute/check loop:
```bash
uv run python examples/salary_analysis.py
```

Both output a `FSM.md` trace file with full execution trace and cost report.

## Tests

```bash
# Unit tests (no LLM calls, instant)
uv run pytest tests/test_core.py -v

# Integration tests (real Claude CLI calls)
uv run pytest tests/test_claude.py -v
```

24 unit tests cover: state config, schema generation, context management, FSM compilation (valid/invalid graphs), token tracking, tool whitelist filtering.

4 integration tests cover: simple Q&A, multi-tool math, cost tracking, tool constraints.

## Key decisions

| Decision | Choice | Why |
|---|---|---|
| Transition mechanism | JSON schema in system prompt | Claude CLI doesn't support `tool_choice`; prompt-based schema is reliable |
| LLM provider | `claude` CLI via subprocess | No API key needed, uses existing auth, single-session via `--resume` |
| Per-state prompts | Swapped each step | StateFlow research showed -81% tokens vs monolithic prompt |
| Tool constraints | Filter before LLM call | Tools not in whitelist don't appear in schema — LLM can't hallucinate them |
| Session management | Single session + `--resume` | Full conversation context across FSM steps without re-sending history |
| Structured output | System prompt + JSON schema | `--json-schema` flag unreliable; explicit prompt instructions work |

## License

MIT
