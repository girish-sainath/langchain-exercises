# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for dependency and environment management (Python 3.13).

```bash
# Run the interactive CLI menu
uv run python main.py

# Run individual modules directly (each has an if __name__ == '__main__' block)
uv run python src/agents/weather/agent_executor.py
uv run python src/agents/knowledge_base/agent_executor.py
uv run python src/agents/explainer/agent_executor.py
uv run python src/agents/explainer/agent.py
uv run python src/tools/knowledge_base/tool.py

# Add a dependency
uv add <package>
```

No test suite or linter configuration exists yet.

## Environment

All modules call `load_dotenv()` at import time. A `.env` file at the project root is expected to provide LiteLLM/proxy credentials. The LLM backend is SAP AI Core, accessed through LiteLLM with model IDs prefixed `sap/anthropic--` (e.g. `sap/anthropic--claude-4.5-sonnet`).

## Architecture

The project is a collection of LangChain exercises organized under `src/` into six top-level packages:

### `src/chats/`

Simple stateful chat loops (no tools, no agents). Each module holds its own `ChatLiteLLM` instance and a growing `list[AnyMessage]` conversation history. Two modes: `.invoke()` (full response) and `.stream()` (token-by-token). Currently: `programming/` and `image_description/`.

### `src/agents/`

Each agent lives in its own subdirectory with three files:

- `agent.py` — creates and returns a LangChain agent via `create_agent(model, tools, system_prompt, ...)`
- `agent_executor.py` — handles I/O, calls the agent, prints results; has a `if __name__ == '__main__'` entry point
- `Context.py` (optional) — a `@dataclass` that carries runtime state injected at `.invoke()` time via `context=`

Agents:

- **weather** — two variants: city-input (stateless) and user-memory (stateful with `InMemorySaver` + `JsonPlusSerializer`, uses a `Context(user_id)` to look up city via the `locate_user` tool)
- **knowledge_base** — RAG agent using a Chroma retriever tool (`kb_search`)
- **explainer** — demonstrates the middleware API; dynamically selects model and system prompt based on `Context(user_role)`

### `src/tools/`

LangChain `@tool`-decorated functions consumed by agents, organized by domain:

- `src/tools/weather/tool.py` — `get_weather(city)`: hits `wttr.in` JSON API
- `src/tools/user/tool.py` — `locate_user(runtime: ToolRuntime[Context])`: resolves a hardcoded `user_id → city` mapping from the injected runtime context
- `src/tools/knowledge_base/tool.py` — builds an in-memory Chroma vector store from hardcoded strings, wraps it as a LangChain retriever tool (`kb_search`) using `LiteLLMEmbeddings(model='sap/text-embedding-ada-002')`

### `src/models/`

Centralized model configuration and metadata:

- `ModelInfo.py` — model definitions and configuration parameters used across agents and chats

### `src/prompts/`

Centralized prompt templates and catalogs:

- `PromptCatalog.py` — prompt templates and system prompts shared across agents and chats

### Middleware pattern (`src/agents/explainer/`)

Two approaches shown side-by-side:

1. **Functional decorators** (`@wrap_model_call`, `@dynamic_prompt`) passed as a `middleware=[]` list to `create_agent`
2. **Class-based** `ExplainerAgentMiddleware(AgentMiddleware)` with lifecycle hooks (`before_agent`, `before_model`, `wrap_model_call`, `dynamic_prompt`, `after_model`, `after_agent`)

Both approaches swap models per role: `expert → claude-4.7-opus`, `child → claude-4.5-haiku`, `beginner → claude-4.6-sonnet` (default).
