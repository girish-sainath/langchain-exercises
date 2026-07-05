# langchain-exercises

A collection of LangChain exercises demonstrating chat, agents, tools, RAG, and middleware patterns connecting to LLM via LiteLLM.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- LLM credentials in `.env`

## Setup

```bash
uv sync
```

Create a `.env` file at the project root with your LiteLLM/SAP AI Core credentials.

## Running

```bash
uv run python main.py
```

The interactive menu offers three modes:

| Option | Description                                                                         |
|--------|-------------------------------------------------------------------------------------|
| `1`    | Weather agent — provide a city name, or leave blank to look up your city by user ID |
| `2`    | Programming chat — invoke (full response) or streaming mode                         |
| `3`    | Image description — describe the bundled `brain_logo.png`                           |

Individual modules can also be run directly:

```bash
uv run python src/agents/weather/agent_executor.py
uv run python src/agents/knowledge_base/agent_executor.py
uv run python src/agents/explainer/agent_executor.py
uv run python src/tools/knowledge_base/tool.py
```

## Project Structure

```text
├── main.py                        # Interactive CLI entry point
├── src/
│   ├── chats/
│   │   ├── programming/           # Stateful programming assistant (invoke + stream)
│   │   └── image_description/     # Multimodal image description chat
│   ├── agents/
│   │   ├── weather/               # Weather agent with optional user-memory context
│   │   ├── knowledge_base/        # RAG agent backed by a Chroma vector store
│   │   └── explainer/             # Middleware demo: dynamic model + prompt by user role
│   ├── tools/
│   │   ├── weather/               # @tool: fetches weather from wttr.in
│   │   ├── user/                  # @tool: resolves user ID → city from runtime context
│   │   └── knowledge_base/        # In-memory Chroma store + LiteLLM embeddings retriever
│   ├── models/
│   │   └── ModelInfo.py           # Model configuration and metadata
│   └── prompts/
│       └── PromptCatalog.py       # Centralized prompt templates
```

## Key Concepts

- **Chats** (`src/chats/`) — simple `ChatLiteLLM` loops with growing conversation history; no agents or tools.
- **Agents** (`src/agents/`) — each agent follows a three-file layout: `agent.py` (construction), `agent_executor.py` (I/O + invocation), and an optional `Context.py` dataclass for runtime state.
- **Tools** (`src/tools/`) — LangChain `@tool`-decorated functions organized by domain. The `knowledge_base` tool demonstrates building a retriever from an in-memory Chroma vector store using `LiteLLMEmbeddings`. The `user` tool accesses `ToolRuntime[Context]` to read injected runtime state.
- **Models** (`src/models/`) — centralized model configuration and metadata.
- **Prompts** (`src/prompts/`) — centralized prompt templates and catalogs.
- **Middleware** (`src/agents/explainer/`) — two styles side by side: functional decorators (`@wrap_model_call`, `@dynamic_prompt`) and a class-based `AgentMiddleware` with full lifecycle hooks.


## LangChain

```python
from typing import Any

from langchain.tools import tool, ToolRuntime
@tool('<tool_name>', description='<tool_description>', return_direct=False)
def tool_definition(runtime: ToolRuntime) -> Any:
    """Tool description"""
    return '<tool_output>'





```