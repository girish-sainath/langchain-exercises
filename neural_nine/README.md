# LangChain Exercises by Neural Nine

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

"""
Tool definition example for LangChain 1.0.0+ with LiteLLM
"""
from langchain.tools import tool, ToolRuntime
@tool('get_weather', description='Tool to get the latest weather information', return_direct=False)
def get_weather(runtime: ToolRuntime) -> Any:
    """
    Tool definition
    """
    return '<tool_output>'


"""
RAG based Retriever Tool example for LangChain 1.0.0+ with LiteLLM
"""
from langchain_core.tools import create_retriever_tool, StructuredTool
from langchain_core.retrievers import BaseRetriever

def retriever_tool(retriever: BaseRetriever) -> StructuredTool:
    """
    Create a retriever tool from a BaseRetriever
    """
    return create_retriever_tool(
        name='retriever_tool',
        description='A retriever tool for LangChain with LiteLLM',
        retriever=retriever,
    )


"""
Chat Model example for LangChain 1.0.0+ with LiteLLM
"""
from langchain.chat_models import BaseChatModel
from langchain_litellm import ChatLiteLLM

def create_model_with_litellm() -> BaseChatModel:
    """
    Create a chat model using LiteLLM
    """
    model = ChatLiteLLM(
        model='gpt-4o-mini',
        temperature=0.1,
    )
    return model


"""
Chat Model example for LangChain 1.0.0+ with init_chat_model
"""
from langchain.chat_models import init_chat_model

def create_model_with_init_chat_model() -> BaseChatModel:
    """
    Create a chat model using init_chat_model
    """
    model: BaseChatModel = init_chat_model(
        model='gpt-4o-mini',
        temperature=0.7,
    )
    return model


"""
Invoke BaseChatModel example for LangChain 1.0.0+ with LiteLLM
"""
chat_model: BaseChatModel = create_model_with_litellm()
chat_response = chat_model.invoke(
    [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Hello! Can you help me with LangChain?'},
    ],
)

"""
Invoke BaseChatModel example for LangChain 1.0.0+ with init_chat_model with streaming and conversation history
"""

from langchain.messages import SystemMessage, HumanMessage, AIMessage

chat_model: BaseChatModel = create_model_with_init_chat_model()
conversations = [
    SystemMessage(content='You are a helpful assistant.'),
]

while True:
    user_input = input("User: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    conversations.append(HumanMessage(content=user_input))
    full_response = ''
    for chunk in chat_model.stream(conversations):
        print(chunk.content, end='', flush=True)
        full_response += chunk.content
    conversations.append(AIMessage(content=full_response))

"""
Agent example for LangChain 1.0.0+
"""

from langchain.agents import create_agent

model: BaseChatModel = create_model_with_litellm()
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt = 'Your are a helpful assistant that can provide weather information.',
)

agent_response = agent.invoke(
    {
        'messages': [
            {
                'role': 'user',
                'content': 'What is the weather like in New York City today?',
            }
        ],
    },
)


"""
Agent example for LangChain 1.0.0+ with init_chat_model with streaming and conversation history
"""
chat_model: BaseChatModel = create_model_with_init_chat_model()
agent = create_agent(
    model=chat_model,
    tools=[get_weather],
)
conversations = [
    SystemMessage(content='Your are a helpful assistant that can provide weather information.'),
]

while True:
    user_input = input("User: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    conversations.append(HumanMessage(content=user_input))
    full_response = ''
    for chunk in agent.stream(conversations):
        print(chunk.content, end='', flush=True)
        full_response += chunk.content
    conversations.append(AIMessage(content=full_response))

```
