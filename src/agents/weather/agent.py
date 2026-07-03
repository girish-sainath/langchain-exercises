"""
Weather agent initialization and configuration.
"""
from typing import Any, cast

from langchain_litellm import ChatLiteLLM

from langchain.agents import create_agent

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from src.agents.weather.Context import Context
from src.tools.weather.tool import get_weather
from src.tools.user.tool import locate_user

from src.models.ModelInfo import ModelInfo
from src.prompts.PromptCatalog import PromptCatalog


load_dotenv()


def _create_chat_model(
    model: str = ModelInfo.DEFAULT_MODEL.value,
    temperature: float = ModelInfo.DEFAULT_MODEL_TEMPERATURE.value,
) -> ChatLiteLLM:
    """Create and configure the chat model used by the weather assistant."""
    return ChatLiteLLM(
        model=model,
        temperature=temperature,
    )


def get_agent_with_city_input():
    """Create and return a weather agent equipped with the weather tool.

    Returns:
        A LangChain agent configured with the weather tool and a system prompt
        for providing reliable weather information.
    """
    model: ChatLiteLLM = _create_chat_model()
    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt = PromptCatalog.WEATHER_SYSTEM_PROMPT.value,
    )

    return agent


def get_agent_with_user_memory():
    """Create and return a weather agent that can handle user memory.

    Returns:
        A LangChain agent configured with the weather tool and a system prompt
        for providing reliable weather information, capable of handling user memory.
    """

    serde = JsonPlusSerializer(allowed_msgpack_modules=[Context])
    checkpointer = InMemorySaver(serde=serde)
    runtime_context_schema = cast(Any, Context)
    agent_checkpointer = cast(Any, checkpointer)

    model: ChatLiteLLM = _create_chat_model()
    agent = create_agent(
        model=model,
        tools=[get_weather, locate_user],
        system_prompt = PromptCatalog.WEATHER_SYSTEM_PROMPT.value,
        context_schema=runtime_context_schema,
        checkpointer=agent_checkpointer,
    )

    return agent


__all__ = [
    'get_agent_with_city_input',
    'get_agent_with_user_memory',
]


def main() -> None:
    """
    Test the weather agent with city input.
    """
    # pylint: disable=import-outside-toplevel
    # pylint: disable=cyclic-import
    from src.agents.weather.agent_executor import execute_weather_agent
    execute_weather_agent()


if __name__ == '__main__':
    main()
