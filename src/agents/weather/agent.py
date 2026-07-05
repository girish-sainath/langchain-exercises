"""
Weather agent initialization and configuration.
"""
from typing import Any, cast

from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel

from dotenv import load_dotenv

from src.agents.weather.Context import Context
from src.tools.weather.tool import get_weather
from src.tools.user.tool import locate_user

from src.models.ModelInfo import ModelInfo
from src.models.ModelFactory import ModelFactory
from src.prompts.PromptCatalog import PromptCatalog


load_dotenv()


def get_agent_with_city_input():
    """Create and return a weather agent equipped with the weather tool.

    Returns:
        A LangChain agent configured with the weather tool and a system prompt
        for providing reliable weather information.
    """
    model: BaseChatModel = ModelFactory.create_model(ModelInfo.DEFAULT_MODEL_TYPE.value)
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
    runtime_context_schema = cast(Any, Context)

    model: BaseChatModel = ModelFactory.create_model(ModelInfo.DEFAULT_MODEL_TYPE.value)
    agent = create_agent(
        model=model,
        tools=[get_weather, locate_user],
        system_prompt = PromptCatalog.WEATHER_SYSTEM_PROMPT.value,
        context_schema=runtime_context_schema,
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
