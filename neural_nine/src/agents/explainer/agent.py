"""Explainer agent initialization and configuration."""
from typing import Any, Literal, cast

from dotenv import load_dotenv

from langchain.chat_models import BaseChatModel

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    dynamic_prompt,
    wrap_model_call,
    SummarizationMiddleware
)

from src.agents.explainer.Context import Context
from src.agents.explainer.ExplainerAgentMiddleware import ExplainerAgentMiddleware

from src.models.ModelInfo import ModelInfo
from src.models.ModelFactory import ModelFactory


load_dotenv()

SUMMARIZATION_TRIGGER: tuple[Literal['tokens'], int] = ('tokens', 4000)
SUMMARIZATION_KEEP: tuple[Literal['messages'], int] = ('messages', 20)


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Dynamic model selection based on the user's role in the system prompt."""
    user_role = request.runtime.context.user_role
    model = ExplainerAgentMiddleware.get_model_for_role(user_role, request.model)
    return handler(request.override(model=model))


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Dynamic prompt to include the user's role in the system prompt."""
    user_role = request.runtime.context.user_role
    return ExplainerAgentMiddleware.get_prompt_for_role(user_role)


def get_agent_with_custom_middlewares():
    """Create and return a knowledge base agent equipped with the knowledge base search tool.

    Returns:
        A LangChain agent configured with the knowledge base tool and a system prompt
        for providing reliable weather information.
    """
    model: BaseChatModel = ModelFactory.create_model(ModelInfo.DEFAULT_MODEL_TYPE.value)
    runtime_context_schema = cast(Any, Context)
    agent = create_agent(
        model=model,
        middleware=[
            user_role_prompt,
            dynamic_model_selection,
            SummarizationMiddleware(
                model=model,
                trigger=SUMMARIZATION_TRIGGER,
                keep=SUMMARIZATION_KEEP,
                summary_prompt='Summarize the conversation so far in a concise manner, '
                               'keeping the context relevant to the user\'s role '
                               'and the ongoing discussion.',
            ),
        ],
        context_schema=runtime_context_schema,
    )
    return agent


def get_agent_with_agent_middleware():
    """Create and return a knowledge base agent equipped with the knowledge base search tool.

    Returns:
        A LangChain agent configured with the knowledge base tool and a system prompt
        for providing reliable weather information.
    """
    model: BaseChatModel = ModelFactory.create_model(ModelInfo.DEFAULT_MODEL_TYPE.value)
    runtime_context_schema = cast(Any, Context)
    agent = create_agent(
        model=model,
        middleware=[ExplainerAgentMiddleware()],
        context_schema=runtime_context_schema,
    )
    return agent


__all__ = [
    'get_agent_with_custom_middlewares',
    'get_agent_with_agent_middleware',
]

def main():
    """Main function to demonstrate the explainer agent."""
    # pylint: disable=import-outside-toplevel
    # pylint: disable=cyclic-import
    from src.agents.explainer.agent_executor import execute_explainer_agent
    execute_explainer_agent()


if __name__ == '__main__':
    main()
