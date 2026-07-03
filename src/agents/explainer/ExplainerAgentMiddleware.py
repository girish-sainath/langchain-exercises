"""Middleware hooks for dynamic model selection in the explainer agent."""
# pylint: disable=invalid-name

from collections.abc import Callable
from time import perf_counter
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_litellm import ChatLiteLLM
from langgraph.runtime import Runtime

from src.models.ModelInfo import ModelInfo
from src.prompts.PromptCatalog import PromptCatalog


class ExplainerAgentMiddleware(AgentMiddleware):
    """Middleware that logs hooks and swaps models by user role."""

    def __init__(self):
        """Initialize middleware timing state."""
        super().__init__()
        self.start_time = perf_counter()
        print('ExplainerAgentMiddleware: initialized...')


    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Run before the agent starts processing a request."""
        elapsed = perf_counter() - self.start_time
        print(f'ExplainerAgentMiddleware: before_agent called after {elapsed:.3f}s...')


    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Run before the model is invoked (logging only)."""
        elapsed = perf_counter() - self.start_time
        print(f'ExplainerAgentMiddleware: before_model called after {elapsed:.3f}s...')


    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Select model by user role and forward the updated request.

        Uses ``request.override(model=...)`` to switch models safely.
        """
        elapsed = perf_counter() - self.start_time
        print(f'ExplainerAgentMiddleware: wrap_model_call called after {elapsed:.3f}s...')
        user_role = getattr(request.runtime.context, 'user_role', 'beginner')
        model = ExplainerAgentMiddleware.get_model_for_role(user_role, request.model)
        return handler(request.override(model=model))


    def dynamic_prompt(self, request: ModelRequest) -> str:
        """Dynamic prompt to include the user's role in the system prompt."""
        elapsed = perf_counter() - self.start_time
        print(f'ExplainerAgentMiddleware: wrap_model_call called after {elapsed:.3f}s...')
        user_role = getattr(request.runtime.context, 'user_role', 'beginner')
        return ExplainerAgentMiddleware.get_prompt_for_role(user_role)


    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Run after the model has produced its output."""
        elapsed = perf_counter() - self.start_time
        print(f'ExplainerAgentMiddleware: after_model called after {elapsed:.3f}s...')


    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Run after the agent finishes processing a request."""
        elapsed = perf_counter() - self.start_time
        print(f'ExplainerAgentMiddleware: after_agent called after {elapsed:.3f}s...')

    @staticmethod
    def get_prompt_for_role(user_role: str) -> str:
        """Return the system prompt based on the user's role."""
        match user_role:
            case 'expert':
                return (f'{PromptCatalog.EXPLAINER_SYSTEM_PROMPT.value} '
                        f'Provide detailed and technical explanations.')
            case 'beginner':
                return (f'{PromptCatalog.EXPLAINER_SYSTEM_PROMPT.value} '
                        f'Provide simple and easy-to-understand explanations.')
            case 'child':
                return (f'{PromptCatalog.EXPLAINER_SYSTEM_PROMPT.value} '
                        f'Provide explanations suitable for a 5 year old child.')
            case '_':
                return PromptCatalog.EXPLAINER_SYSTEM_PROMPT.value
        return PromptCatalog.EXPLAINER_SYSTEM_PROMPT.value

    @staticmethod
    def get_model_for_role(user_role: str, model):
        """Return the model name based on the user's role."""

        if user_role == 'expert':
            model = ChatLiteLLM(
                model=ModelInfo.ADVANCED_MODEL.value,
            )
        elif user_role == 'child':
            model = ChatLiteLLM(
                model=ModelInfo.BASIC_MODEL.value,
                temperature=ModelInfo.BASIC_MODEL_TEMPERATURE.value,
            )

        return model
