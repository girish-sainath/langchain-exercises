"""Knowledge Base agent initialization and configuration."""
from dotenv import load_dotenv

from langchain_litellm import ChatLiteLLM

from langchain.agents import create_agent

from src.tools.knowledge_base.tool import get_retriever_tool
from src.models.ModelInfo import ModelInfo
from src.prompts.PromptCatalog import PromptCatalog


load_dotenv()


def _create_chat_model(
    model: str = ModelInfo.DEFAULT_MODEL.value,
    temperature: float = ModelInfo.DEFAULT_MODEL_TEMPERATURE.value,
) -> ChatLiteLLM:
    """Create and configure the chat model used by the knowledge base assistant."""
    return ChatLiteLLM(
        model=model,
        temperature=temperature,
    )


def get_agent():
    """Create and return a knowledge base agent equipped with the knowledge base search tool.

    Returns:
        A LangChain agent configured with the knowledge base tool and a system prompt
        for providing reliable weather information.
    """
    retriever_tool = get_retriever_tool()
    model: ChatLiteLLM = _create_chat_model()
    agent = create_agent(
        model=model,
        tools=[retriever_tool],
        system_prompt = PromptCatalog.KNOWLEDGE_BASE_SYSTEM_PROMPT.value,
    )

    return agent


__all__ = [
    'get_agent',
]


def main():
    """Test the knowledge base agent with a query input."""
    # pylint: disable=import-outside-toplevel
    # pylint: disable=cyclic-import
    from src.agents.knowledge_base.agent_executor import execute_kb_agent
    execute_kb_agent()


if __name__ == '__main__':
    main()
