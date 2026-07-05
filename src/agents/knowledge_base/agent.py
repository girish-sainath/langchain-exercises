"""Knowledge Base agent initialization and configuration."""
from dotenv import load_dotenv

from langchain.chat_models import BaseChatModel

from langchain.agents import create_agent

from src.tools.knowledge_base.tool import get_retriever_tool
from src.models.ModelInfo import ModelInfo
from src.models.ModelFactory import ModelFactory
from src.prompts.PromptCatalog import PromptCatalog


load_dotenv()


def get_agent():
    """Create and return a knowledge base agent equipped with the knowledge base search tool.

    Returns:
        A LangChain agent configured with the knowledge base tool and a system prompt
        for providing reliable weather information.
    """
    retriever_tool = get_retriever_tool()
    model: BaseChatModel = ModelFactory.create_model(ModelInfo.DEFAULT_MODEL_TYPE.value)
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
