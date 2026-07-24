"""
Knowledge Base agent execution and query handling.
"""
from src.agents.knowledge_base.agent import get_agent as knowledge_base_agent
from src.prompts.PromptCatalog import PromptCatalog


def execute_kb_agent() -> None:
    """Execute the knowledge base agent to retrieve and display relevant information.
    """

    query: str = input('Enter a query to get relevant information '
                       '(or leave blank to use default query): ')

    if not query:
        query = PromptCatalog.PROMPT_KNOWLEDGE_BASE.value

    print('\nBot: The agent is thinking....')
    print(f'\nBot: Calling the knowledge base tool to get '
          f'the relevant information for the query \'{query}\'...\n')

    agent = knowledge_base_agent()
    agent_response = agent.invoke(
        {
            'messages': [
                {
                    'role': 'user',
                    'content': query,
                }
            ],
        },
    )
    print('Bot: Start of Knowledge Base Agent Response:')
    print('Bot: ', agent_response)
    print('Bot: End of Knowledge Base Agent Response:\n')
    print('Bot:\n', agent_response['messages'][-1].content)


__all__ = [
    'execute_kb_agent',
]


if __name__ == '__main__':
    execute_kb_agent()
