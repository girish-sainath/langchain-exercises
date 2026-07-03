"""Explainer agent execution and query handling."""
from typing import Any, cast

from src.agents.explainer.Context import Context
from src.agents.explainer.agent import (
    get_agent_with_custom_middlewares,
    get_agent_with_agent_middleware,
)

DEFAULT_PROMPT = 'Explain recursion in Python.'
DEFAULT_USER_ROLE = 'child'


def execute_explainer_agent() -> None:
    """Execute the explainer agent and display the final response.
    """

    agent_type = input('Bot: Choose agent type '
                       '(1 for custom middlewares, 2 for agent middleware): ')
    agent = get_agent_with_custom_middlewares()
    if agent_type == '1':
        agent = get_agent_with_custom_middlewares()
    elif agent_type == '2':
        agent = get_agent_with_agent_middleware()
    else:
        print('Bot: Invalid choice. Defaulting to custom middlewares.')

    print('Bot: Explainer Agent is ready! Type "exit" to quit.')
    while True:
        user_role:str = input('\nBot: What is your role? '
                              'Provide expert or beginner or child\nYou: ')
        if user_role == 'exit':
            print('\nBot: Exiting the explainer agent. Goodbye!')
            break

        if user_role not in ['expert', 'beginner', 'child']:
            print('Bot: Invalid role. Defaulting to beginner.')
            user_role = 'beginner'

        query_input = input('\nBot: What would you like to ask?\nYou: ')
        if not query_input:
            print('\nBot: No query provided...')
            return

        agent_input = cast(Any, {
            'messages': [
                {
                    'role': 'user',
                    'content': query_input,
                }
            ],
        })
        runtime_context = cast(Any, Context(user_role=user_role))
        agent_response = agent.invoke(
            agent_input,
            context=runtime_context,
        )
        print('Bot: ', agent_response['messages'][-1].content)
        print('\nBot: Model used:', agent_response['messages'][-1].response_metadata['model_name'])


__all__ = [
    'execute_explainer_agent',
]


def main() -> None:
    """Run the explainer agent executor from the command line."""
    execute_explainer_agent()


if __name__ == '__main__':
    main()
