"""
Weather agent execution and query handling.
"""
from typing import Any
from uuid import uuid4

from src.agents.weather.Context import Context
from src.agents.weather.agent import get_agent_with_city_input, get_agent_with_user_memory

from src.prompts.PromptCatalog import PromptCatalog


def _get_prompt_with_city_input(template: str, city: str) -> str:
    """Format a prompt template with the given city name.

    Args:
        template: A format string with a {city} placeholder.
        city: The city name to insert into the template.

    Returns:
        The formatted prompt string.
    """
    return template.format(city=city)


def _runtime_config(thread_id = None) -> dict[str, dict[str, Any]]:
    if thread_id is None:
        thread_id = str(uuid4())
    return {'configurable': {'thread_id': thread_id}}


def execute_weather_agent() -> None:
    """Execute the weather agent to retrieve and display weather information.
    """

    city: str = input('Enter a city name to get the weather information '
                      '(or leave blank to use User ID): ')

    if not city:
        user_id: str = input('\nBot: Provide your User ID\nYou: ')
        if not user_id:
            print('\nBot: No User ID provided. Using default User ID: ABC123')
            user_id = 'ABC123'

        print('\nBot: The agent is thinking....')
        print('\nBot: Calling the weather tool to get '
              'the weather information for your city...\n')

        query: str = PromptCatalog.PROMPT_TEMPLATE_FOR_USER_CITY.value
        config: dict[str, dict[str, Any]] = _runtime_config('1')
        agent = get_agent_with_user_memory()
        agent_response = agent.invoke(
            {
                'messages': [
                    {
                        'role': 'user',
                        'content': query,
                    }
                ],
            },
            config=config,
            context=Context(user_id=user_id),
        )
        print('Start: Agent Response:')
        print(agent_response)
        print('End: Agent Response:\n')
        if not agent_response.get('structured_response'):
            print(agent_response['messages'][-1].content)
        else:
            print(agent_response['structured_response'])
    else:
        print('\nBot: The agent is thinking....')
        print(f'\nBot: Calling the weather tool to get '
              f'the weather information for \'{city}\' city...\n')
        query_for_city: str = _get_prompt_with_city_input(
            PromptCatalog.PROMPT_TEMPLATE_WITH_CITY_INPUT.value,
            city,
        )
        agent = get_agent_with_city_input()
        agent_response = agent.invoke(
            {
                'messages': [
                    {
                        'role': 'user',
                        'content': query_for_city,
                    }
                ],
            },
        )
        print('Bot: Start of Weather Agent Response:')
        print('Bot: ', agent_response)
        print('Bot: End of Weather Agent Response:\n')
        print('Bot:\n', agent_response['messages'][-1].content)


__all__ = [
    'execute_weather_agent',
]


def main() -> None:
    """
    Test the weather agent execute with city input.
    """
    execute_weather_agent()


if __name__ == '__main__':
    main()
