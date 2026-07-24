"""Main entry point for the LangChain exercises application."""

from src.agents.weather.agent_executor import execute_weather_agent
from src.agents.explainer.agent_executor import execute_explainer_agent
from src.agents.knowledge_base.agent_executor import execute_kb_agent
from src.chats.programming.chat import execute_programming_chat
from src.chats.image_description.chat import execute_image_description_chat

def main() -> None:
    """Display a menu and dispatch to weather, programming chat, or image description flows."""
    print('Type "exit" to quit...')
    while True:
        option: str = input('\nBot: Enter the option '
                            '(1 for weather, 2 for programming, 3 for image_description '
                            '4 for explainer, 5 for knowledge_base)\nYou: ')
        if option == 'exit':
            print('\nBot: Goodbye!')
            break
        if option == '1':
            execute_weather_agent()
        elif option == '2':
            execute_programming_chat()
        elif option == '3':
            execute_image_description_chat()
        elif option == '4':
            execute_explainer_agent()
        elif option == '5':
            execute_kb_agent()
        else:
            print('Bot: Invalid option. Please enter 1 for weather or 2 for programming '
                  'or 4 for explainer or 5 for knowledge_base.')


if __name__ == '__main__':
    main()
