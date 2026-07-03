"""
User lookup tool implementations used by weather agents.
"""

from langchain.tools import tool, ToolRuntime

from src.agents.weather.Context import Context


def _locate_user(user_id: str) -> str:
    """
    Resolve a user's city using the user id.

    Args:
        user_id: ID of the user.

    Returns:
        The mapped city name for known user IDs, otherwise ``"Unknown"``.
    """
    match user_id:
        case 'ABC123':
            return 'Vienna'
        case 'XYZ456':
            return 'London'
        case 'HJK111':
            return 'Paris'
        case _:
            return 'Unknown'


@tool('locate_user', description='Look up a user\'s city based on the context')
def locate_user(runtime: ToolRuntime[Context]) -> str:
    """Tool to resolve a user's city using the runtime context.

    Args:
        runtime: Tool runtime containing the weather-agent `Context`.

    Returns:
        The mapped city name for known user IDs, otherwise ``"Unknown"``.
    """
    return _locate_user(runtime.context.user_id)


__all__ = [
    'locate_user',
]


def main() -> None:
    """Test the locate_user tool."""
    user_id: str = input('Enter a User ID to look up the city: ')
    city: str = _locate_user(user_id)
    print(f'User ID {user_id} is mapped to city: {city}')


if __name__ == '__main__':
    main()
