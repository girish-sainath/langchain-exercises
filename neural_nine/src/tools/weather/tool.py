"""
Weather tool integration used by LangChain agents.
"""

from typing import Any
from requests import get, Response, RequestException

from langchain.tools import tool


def _get_weather(city: str) -> dict[str, Any]:
    """
    Fetch current weather data for a city from wttr.in.

    Args:
        city: City name used in the weather service query.

    Returns:
        A JSON-like dictionary from the weather API, or an error payload when
        the HTTP request fails.
    """
    try:
        response: Response = get(
            url=f'https://wttr.in/{city}?format=j1',
            timeout=10,
            headers={'Accept': 'application/json'},
        )
        response.raise_for_status()
        return response.json()
    except RequestException as e:
        return {
            'error': f'Weather service request failed with error: {e}',
            'city': city,
        }


@tool('get_weather', description='Get the current weather for a given city', return_direct=False)
def get_weather(city: str) -> dict[str, Any]:
    """
    Tool function to fetch current weather data for a city from wttr.in.

    Args:
        city: City name used in the weather service query.

    Returns:
        A JSON-like dictionary from the weather API, or an error payload when
        the HTTP request fails.
    """
    return _get_weather(city)


__all__ = [
    'get_weather',
]


def main() -> None:
    """Test the get_weather tool."""
    city: str = input('Enter a city name to get the current weather: ')
    result: dict[str, Any] = _get_weather(city)
    print(f'Weather data for {city}: {result}')


if __name__ == '__main__':
    main()
