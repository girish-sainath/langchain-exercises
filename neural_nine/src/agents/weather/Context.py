"""
Context model for the weather agent.
"""
# pylint: disable=invalid-name

from dataclasses import dataclass


@dataclass
class Context:
    """Runtime context for weather-agent requests.

    Attributes:
        user_id: Unique identifier for the requesting user.
    """
    user_id: str
