"""
User context class to hold the context of the conversation.
"""
# pylint: disable=invalid-name

from dataclasses import dataclass

@dataclass
class Context:
    """
    Context class to hold the context of the conversation.
    """
    user_role: str
