"""
This module provides a function to retrieve model information from environment variables.
"""
# pylint: disable=invalid-name

from enum import Enum
import os
from dotenv import load_dotenv



load_dotenv()

class ModelInfo(Enum):
    """
    Enum for model information, including default, advanced, and basic models,
    as well as the default embedding model and temperature settings.
    """
    DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'claude-sonnet-4-6')
    ADVANCED_MODEL = os.getenv('ADVANCED_MODEL', 'claude-opus-4-7')
    BASIC_MODEL = os.getenv('BASIC_MODEL', 'claude-haiku-4-5')
    DEFAULT_EMBEDDING_MODEL = os.getenv('DEFAULT_EMBEDDING_MODEL', 'text-embedding-3-large')
    DEFAULT_MODEL_TEMPERATURE = float(os.getenv('DEFAULT_MODEL_TEMPERATURE', '0.5'))
    BASIC_MODEL_TEMPERATURE = float(os.getenv('BASIC_MODEL_TEMPERATURE', '0.7'))
