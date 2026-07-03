"""
This module contains a catalog of prompts for different use cases.
"""
# pylint: disable=invalid-name

from enum import Enum

class PromptCatalog(Enum):
    """
    A catalog of prompts for different use cases.
    """
    PROGRAMMING_SYSTEM_PROMPT = ('You are a helpful assistant for responding questions regarding '
                                 'programming with reliable, accurate and concise answers.')
    IMAGE_DESCRIPTION_SYSTEM_PROMPT = ('Describe the content of the image in detail, '
                                       'including objects, colors, and context.')
    WEATHER_SYSTEM_PROMPT = ('You are a helpful weather assistant that can provide useful and '
                             'reliable weather information for cities around the world')
    KNOWLEDGE_BASE_SYSTEM_PROMPT = ('You are a helpful knowledge base assistant. '
                                  'For questions about Macs, apples or laptops, '
                                  'first call the kb_search tool to retrieve relevant context, '
                                  'then answer succinctly,'
                                  'Maybe you have to use it multiple times before answering.')
    EXPLAINER_SYSTEM_PROMPT = 'You are a helpful assistant that provides explanations to concepts.'
    PROMPT_TEMPLATE_WITH_CITY_INPUT = 'What is the weather like in {city} city?'
    PROMPT_TEMPLATE_FOR_USER_CITY = 'What is the weather like?'
    PROMPT_KNOWLEDGE_BASE = ('What three fruits does the person like and '
                             'what three fruits does the person dislike?')
