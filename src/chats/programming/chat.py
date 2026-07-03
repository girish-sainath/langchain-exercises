"""Interactive programming assistant chat helpers with invoke and streaming modes."""

from collections.abc import Callable

from langchain_litellm import ChatLiteLLM

from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from dotenv import load_dotenv

from src.models.ModelInfo import ModelInfo
from src.prompts.PromptCatalog import PromptCatalog


load_dotenv()


EXIT_COMMAND = 'exit'


def _create_chat_model(
    model: str = ModelInfo.DEFAULT_MODEL.value,
    temperature: float = ModelInfo.DEFAULT_MODEL_TEMPERATURE.value,
    streaming: bool = False,
) -> ChatLiteLLM:
    """Create and configure the chat model used by the programming assistant."""
    return ChatLiteLLM(
        model=model,
        temperature=temperature,
        streaming=streaming,
    )


def _run_chat_session(
    chat_func: Callable[[list[AnyMessage]], AIMessage],
    system_prompt: str = PromptCatalog.PROGRAMMING_SYSTEM_PROMPT.value
) -> None:
    """Run an interactive terminal chat loop with shared conversation state."""
    conversations: list[AnyMessage] = [SystemMessage(content=system_prompt)]

    print('\nProgramming Assistant...')
    print('Type "exit" to quit...')

    while True:
        user_input = input('\nBot: Enter your question\nYou: ').strip()
        if not user_input:
            continue
        if user_input.lower() == EXIT_COMMAND:
            break

        conversations.append(HumanMessage(content=user_input))
        print('\nBot: Thinking...\n')
        response: AIMessage = chat_func(conversations)

        conversations.append(response)
        print('Bot:\n', response.content)


def _build_invoke_chat(
        chat_model_instance: ChatLiteLLM) -> Callable[[list[AnyMessage]], AIMessage]:
    """Build a chat function that returns complete responses in one invoke call."""
    def _chat_with_invoke(conversations: list[AnyMessage]) -> AIMessage:
        return chat_model_instance.invoke(conversations)

    return _chat_with_invoke


def _build_stream_chat(
    chat_model_instance: ChatLiteLLM) -> Callable[[list[AnyMessage]], AIMessage]:
    """Build a chat function that streams output while accumulating the final message."""
    def _chat_with_stream(conversations: list[AnyMessage]) -> AIMessage:
        print('Streaming started...')
        full_response = ''
        for chunk in chat_model_instance.stream(conversations):
            print(chunk.content, end='', flush=True)
            full_response += chunk.content
        return AIMessage(content=full_response)

    return _chat_with_stream


def chat_with_print_option() -> None:
    """Start the programming assistant using non-streaming model invocation."""
    model = _create_chat_model()
    chat_func = _build_invoke_chat(model)
    _run_chat_session(
        chat_func,
        system_prompt=PromptCatalog.PROGRAMMING_SYSTEM_PROMPT.value,
    )


def chat_with_stream_option() -> None:
    """Start the programming assistant with token-by-token streamed output."""
    model: ChatLiteLLM = _create_chat_model(streaming=True)
    chat_func = _build_stream_chat(model)
    _run_chat_session(
        chat_func,
        system_prompt=PromptCatalog.PROGRAMMING_SYSTEM_PROMPT.value,
    )


def execute_programming_chat() -> None:
    """Run the programming assistant with user-selected output mode."""
    chat_option = input('Bot: Enter the chat option '
                        '(1 for print, 2 for stream)\nYou: ')
    if chat_option == '1':
        chat_with_print_option()
    elif chat_option == '2':
        chat_with_stream_option()
    else:
        print('Bot: Invalid chat option. '
              'Please enter 1 for print or 2 for stream.')


__all__ = [
    'execute_programming_chat',
]


def main() -> None:
    """Run the programming assistant with streaming output."""
    execute_programming_chat()


if __name__ == '__main__':
    main()
