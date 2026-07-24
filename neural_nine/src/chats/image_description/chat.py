"""
Interactive terminal chat for image description using a chat model.
"""
from base64 import b64encode
from collections.abc import Buffer
from pathlib import Path

from dotenv import load_dotenv

from langchain.chat_models import BaseChatModel

from src.models.ModelInfo import ModelInfo
from src.models.ModelFactory import ModelFactory
from src.prompts.PromptCatalog import PromptCatalog


load_dotenv()


# Resolve path relative to this module's directory
DEFAULT_IMAGE_PATH = Path(__file__).parent / 'brain_logo.png'


def _get_image_as_base64(file_path: Path | str = DEFAULT_IMAGE_PATH) -> str | None:
    """Read image file and return its base64-encoded content.

    Args:
        file_path: Path to the image file to read and encode.

    Returns:
        Base64-encoded string of the image file, or None if the file is not found.
    """
    image_path = Path(file_path)
    if not image_path.exists():
        print(f'Error: Image file not found at {file_path}')
        return None

    with image_path.open('rb') as image_file:
        image_bytes: Buffer = image_file.read()
    return b64encode(image_bytes).decode('utf-8')


def execute_image_description_chat() -> None:
    """Run an interactive terminal chat loop for image description with print option."""
    chat_model: BaseChatModel = ModelFactory.create_model(ModelInfo.DEFAULT_MODEL_TYPE.value)

    image_base64 = _get_image_as_base64(DEFAULT_IMAGE_PATH)
    if image_base64 is None:
        return

    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': PromptCatalog.IMAGE_DESCRIPTION_SYSTEM_PROMPT.value,
                },
                {
                    'type': 'image',
                    'base64': image_base64,
                    'mime_type': 'image/png',
                },
            ]
        }
    ]

    chat_response = chat_model.invoke(messages)
    print('Bot:\n', chat_response.content)


__all__ = [
    'execute_image_description_chat',
]


def main():
    """Main function to run the image description chat."""
    execute_image_description_chat()


if __name__ == '__main__':
    main()
