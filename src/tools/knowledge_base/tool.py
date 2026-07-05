"""
Knowledge base tool for the RAG example.
"""

from dotenv import load_dotenv

from langchain_core.tools import create_retriever_tool, StructuredTool
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever, VectorStore
from langchain_chroma import Chroma  # pylint: disable=import-error

from src.models.ModelInfo import ModelInfo
from src.models.ModelFactory import ModelFactory

load_dotenv()


def _get_vector_store() -> VectorStore:
    """Create and return a vector store for the knowledge base.

    Returns:
        A Chroma vector store containing the knowledge base documents.
    """
    embedding: Embeddings = ModelFactory.create_embedding_model()

    texts: list[str] = [
        'I love apples',
        'I enjoy oranges',
        'I think pears taste great.',
        'I hate bananas.',
        'I dislike raspberries.',
        'I despise mangoes.',
        'I love Linux',
        'I hate Windows.',
    ]
    return Chroma.from_texts(texts, embedding=embedding)


def _get_retriever() -> VectorStoreRetriever:
    """Create and return a retriever for the knowledge base.

    Returns:
        A retriever that can be used to search the knowledge base.
    """
    vector_store: VectorStore = _get_vector_store()
    return vector_store.as_retriever(search_kwargs={'k': 3})


def get_retriever_tool() -> StructuredTool:
    """Create and return a retriever tool for the knowledge base.

    Returns:
        A retriever tool that can be used to search the knowledge base.
    """
    retriever = _get_retriever()
    return create_retriever_tool(
        retriever,
        name='kb_search',
        description='Search the small fruit knowledge base for information about fruit preferences.'
    )


__all__ = [
    'get_retriever_tool'
]


def main() -> None:
    """
    Demonstrate the functionality of the knowledge base tool.
    """
    vector_store = _get_vector_store()
    questions = [
        'What fruits does the person like?',
        'What fruits does the person hate?',
    ]

    for question in questions:
        documents = vector_store.similarity_search(question, k=3)
        print(f'Output for question: "{question}"')
        print(documents)
        print('\nOutput (page content only):')
        for doc in documents:
            print(doc.id, doc.page_content)
        print('\n' + '-' * 50 + '\n')


if __name__ == '__main__':
    main()
