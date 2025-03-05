from .base import Chunker
from .LangChainChunker import LangChainChunker

class ChunkerFactory:
    """Factory for creating document chunkers"""

    @staticmethod
    def create_chunker(name: str, **kwargs) -> Chunker:
        """Creates a file chunker instance based on name and arguments"""


        chunker_map = {"langchain-markdown": LangChainChunker}

        if name not in chunker_map:
            raise ValueError(f"Unsupported chunker function: {name}")