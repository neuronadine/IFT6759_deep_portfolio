from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")

class Chunker(ABC, Generic[T]):
    """Base class for file chunking strategies"""

    name: str
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    @abstractmethod
    def chunk(self, content: T) -> list[T]:
        """splits file into chunks"""
        pass


    