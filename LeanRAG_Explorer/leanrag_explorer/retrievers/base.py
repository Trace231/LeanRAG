from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from leanrag_explorer.types import RetrievalHit


class BaseRetriever(ABC):
    """Base interface for all retrievers.

    Implementations can choose which query view(s) to consume
    (e.g., formal_query, nl_query).
    """

    @abstractmethod
    def retrieve(self, query_dict: Dict[str, str], k: int = 100) -> List[RetrievalHit]:
        """Return top-k premise candidates with retriever-native scores."""
        raise NotImplementedError

