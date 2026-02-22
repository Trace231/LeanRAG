from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from leanrag_explorer.types import QueryContext


class BaseQueryBuilder(ABC):
    """Strategy interface for multi-view query construction.

    Every implementation must return a dictionary so different retrievers
    can consume the query view they are specialized for.

    Expected keys include:
      - "formal_query": Lean-oriented query for formal retrievers.
      - "nl_query": natural-language query for LLM/NL retrievers.
    """

    @abstractmethod
    def build(self, context: QueryContext) -> Dict[str, str]:
        """Build query views from one proof context."""
        raise NotImplementedError

