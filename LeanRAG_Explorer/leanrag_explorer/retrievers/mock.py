from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from leanrag_explorer.retrievers.base import BaseRetriever
from leanrag_explorer.types import RetrievalHit


@dataclass
class _StaticRetriever(BaseRetriever):
    """Utility retriever for dry-run and wiring tests."""

    index: List[RetrievalHit] = field(default_factory=list)
    query_key: str = "formal_query"

    def retrieve(self, query_dict: Dict[str, str], k: int = 100) -> List[RetrievalHit]:
        # In phase-2 scaffolding, we only validate routing and fusion.
        # Real implementations will use query_dict[self.query_key] for scoring.
        _ = query_dict.get(self.query_key, "")
        return self.index[:k]


class DenseRetriever(_StaticRetriever):
    """Placeholder dense retriever; consumes formal query by default."""

    query_key: str = "formal_query"


class SparseRetriever(_StaticRetriever):
    """Placeholder sparse retriever; configurable key for formal/nl query."""

    query_key: str = "formal_query"


class LLMRetriever(_StaticRetriever):
    """Placeholder LLM retriever; consumes nl_query by default."""

    query_key: str = "nl_query"

