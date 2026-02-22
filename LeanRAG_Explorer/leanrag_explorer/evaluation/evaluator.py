from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from leanrag_explorer.filters.accessibility import AccessibilityFilter
from leanrag_explorer.query_builders.base import BaseQueryBuilder
from leanrag_explorer.retrievers.base import BaseRetriever
from leanrag_explorer.types import EvalSample, RetrievalHit


@dataclass(frozen=True)
class EvalResult:
    """Container for aggregate recall metrics."""

    recall_at_1: float
    recall_at_10: float
    recall_at_50: float
    recall_at_1_filtered: float
    recall_at_10_filtered: float
    recall_at_50_filtered: float
    num_samples: int


def _hit_at_k(hits: List[RetrievalHit], gold_ids: set[str], k: int) -> int:
    topk = hits[:k]
    return 1 if any(h.premise_id in gold_ids for h in topk) else 0


class Evaluator:
    """Evaluator for QueryBuilder x Retriever combinations.

    The evaluator reports recall before and after DAG/accessibility filtering.
    This allows you to measure:
      1) raw retrieval quality
      2) legality-aware effective retrieval quality
    """

    def __init__(
        self,
        query_builder: BaseQueryBuilder,
        retriever: BaseRetriever,
        accessibility_filter: AccessibilityFilter,
    ) -> None:
        self.query_builder = query_builder
        self.retriever = retriever
        self.accessibility_filter = accessibility_filter

    def evaluate(self, dataset: Iterable[EvalSample], k: int = 100) -> EvalResult:
        total = 0
        r1 = r10 = r50 = 0
        r1f = r10f = r50f = 0

        for sample in dataset:
            total += 1
            query_dict: Dict[str, str] = self.query_builder.build(sample.context)
            hits = self.retriever.retrieve(query_dict, k=k)
            filtered_hits = self.accessibility_filter.filter_hits(sample.context, hits)
            gold = set(sample.gold_premise_ids)

            r1 += _hit_at_k(hits, gold, 1)
            r10 += _hit_at_k(hits, gold, 10)
            r50 += _hit_at_k(hits, gold, 50)

            r1f += _hit_at_k(filtered_hits, gold, 1)
            r10f += _hit_at_k(filtered_hits, gold, 10)
            r50f += _hit_at_k(filtered_hits, gold, 50)

        if total == 0:
            return EvalResult(0, 0, 0, 0, 0, 0, 0)

        return EvalResult(
            recall_at_1=r1 / total,
            recall_at_10=r10 / total,
            recall_at_50=r50 / total,
            recall_at_1_filtered=r1f / total,
            recall_at_10_filtered=r10f / total,
            recall_at_50_filtered=r50f / total,
            num_samples=total,
        )

