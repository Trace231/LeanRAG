from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set

from leanrag_explorer.types import QueryContext, RetrievalHit


@dataclass
class AccessibilityFilter:
    """Legality filtering for premise candidates.

    This mock implementation exposes the same shape as a real Lean import-graph
    filter. You can later replace `allowed_map` with:
      - file-path based transitive import closure
      - theorem-position constraints (same-file past-only)
      - project-specific DAG service
    """

    allowed_map: Dict[str, Set[str]] = field(default_factory=dict)
    fallback_allow_all: bool = True

    def filter_hits(
        self,
        context: QueryContext,
        hits: Iterable[RetrievalHit],
    ) -> List[RetrievalHit]:
        """Keep only legally accessible premise IDs for this context.

        Contract:
          - The context should provide a stable key in metadata, for example:
            context.metadata["theorem_id"] or context.metadata["file_path"].
          - If no key is found and `fallback_allow_all` is True, we keep all hits.
        """

        theorem_id = str(context.metadata.get("theorem_id", ""))
        if theorem_id == "" or theorem_id not in self.allowed_map:
            return list(hits) if self.fallback_allow_all else []

        allowed = self.allowed_map[theorem_id]
        return [h for h in hits if h.premise_id in allowed]

