from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class QueryContext:
    """Raw context used to build one or more query views."""

    theorem_statement: str
    current_state: str
    recent_tactics: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    goal: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalHit:
    """A normalized retrieval output item.

    Notes:
        - `premise_id` should be a stable identifier (e.g., full_name or UUID).
        - `score` is the final score used for sorting/ranking in current stage.
        - `dense_score`/`sparse_score` and rank fields store provenance for
          downstream error analysis and fusion introspection.
        - `raw` stores engine-specific payload for debugging.
    """

    premise_id: str
    score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rank_in_dense: Optional[int] = None
    rank_in_sparse: Optional[int] = None
    premise_text: Optional[str] = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Backward-compatible alias used by early scaffolding code."""
        return self.premise_id


@dataclass(frozen=True)
class EvalSample:
    """Evaluation unit for premise selection."""

    context: QueryContext
    gold_premise_ids: List[str]

