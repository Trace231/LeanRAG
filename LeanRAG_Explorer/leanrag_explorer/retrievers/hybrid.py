from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Sequence

from leanrag_explorer.retrievers.base import BaseRetriever
from leanrag_explorer.types import RetrievalHit


class FusionMethod(str, Enum):
    LINEAR = "linear"
    RRF = "rrf"


def normalize_scores_minmax(hits: Sequence[RetrievalHit]) -> List[RetrievalHit]:
    """Normalize retriever scores to [0, 1] using min-max scaling.

    Why this is needed:
      Dense cosine scores, BM25 scores, and LLM confidence scores are usually
      on different numeric scales. Linear fusion needs a common scale to avoid
      one retriever dominating only due to score magnitude.
    """

    if not hits:
        return []
    scores = [h.score for h in hits]
    low = min(scores)
    high = max(scores)
    if high == low:
        return [
            RetrievalHit(
                premise_id=h.premise_id,
                score=1.0,
                dense_score=h.dense_score,
                sparse_score=h.sparse_score,
                rank_in_dense=h.rank_in_dense,
                rank_in_sparse=h.rank_in_sparse,
                premise_text=h.premise_text,
                raw=h.raw,
            )
            for h in hits
        ]
    return [
        RetrievalHit(
            premise_id=h.premise_id,
            score=(h.score - low) / (high - low),
            dense_score=h.dense_score,
            sparse_score=h.sparse_score,
            rank_in_dense=h.rank_in_dense,
            rank_in_sparse=h.rank_in_sparse,
            premise_text=h.premise_text,
            raw=h.raw,
        )
        for h in hits
    ]


def _rank_map(hits: Sequence[RetrievalHit]) -> Dict[str, int]:
    """Create rank map: premise_id -> 1-based rank."""
    return {h.premise_id: i + 1 for i, h in enumerate(hits)}


def rrf_fuse(
    hit_lists: Iterable[Sequence[RetrievalHit]],
    top_k: int,
    rrf_k: int = 60,
) -> List[RetrievalHit]:
    """Reciprocal Rank Fusion (RRF).

    Formula:
        RRF(d) = sum_i 1 / (rrf_k + rank_i(d))

    Design choices:
      - Rank-based fusion is robust when score scales are not comparable.
      - Missing documents in a retriever contribute 0 in that channel.
      - We keep the highest available raw score/premise_text as metadata.
    """

    lists = [list(h) for h in hit_lists]
    rank_maps = [_rank_map(h) for h in lists]
    all_ids = {h.premise_id for hits in lists for h in hits}

    best_payload: Dict[str, RetrievalHit] = {}
    for hits in lists:
        for h in hits:
            # Keep the max-score payload for readability/debugging.
            if (
                h.premise_id not in best_payload
                or h.score > best_payload[h.premise_id].score
            ):
                best_payload[h.premise_id] = h

    fused: List[RetrievalHit] = []
    for pid in all_ids:
        score = 0.0
        for rm in rank_maps:
            rank = rm.get(pid)
            if rank is not None:
                score += 1.0 / (rrf_k + rank)
        payload = best_payload[pid]
        fused.append(
            RetrievalHit(
                premise_id=pid,
                score=score,
                premise_text=payload.premise_text,
                raw=payload.raw,
            )
        )

    fused.sort(key=lambda h: h.score, reverse=True)
    return fused[:top_k]


@dataclass
class HybridRetriever(BaseRetriever):
    """Late-fusion retriever orchestrating multiple specialized retrievers.

    Typical setup:
      - DenseRetriever consumes formal_query
      - SparseRetriever consumes formal_query or nl_query
      - LLMRetriever consumes nl_query
    """

    retrievers: List[BaseRetriever]
    fusion: FusionMethod = FusionMethod.RRF
    alpha: float = 0.5
    rrf_k: int = 60

    def _linear_fuse_two(
        self,
        name_a: str,
        name_b: str,
        hits_a: Sequence[RetrievalHit],
        hits_b: Sequence[RetrievalHit],
        top_k: int,
    ) -> List[RetrievalHit]:
        """Linear weighting for two retriever streams.

        Formula:
            score(d) = alpha * score_A(d) + (1 - alpha) * score_B(d)

        Important:
          We min-max normalize both streams before weighting to make alpha
          meaningful. Without normalization, score ranges are not comparable.
        """

        a_norm = normalize_scores_minmax(hits_a)
        b_norm = normalize_scores_minmax(hits_b)

        map_a = {h.premise_id: h for h in a_norm}
        map_b = {h.premise_id: h for h in b_norm}
        all_ids = set(map_a.keys()) | set(map_b.keys())
        rank_a = _rank_map(hits_a)
        rank_b = _rank_map(hits_b)

        fused: List[RetrievalHit] = []
        for pid in all_ids:
            sa = map_a[pid].score if pid in map_a else 0.0
            sb = map_b[pid].score if pid in map_b else 0.0
            score = self.alpha * sa + (1.0 - self.alpha) * sb

            # Keep whichever payload exists for traceability.
            payload = map_a.get(pid) or map_b[pid]
            dense_score = None
            sparse_score = None
            rank_in_dense = None
            rank_in_sparse = None

            if "dense" in name_a.lower():
                dense_score = map_a.get(pid).score if pid in map_a else None
                rank_in_dense = rank_a.get(pid)
            elif "sparse" in name_a.lower():
                sparse_score = map_a.get(pid).score if pid in map_a else None
                rank_in_sparse = rank_a.get(pid)

            if "dense" in name_b.lower():
                dense_score = map_b.get(pid).score if pid in map_b else dense_score
                rank_in_dense = rank_b.get(pid) or rank_in_dense
            elif "sparse" in name_b.lower():
                sparse_score = map_b.get(pid).score if pid in map_b else sparse_score
                rank_in_sparse = rank_b.get(pid) or rank_in_sparse

            fused.append(
                RetrievalHit(
                    premise_id=pid,
                    score=score,
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                    rank_in_dense=rank_in_dense,
                    rank_in_sparse=rank_in_sparse,
                    premise_text=payload.premise_text,
                    raw=payload.raw,
                )
            )

        fused.sort(key=lambda h: h.score, reverse=True)
        return fused[:top_k]

    def retrieve(self, query_dict: Dict[str, str], k: int = 100) -> List[RetrievalHit]:
        if not self.retrievers:
            return []

        named_streams = [
            (type(r).__name__, r.retrieve(query_dict, k=k)) for r in self.retrievers
        ]
        named_streams = [(n, s) for (n, s) in named_streams if s]
        if not named_streams:
            return []
        if len(named_streams) == 1:
            return named_streams[0][1][:k]

        if self.fusion == FusionMethod.RRF:
            streams = [s for _, s in named_streams]
            fused = rrf_fuse(streams, top_k=k, rrf_k=self.rrf_k)
            # Add provenance ranks when dense/sparse streams are present.
            dense_hits = next(
                (s for n, s in named_streams if "dense" in n.lower()), []
            )
            sparse_hits = next(
                (s for n, s in named_streams if "sparse" in n.lower()), []
            )
            dense_rank = _rank_map(dense_hits)
            sparse_rank = _rank_map(sparse_hits)
            dense_map = {h.premise_id: h.score for h in dense_hits}
            sparse_map = {h.premise_id: h.score for h in sparse_hits}
            return [
                RetrievalHit(
                    premise_id=h.premise_id,
                    score=h.score,
                    dense_score=dense_map.get(h.premise_id),
                    sparse_score=sparse_map.get(h.premise_id),
                    rank_in_dense=dense_rank.get(h.premise_id),
                    rank_in_sparse=sparse_rank.get(h.premise_id),
                    premise_text=h.premise_text,
                    raw=h.raw,
                )
                for h in fused
            ]

        # Linear is intentionally defined for exactly two streams for clarity.
        # For >2 streams, use RRF or extend this method with learned weights.
        if len(named_streams) != 2:
            raise ValueError("Linear fusion currently supports exactly 2 streams.")
        (name_a, stream_a), (name_b, stream_b) = named_streams
        return self._linear_fuse_two(
            name_a, name_b, stream_a, stream_b, top_k=k
        )

