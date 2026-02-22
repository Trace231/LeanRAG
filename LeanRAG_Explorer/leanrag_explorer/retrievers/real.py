from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi

from leanrag_explorer.retrievers.base import BaseRetriever
from leanrag_explorer.types import RetrievalHit


def _whitespace_tokenize(text: str) -> List[str]:
    """Simple tokenizer baseline for BM25.

    We keep this deliberately simple now (split by whitespace and lowercase)
    so the ablation remains transparent. Lean-specific tokenization can be
    added later (e.g., preserving `Nat.succ`, `⊢`, symbols).
    """

    return [tok.strip().lower() for tok in text.split() if tok.strip()]


@dataclass
class RealSparseRetriever(BaseRetriever):
    """BM25 sparse retriever using `rank_bm25.BM25Okapi`."""

    corpus_dict: Dict[str, str]
    query_key: str = "formal_query"

    def __post_init__(self) -> None:
        self.premise_ids: List[str] = list(self.corpus_dict.keys())
        self.documents: List[str] = [self.corpus_dict[pid] for pid in self.premise_ids]
        tokenized = [_whitespace_tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query_dict: Dict[str, str], k: int = 100) -> List[RetrievalHit]:
        query = query_dict.get(self.query_key, "")
        q_tokens = _whitespace_tokenize(query)
        if not q_tokens:
            return []

        scores = np.asarray(self.bm25.get_scores(q_tokens), dtype=float)
        top_idx = np.argsort(-scores)[:k]

        hits: List[RetrievalHit] = []
        for rank, idx in enumerate(top_idx, start=1):
            pid = self.premise_ids[int(idx)]
            score = float(scores[int(idx)])
            hits.append(
                RetrievalHit(
                    premise_id=pid,
                    score=score,
                    sparse_score=score,
                    rank_in_sparse=rank,
                    premise_text=self.corpus_dict[pid],
                )
            )
        return hits


class ByT5RetrieverProtocol(Protocol):
    """Adapter protocol for existing ByT5 retriever implementation."""

    def retrieve_premises(self, state: str) -> List[Tuple[str, float]]:
        ...


@dataclass
class RealDenseRetriever(BaseRetriever):
    """Adapter to integrate an existing ByT5 premise retriever.

    Expected wrapped API:
        retriever.retrieve_premises(state: str) -> List[Tuple[premise_id, score]]
    """

    by_t5_retriever: ByT5RetrieverProtocol

    def retrieve(self, query_dict: Dict[str, str], k: int = 100) -> List[RetrievalHit]:
        # IMPORTANT: strict routing to formal query to avoid OOD drift.
        formal_query = query_dict.get("formal_query", "")
        if not formal_query:
            return []

        pairs = self.by_t5_retriever.retrieve_premises(formal_query)
        hits: List[RetrievalHit] = []
        for rank, (premise_id, score) in enumerate(pairs[:k], start=1):
            hits.append(
                RetrievalHit(
                    premise_id=str(premise_id),
                    score=float(score),
                    dense_score=float(score),
                    rank_in_dense=rank,
                )
            )
        return hits


def _extract_json_array(text: str) -> Optional[List[Mapping[str, object]]]:
    """Try to parse an LLM response containing a JSON array."""

    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, Mapping)]
    except Exception:
        pass

    # Fallback: attempt to capture the first [...] block.
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        if isinstance(data, list):
            return [x for x in data if isinstance(x, Mapping)]
    except Exception:
        return None
    return None


@dataclass
class LLMRetriever(BaseRetriever):
    """LLM-based retriever operating on `nl_query`.

    Mechanism:
      1) Build candidate list from provided corpus.
      2) Ask LLM to select the most relevant premise IDs for the query.
      3) Parse response and convert to `RetrievalHit`.
    """

    corpus_dict: Dict[str, str]
    model: str = "gpt-4o-mini"
    query_key: str = "nl_query"
    candidate_ids: Optional[List[str]] = None
    max_candidates: int = 200
    max_output_tokens: int = 600
    temperature: float = 0.0

    def __post_init__(self) -> None:
        self.client = OpenAI()

    def _build_candidates(self, candidate_ids: Optional[Sequence[str]]) -> List[str]:
        if candidate_ids is None:
            return list(self.corpus_dict.keys())[: self.max_candidates]
        selected = [pid for pid in candidate_ids if pid in self.corpus_dict]
        return selected[: self.max_candidates]

    def _prompt(self, nl_query: str, candidate_ids: List[str]) -> str:
        lines = []
        for pid in candidate_ids:
            text = self.corpus_dict[pid].replace("\n", " ").strip()
            lines.append(f'- {pid}: "{text[:260]}"')
        cands = "\n".join(lines)
        return (
            "You are a retrieval reranker for Lean theorem proving.\n"
            "Given a natural-language query and candidate premises, return the most relevant premise IDs.\n"
            "Output STRICT JSON array only. Each item must be:\n"
            '{"premise_id": "<id>", "score": <float between 0 and 1>}.\n'
            "Sort by relevance descending. Return at most 50 items.\n\n"
            f"Query:\n{nl_query}\n\n"
            f"Candidates:\n{cands}\n"
        )

    def retrieve(self, query_dict: Dict[str, str], k: int = 100) -> List[RetrievalHit]:
        nl_query = query_dict.get(self.query_key, "")
        if not nl_query.strip():
            return []

        candidate_ids = self._build_candidates(self.candidate_ids)
        if not candidate_ids:
            return []

        prompt = self._prompt(nl_query, candidate_ids)
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=self.max_output_tokens,
                temperature=self.temperature,
            )
            text = (resp.output_text or "").strip()
        except Exception:
            return []

        parsed = _extract_json_array(text)
        if parsed is None:
            return []

        hits: List[RetrievalHit] = []
        for rank, item in enumerate(parsed, start=1):
            premise_id = str(item.get("premise_id", "")).strip()
            if not premise_id or premise_id not in self.corpus_dict:
                continue
            try:
                score = float(item.get("score", 0.0))
            except Exception:
                score = 0.0
            hits.append(
                RetrievalHit(
                    premise_id=premise_id,
                    score=score,
                    premise_text=self.corpus_dict[premise_id],
                    raw={"llm_item": dict(item)},
                )
            )
            if len(hits) >= k:
                break

        # If the model did not return enough valid results, keep stable behavior.
        return hits

