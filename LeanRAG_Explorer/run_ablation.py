from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from leanrag_explorer.evaluation.evaluator import _hit_at_k
from leanrag_explorer.filters.accessibility import AccessibilityFilter
from leanrag_explorer.query_builders.base import BaseQueryBuilder
from leanrag_explorer.query_builders.strategies import (
    DualTrackLLMWeightedQueryBuilder,
    MacroContextQueryBuilder,
    TemporalContextQueryBuilder,
    TwoStageDenoisedQueryBuilder,
)
from leanrag_explorer.retrievers.base import BaseRetriever
from leanrag_explorer.retrievers.hybrid import FusionMethod, HybridRetriever
from leanrag_explorer.types import EvalSample, QueryContext, RetrievalHit


def _tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in text.replace("\n", " ").split() if tok.strip()]


def _char_ngrams(text: str, n: int = 3) -> Counter[str]:
    s = text.lower()
    if len(s) < n:
        return Counter([s]) if s else Counter()
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def _cosine_counter(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    common = set(a.keys()) & set(b.keys())
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class SimpleDenseRetriever(BaseRetriever):
    """Dense-like retriever scaffold using character n-gram cosine similarity.

    This is not a neural encoder. It is a deterministic surrogate that keeps
    the ablation plumbing and provenance logic runnable before integrating T5.
    """

    def __init__(self, premise_texts: Dict[str, str], query_key: str = "formal_query"):
        self.query_key = query_key
        self.premise_texts = premise_texts
        self.index = {pid: _char_ngrams(text) for pid, text in premise_texts.items()}

    def retrieve(self, query_dict: Dict[str, str], k: int = 100) -> List[RetrievalHit]:
        query = query_dict.get(self.query_key, "")
        qv = _char_ngrams(query)
        scored: List[RetrievalHit] = []
        for pid, pv in self.index.items():
            score = _cosine_counter(qv, pv)
            scored.append(
                RetrievalHit(
                    premise_id=pid,
                    score=score,
                    dense_score=score,
                    premise_text=self.premise_texts[pid],
                )
            )
        scored.sort(key=lambda h: h.score, reverse=True)
        ranked = [
            RetrievalHit(
                premise_id=h.premise_id,
                score=h.score,
                dense_score=h.score,
                rank_in_dense=rank,
                premise_text=h.premise_text,
                raw=h.raw,
            )
            for rank, h in enumerate(scored, start=1)
        ]
        return ranked[:k]


class SimpleSparseRetriever(BaseRetriever):
    """Sparse lexical retriever scaffold with TF-IDF style scoring.

    Score(query, doc) = sum_{t in query} tf_doc(t) * idf(t)
    """

    def __init__(self, premise_texts: Dict[str, str], query_key: str = "formal_query"):
        self.query_key = query_key
        self.premise_texts = premise_texts
        self.doc_tokens: Dict[str, List[str]] = {
            pid: _tokenize(text) for pid, text in premise_texts.items()
        }
        self.idf = self._build_idf(self.doc_tokens.values())

    def _build_idf(self, all_docs: Iterable[Sequence[str]]) -> Dict[str, float]:
        docs = list(all_docs)
        n = len(docs)
        df: Dict[str, int] = defaultdict(int)
        for toks in docs:
            for t in set(toks):
                df[t] += 1
        # Smooth IDF to avoid division by zero and extreme values.
        return {t: math.log((n + 1) / (f + 1)) + 1.0 for t, f in df.items()}

    def retrieve(self, query_dict: Dict[str, str], k: int = 100) -> List[RetrievalHit]:
        query = query_dict.get(self.query_key, "")
        q_tokens = _tokenize(query)
        scored: List[RetrievalHit] = []

        for pid, doc in self.doc_tokens.items():
            tf = Counter(doc)
            score = 0.0
            for t in q_tokens:
                score += tf.get(t, 0) * self.idf.get(t, 0.0)
            scored.append(
                RetrievalHit(
                    premise_id=pid,
                    score=score,
                    sparse_score=score,
                    premise_text=self.premise_texts[pid],
                )
            )

        scored.sort(key=lambda h: h.score, reverse=True)
        ranked = [
            RetrievalHit(
                premise_id=h.premise_id,
                score=h.score,
                sparse_score=h.score,
                rank_in_sparse=rank,
                premise_text=h.premise_text,
                raw=h.raw,
            )
            for rank, h in enumerate(scored, start=1)
        ]
        return ranked[:k]


class SimpleLLMRetriever(SimpleSparseRetriever):
    """LLM-view retriever scaffold.

    For phase-3 plumbing we reuse sparse lexical scoring but route to `nl_query`.
    Replace this class with a real LLM retrieval service later.
    """

    def __init__(self, premise_texts: Dict[str, str]):
        super().__init__(premise_texts=premise_texts, query_key="nl_query")


def load_premises(path: Path) -> Dict[str, str]:
    """Load premise table from JSON/JSONL.

    Accepted keys per record:
      - premise_id / id / full_name
      - text / code / premise_text
    """

    premise_texts: Dict[str, str] = {}
    if path.suffix == ".jsonl":
        lines = path.read_text().splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
    else:
        payload = json.loads(path.read_text())
        records = payload if isinstance(payload, list) else payload.get("premises", [])

    for r in records:
        pid = str(r.get("premise_id") or r.get("id") or r.get("full_name"))
        text = str(r.get("text") or r.get("code") or r.get("premise_text") or "")
        if pid and text:
            premise_texts[pid] = text
    return premise_texts


def load_eval_dataset(path: Path) -> List[EvalSample]:
    """Load ablation dataset from JSON/JSONL.

    Required record fields:
      - theorem_statement
      - current_state
      - gold_premise_ids
    Optional:
      - recent_tactics, hypotheses, goal, metadata
    """

    if path.suffix == ".jsonl":
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    else:
        obj = json.loads(path.read_text())
        rows = obj if isinstance(obj, list) else obj.get("samples", [])

    out: List[EvalSample] = []
    for r in rows:
        ctx = QueryContext(
            theorem_statement=str(r.get("theorem_statement", "")),
            current_state=str(r.get("current_state", "")),
            recent_tactics=list(r.get("recent_tactics", [])),
            hypotheses=list(r.get("hypotheses", [])),
            goal=str(r.get("goal", "")),
            metadata=dict(r.get("metadata", {})),
        )
        out.append(EvalSample(context=ctx, gold_premise_ids=list(r.get("gold_premise_ids", []))))
    return out


def load_allowed_map(path: Path | None) -> Dict[str, set[str]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    return {str(k): set(v) for k, v in payload.items()}


def evaluate_combo(
    combo_name: str,
    query_builder: BaseQueryBuilder,
    retriever: BaseRetriever,
    dataset: List[EvalSample],
    accessibility_filter: AccessibilityFilter,
    top_k: int,
    error_k: int,
) -> tuple[Dict[str, float], List[Dict[str, object]]]:
    """Run one ablation combo and collect metrics + hard-error cases."""

    total = len(dataset)
    r1 = r10 = r50 = 0
    r1f = r10f = r50f = 0
    errors: List[Dict[str, object]] = []

    for idx, sample in enumerate(dataset):
        q = query_builder.build(sample.context)
        hits = retriever.retrieve(q, k=top_k)
        filtered_hits = accessibility_filter.filter_hits(sample.context, hits)
        gold = set(sample.gold_premise_ids)

        r1 += _hit_at_k(hits, gold, 1)
        r10 += _hit_at_k(hits, gold, 10)
        r50 += _hit_at_k(hits, gold, 50)
        r1f += _hit_at_k(filtered_hits, gold, 1)
        r10f += _hit_at_k(filtered_hits, gold, 10)
        r50f += _hit_at_k(filtered_hits, gold, 50)

        # Error analysis: collect misses at filtered @error_k with provenance.
        if _hit_at_k(filtered_hits, gold, error_k) == 0:
            errors.append(
                {
                    "combo": combo_name,
                    "sample_index": idx,
                    "goal": sample.context.goal,
                    "state": sample.context.current_state,
                    "gold_premise_ids": sample.gold_premise_ids,
                    "top_candidates": [
                        {
                            "premise_id": h.premise_id,
                            "score": h.score,
                            "dense_score": h.dense_score,
                            "sparse_score": h.sparse_score,
                            "rank_in_dense": h.rank_in_dense,
                            "rank_in_sparse": h.rank_in_sparse,
                        }
                        for h in filtered_hits[:error_k]
                    ],
                }
            )

    denom = max(total, 1)
    metrics = {
        "combo": combo_name,
        "num_samples": total,
        "recall_at_1": r1 / denom,
        "recall_at_10": r10 / denom,
        "recall_at_50": r50 / denom,
        "recall_at_1_filtered": r1f / denom,
        "recall_at_10_filtered": r10f / denom,
        "recall_at_50_filtered": r50f / denom,
        "num_errors_at_k_filtered": float(len(errors)),
    }
    return metrics, errors


def write_metrics_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LeanRAG ablation studies.")
    parser.add_argument("--dataset", required=True, help="Path to eval dataset json/jsonl.")
    parser.add_argument("--premises", required=True, help="Path to premises json/jsonl.")
    parser.add_argument("--allowed-map", default=None, help="Path to theorem_id->allowed premise ids JSON.")
    parser.add_argument("--output-dir", default="outputs/ablation", help="Output directory.")
    parser.add_argument("--top-k", type=int, default=100, help="Retriever top-k.")
    parser.add_argument("--error-k", type=int, default=10, help="Miss@k threshold for error logs.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Linear fusion weight for stream A.")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF smoothing constant.")
    args = parser.parse_args()

    dataset = load_eval_dataset(Path(args.dataset))
    premise_texts = load_premises(Path(args.premises))
    allowed_map = load_allowed_map(Path(args.allowed_map) if args.allowed_map else None)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    access_filter = AccessibilityFilter(allowed_map=allowed_map, fallback_allow_all=True)

    # Query builder strategies for ablation.
    builders: Dict[str, BaseQueryBuilder] = {
        "dual_track": DualTrackLLMWeightedQueryBuilder(),
        "macro_context": MacroContextQueryBuilder(),
        "temporal_context": TemporalContextQueryBuilder(max_tactics=8),
        "denoised_state": TwoStageDenoisedQueryBuilder(top_n=6),
    }

    # Retriever backends.
    dense = SimpleDenseRetriever(premise_texts, query_key="formal_query")
    sparse_formal = SimpleSparseRetriever(premise_texts, query_key="formal_query")
    llm_nl = SimpleLLMRetriever(premise_texts)

    retrievers: Dict[str, BaseRetriever] = {
        "dense_only": dense,
        "sparse_only": sparse_formal,
        "hybrid_linear_dense_sparse": HybridRetriever(
            retrievers=[dense, sparse_formal],
            fusion=FusionMethod.LINEAR,
            alpha=args.alpha,
            rrf_k=args.rrf_k,
        ),
        "hybrid_rrf_dense_sparse": HybridRetriever(
            retrievers=[dense, sparse_formal],
            fusion=FusionMethod.RRF,
            alpha=args.alpha,
            rrf_k=args.rrf_k,
        ),
        "hybrid_rrf_dense_llm": HybridRetriever(
            retrievers=[dense, llm_nl],
            fusion=FusionMethod.RRF,
            alpha=args.alpha,
            rrf_k=args.rrf_k,
        ),
    }

    metric_rows: List[Dict[str, float]] = []
    all_errors: List[Dict[str, object]] = []

    for b_name, builder in builders.items():
        for r_name, retriever in retrievers.items():
            combo = f"{b_name}__{r_name}"
            metrics, errors = evaluate_combo(
                combo_name=combo,
                query_builder=builder,
                retriever=retriever,
                dataset=dataset,
                accessibility_filter=access_filter,
                top_k=args.top_k,
                error_k=args.error_k,
            )
            metric_rows.append(metrics)
            all_errors.extend(errors)

    metric_rows.sort(key=lambda x: x["recall_at_10_filtered"], reverse=True)
    write_metrics_csv(out_dir / "metrics.csv", metric_rows)
    write_jsonl(out_dir / "error_cases.jsonl", all_errors)

    summary = {
        "num_samples": len(dataset),
        "num_premises": len(premise_texts),
        "num_combos": len(metric_rows),
        "top_combo_by_recall10_filtered": metric_rows[0]["combo"] if metric_rows else "",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

