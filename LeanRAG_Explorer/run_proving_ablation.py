from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from pantograph import Server

from lean_dojo_v2.agent.lean_agent import LeanAgent
from lean_dojo_v2.lean_agent.common import Corpus
from lean_dojo_v2.lean_dojo.data_extraction.lean import LeanGitRepo, Pos
from lean_dojo_v2.lean_dojo.data_extraction.trace import get_traced_repo_path
from lean_dojo_v2.prover.retrieval_prover import RetrievalProver
from lean_dojo_v2.utils.constants import RAID_DIR

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore[assignment]


def _split_state(state: str) -> Tuple[List[str], str]:
    if "⊢" in state:
        lhs, rhs = state.split("⊢", 1)
        hyps = [x.strip() for x in lhs.splitlines() if x.strip()]
        goal = rhs.strip()
        return hyps, goal
    return [], state.strip()


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_'.]*")


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def _goal_keyword_filter(goal: str, hypotheses: List[str], top_n: int = 6) -> List[str]:
    goal_kw = {t for t in _tokens(goal) if len(t) > 2}
    scored: List[Tuple[int, str]] = []
    for h in hypotheses:
        score = sum(1 for t in _tokens(h) if t in goal_kw)
        if score > 0:
            scored.append((score, h))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in scored[:top_n]]


def build_query_variant(
    variant: str,
    state: str,
    theorem_statement: str,
    recent_tactics: List[str],
    recent_states: Optional[List[str]] = None,
) -> str:
    """Build transformed retrieval query from original proof state.

    Important:
      - We ONLY transform the query sent to retriever.
      - We do NOT change the proving engine, search budget, or theorem set.
    """

    hyps, goal = _split_state(state)

    if variant == "raw_state":
        return state

    if variant == "goal_only":
        return goal if goal else state

    if variant == "macro_context":
        ts = theorem_statement.strip()
        return f"{ts}\n\n{state}" if ts else state

    if variant == "temporal_context":
        tail = recent_tactics[-8:]
        if not tail:
            return state
        history = "\n".join(f"- {t}" for t in tail)
        return f"Recent tactics:\n{history}\n\nCurrent state:\n{state}"

    if variant == "recent_states_context":
        tail_states = (recent_states or [])[-3:]
        if not tail_states:
            return state
        history = "\n\n".join(
            f"State -{len(tail_states) - i}:\n{s}" for i, s in enumerate(tail_states)
        )
        return f"Recent states:\n{history}\n\nCurrent state:\n{state}"

    if variant == "denoised_state":
        selected = _goal_keyword_filter(goal, hyps, top_n=6)
        if selected:
            return f"Goal:\n{goal}\n\nRelevant hypotheses:\n" + "\n".join(selected)
        return f"Goal:\n{goal}" if goal else state

    raise ValueError(f"Unknown variant: {variant}")


class AblationRetrievalProver(RetrievalProver):
    """RetrievalProver subclass with query-stage-only ablation hooks."""

    def __init__(
        self,
        ret_ckpt_path: Optional[str],
        gen_ckpt_path: str,
        indexed_corpus_path: str,
        variant: str,
        summary_alpha: float = 0.7,
        summary_model: str = "gpt-4o-mini",
        summary_max_output_tokens: int = 120,
        summary_temperature: float = 0.1,
    ):
        super().__init__(ret_ckpt_path, gen_ckpt_path, indexed_corpus_path)
        self.variant = variant
        self.summary_alpha = float(summary_alpha)
        self.summary_model = summary_model
        self.summary_max_output_tokens = int(summary_max_output_tokens)
        self.summary_temperature = float(summary_temperature)
        self._history_by_theorem: Dict[str, List[str]] = {}
        self._state_history_by_theorem: Dict[str, List[str]] = {}
        self._ctx_by_theorem: Dict[str, Dict[str, object]] = {}
        self._summary_cache: Dict[str, str] = {}
        self._summary_client = None
        if OpenAI is not None and os.getenv("OPENAI_API_KEY"):
            try:
                self._summary_client = OpenAI()
            except Exception:
                self._summary_client = None

        original = self.tactic_generator.retriever
        if original is None:
            logger.info(
                "Retriever is disabled (generator-only mode). "
                "Query transformation patch is skipped."
            )
        else:
            self._install_query_transform_patch(original)

    def _install_query_transform_patch(self, retriever_module: object) -> None:
        """Patch retriever.retrieve without replacing the nn.Module instance.

        Why:
          `tactic_generator` is a torch.nn.Module and its child attribute
          `retriever` must remain an nn.Module (or None). Replacing it with
          a plain Python object raises:
            TypeError: ... child module 'retriever' (torch.nn.Module or None expected)
        """

        original_retrieve = retriever_module.retrieve
        corpus_nodes: List[str] = []
        try:
            corpus_nodes = list(retriever_module.corpus.transitive_dep_graph.nodes)
        except Exception:
            corpus_nodes = []

        def resolve_file_path(path: str) -> str:
            """Map theorem file path to a corpus node path when possible."""
            if not corpus_nodes:
                return path
            if path in corpus_nodes:
                return path

            normalized = path[2:] if path.startswith("./") else path
            if normalized in corpus_nodes:
                return normalized

            # Suffix match first.
            suffix_matches = [p for p in corpus_nodes if p.endswith(normalized)]
            if len(suffix_matches) == 1:
                return suffix_matches[0]

            # Extension normalization.
            alt = (
                normalized[:-5] if normalized.endswith(".lean") else normalized + ".lean"
            )
            suffix_matches = [p for p in corpus_nodes if p.endswith(alt)]
            if len(suffix_matches) == 1:
                return suffix_matches[0]

            # Basename unique match as last resort.
            basename = normalized.split("/")[-1]
            base_matches = [p for p in corpus_nodes if p.split("/")[-1] == basename]
            if len(base_matches) == 1:
                return base_matches[0]

            return path

        def premise_key(premise: Any) -> Tuple[str, str, Tuple[int, int]]:
            path = str(getattr(premise, "path", ""))
            full_name = str(getattr(premise, "full_name", ""))
            start = getattr(premise, "start", None)
            if start is None:
                return path, full_name, (0, 0)
            return path, full_name, tuple(start)

        def normalize(scores: List[float]) -> List[float]:
            if not scores:
                return []
            low = min(scores)
            high = max(scores)
            if high == low:
                return [1.0 for _ in scores]
            return [(s - low) / (high - low) for s in scores]

        def build_summary_query(state_text: str, thm_name: str) -> str:
            ctx = self._ctx_by_theorem.get(thm_name, {})
            recent_states = list(ctx.get("recent_states", []))[-3:]
            recent_tactics = list(ctx.get("recent_tactics", []))[-8:]
            cache_key = "||".join(
                [
                    thm_name,
                    state_text,
                    "\n".join(recent_states),
                    "\n".join(recent_tactics),
                ]
            )
            cached = self._summary_cache.get(cache_key)
            if cached is not None:
                return cached

            fallback_summary = (
                "Proof trajectory summary: "
                + (" | ".join(recent_tactics) if recent_tactics else "No recent tactics")
                + f"\nCurrent objective:\n{state_text}"
            )
            if self._summary_client is None:
                self._summary_cache[cache_key] = fallback_summary
                return fallback_summary

            prompt = (
                "You are helping retrieval for Lean theorem proving.\n"
                "Write one concise retrieval query describing what the proof is doing now.\n"
                "Focus on goal intent, key objects, and useful lemma direction.\n"
                "Output plain text only.\n\n"
                f"Current state:\n{state_text}\n\n"
                f"Recent states:\n{chr(10).join(recent_states) if recent_states else '(none)'}\n\n"
                f"Recent tactics:\n{chr(10).join(recent_tactics) if recent_tactics else '(none)'}\n"
            )
            try:
                resp = self._summary_client.responses.create(
                    model=self.summary_model,
                    input=prompt,
                    max_output_tokens=self.summary_max_output_tokens,
                    temperature=self.summary_temperature,
                )
                summary = (resp.output_text or "").strip()
                if summary:
                    self._summary_cache[cache_key] = summary
                    return summary
            except Exception:
                pass

            self._summary_cache[cache_key] = fallback_summary
            return fallback_summary

        def patched_retrieve(
            state: List[str],
            file_name: List[str],
            theorem_full_name: List[str],
            theorem_pos: List[Pos],
            k: int,
        ):
            resolved_file_names: List[str] = []
            for p in file_name:
                resolved_file_names.append(resolve_file_path(p))

            if self.variant == "dual_summary_fusion":
                state_results = original_retrieve(
                    state,
                    resolved_file_names,
                    theorem_full_name,
                    theorem_pos,
                    k,
                )
                summary_queries = [
                    build_summary_query(s, thm_name)
                    for s, thm_name in zip(state, theorem_full_name)
                ]
                summary_results = original_retrieve(
                    summary_queries,
                    resolved_file_names,
                    theorem_full_name,
                    theorem_pos,
                    k,
                )
                state_premises_batch, state_scores_batch = state_results
                sum_premises_batch, sum_scores_batch = summary_results
                fused_premises_batch = []
                fused_scores_batch = []
                for st_premises, st_scores, su_premises, su_scores in zip(
                    state_premises_batch,
                    state_scores_batch,
                    sum_premises_batch,
                    sum_scores_batch,
                ):
                    st_norm = normalize([float(x) for x in st_scores])
                    su_norm = normalize([float(x) for x in su_scores])
                    fused_by_key: Dict[Tuple[str, str, Tuple[int, int]], Dict[str, Any]] = {}
                    for premise, score in zip(st_premises, st_norm):
                        fused_by_key[premise_key(premise)] = {
                            "premise": premise,
                            "state": score,
                            "summary": 0.0,
                        }
                    for premise, score in zip(su_premises, su_norm):
                        key = premise_key(premise)
                        if key not in fused_by_key:
                            fused_by_key[key] = {
                                "premise": premise,
                                "state": 0.0,
                                "summary": score,
                            }
                        else:
                            fused_by_key[key]["summary"] = score
                    fused_items = []
                    alpha = min(max(self.summary_alpha, 0.0), 1.0)
                    for payload in fused_by_key.values():
                        fused_score = alpha * payload["state"] + (1.0 - alpha) * payload["summary"]
                        fused_items.append((payload["premise"], float(fused_score)))
                    fused_items.sort(key=lambda x: x[1], reverse=True)
                    fused_items = fused_items[:k]
                    fused_premises_batch.append([p for p, _ in fused_items])
                    fused_scores_batch.append([s for _, s in fused_items])
                return fused_premises_batch, fused_scores_batch

            transformed: List[str] = []
            for s, thm_name in zip(state, theorem_full_name):
                ctx = self._ctx_by_theorem.get(thm_name, {})
                theorem_statement = str(ctx.get("theorem_statement", ""))
                recent_tactics = list(ctx.get("recent_tactics", []))
                recent_states = list(ctx.get("recent_states", []))
                transformed.append(
                    build_query_variant(
                        self.variant,
                        s,
                        theorem_statement=theorem_statement,
                        recent_tactics=recent_tactics,
                        recent_states=recent_states,
                    )
                )
            return original_retrieve(
                transformed,
                resolved_file_names,
                theorem_full_name,
                theorem_pos,
                k,
            )

        # Keep module type unchanged; only swap behavior of retrieve().
        retriever_module.retrieve = patched_retrieve

    def next_tactic(self, state, goal_id):
        if not hasattr(self, "theorem") or self.theorem is None:
            return None

        theorem_name = self.theorem.full_name
        if theorem_name not in self._history_by_theorem:
            self._history_by_theorem[theorem_name] = []
        if theorem_name not in self._state_history_by_theorem:
            self._state_history_by_theorem[theorem_name] = []
        self._ctx_by_theorem[theorem_name] = {
            "theorem_statement": getattr(self.theorem, "theorem_statement", "") or "",
            "recent_tactics": self._history_by_theorem[theorem_name],
            "recent_states": self._state_history_by_theorem[theorem_name],
        }

        suggestions = self.tactic_generator.generate(
            state=str(state),
            file_path=str(self.theorem.file_path),
            theorem_full_name=self.theorem.full_name,
            theorem_pos=self.theorem.start,
            num_samples=10,
        )
        if not suggestions:
            return None
        tactics, log_probs = zip(*suggestions)
        probs = np.exp(log_probs) / np.sum(np.exp(log_probs))
        selected = random.choices(tactics, weights=probs, k=1)[0]
        self._history_by_theorem[theorem_name].append(str(selected))
        cur_state = str(state)
        recent_states = self._state_history_by_theorem[theorem_name]
        if not recent_states or recent_states[-1] != cur_state:
            recent_states.append(cur_state)
        return selected


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def run_variant(
    variant: str,
    theorem_repo_pairs,
    ret_ckpt_path: Optional[str],
    gen_ckpt_path: str,
    corpus_jsonl_path: str,
    max_theorems: int,
    build_deps: bool,
    summary_alpha: float,
    summary_model: str,
    summary_max_output_tokens: int,
    summary_temperature: float,
) -> List[Dict[str, object]]:
    prover = AblationRetrievalProver(
        ret_ckpt_path=ret_ckpt_path,
        gen_ckpt_path=gen_ckpt_path,
        indexed_corpus_path=corpus_jsonl_path,
        variant=variant,
        summary_alpha=summary_alpha,
        summary_model=summary_model,
        summary_max_output_tokens=summary_max_output_tokens,
        summary_temperature=summary_temperature,
    )

    rows: List[Dict[str, object]] = []
    for idx, (theorem, repo) in enumerate(theorem_repo_pairs[:max_theorems]):
        traced_repo_path = get_traced_repo_path(repo, build_deps=build_deps)
        server = Server(
            imports=["Init", str(theorem.file_path).replace(".lean", "")],
            project_path=traced_repo_path,
        )
        result, used_tactics = prover.search(server=server, theorem=theorem, verbose=False)
        rows.append(
            {
                "variant": variant,
                "index": idx,
                "theorem_full_name": theorem.full_name,
                "file_path": str(theorem.file_path),
                "success": bool(result.success),
                "steps": int(result.steps),
                "duration": _safe_float(result.duration),
                "num_used_tactics": len(used_tactics) if used_tactics is not None else 0,
            }
        )
    return rows


def _run_corpus_consistency_precheck(
    theorem_repo_pairs, corpus_jsonl_path: str
) -> None:
    """Fail fast if theorem file paths cannot be resolved in the retrieval corpus."""
    sample_theorem, _ = random.choice(theorem_repo_pairs)
    sample_path = str(sample_theorem.file_path)
    corpus = Corpus(corpus_jsonl_path)
    _ = corpus._get_file(sample_path)
    if corpus._resolve_path(sample_path) is None:
        raise RuntimeError(
            "Retrieval precheck failed: current corpus.jsonl does not include the target "
            f"file path '{sample_path}'. Please regenerate corpus data for the current "
            "repository/commit or fix path mapping."
        )
    logger.info("Retrieval precheck passed for theorem file path: {}", sample_path)


def summarize_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {
            "num_theorems": 0,
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "avg_duration_sec": 0.0,
        }
    success_vals = [1.0 if r["success"] else 0.0 for r in rows]
    return {
        "num_theorems": len(rows),
        "success_rate": mean(success_vals),
        "avg_steps": mean(float(r["steps"]) for r in rows),
        "avg_duration_sec": mean(float(r["duration"]) for r in rows),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end proving ablation by query-stage retrieval variants."
    )
    parser.add_argument("--url", required=True, help="GitHub repository URL.")
    parser.add_argument("--commit", required=True, help="Commit hash.")
    parser.add_argument("--database-path", default="dynamic_database.json")
    parser.add_argument("--output-dir", default="outputs/proving_ablation")
    parser.add_argument("--max-theorems", type=int, default=50)
    parser.add_argument(
        "--variants",
        default="raw_state,goal_only,macro_context,temporal_context,recent_states_context,denoised_state,dual_summary_fusion",
        help="Comma-separated query variants.",
    )
    parser.add_argument("--ret-ckpt-path", default="")
    parser.add_argument("--gen-ckpt-path", default=f"{RAID_DIR}/model_lightning.ckpt")
    parser.add_argument("--corpus-jsonl-path", default="")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--summary-alpha",
        type=float,
        default=0.7,
        help="Fusion weight for state retrieval in dual_summary_fusion.",
    )
    parser.add_argument(
        "--summary-model",
        default="gpt-4o-mini",
        help="LLM model for summary retrieval query generation.",
    )
    parser.add_argument(
        "--summary-max-output-tokens",
        type=int,
        default=120,
        help="Max output tokens for LLM summary generation.",
    )
    parser.add_argument(
        "--summary-temperature",
        type=float,
        default=0.1,
        help="Temperature for LLM summary generation.",
    )
    parser.add_argument(
        "--build-deps",
        action="store_true",
        help="Enable full dependency tracing. Default is False (noDeps).",
    )
    parser.add_argument(
        "--reuse-db",
        action="store_true",
        help="Reuse repository from dynamic_database.json if available before tracing.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    agent = LeanAgent(database_path=args.database_path)

    # Avoid setup_github_repository() because LeanAgent hardcodes build_deps=True.
    # We keep build_deps as an explicit CLI knob for stable experiments.
    traced_repo = None
    if args.reuse_db:
        traced_repo = agent.database.get_repository(args.url, args.commit)
        if traced_repo is not None:
            logger.info(
                "Reused repository from dynamic database: {} @ {}",
                args.url,
                args.commit,
            )
    if traced_repo is None:
        traced_repo = agent.trace_repository(
            url=args.url, commit=args.commit, build_deps=args.build_deps
        )
    agent.add_repository(traced_repo)

    # Same theorem pool for all variants.
    theorem_repo_pairs = []
    for repo in agent.repos:
        repository = agent.database.get_repository(repo.url, repo.commit)
        for theorem in repository.sorry_theorems_unproved:
            theorem_repo_pairs.append((theorem, repo))
    if not theorem_repo_pairs:
        raise RuntimeError("No sorry theorems found in repository.")

    retrieval_enabled = bool(args.ret_ckpt_path and args.ret_ckpt_path.strip())
    ret_ckpt_path = args.ret_ckpt_path.strip() if retrieval_enabled else ""
    if retrieval_enabled:
        logger.info("Retrieval checkpoint provided. Performing consistency precheck.")
    else:
        logger.info(
            "No retrieval checkpoint provided. Skipping consistency check and running in generator-only mode."
        )

    corpus_jsonl_path = ""
    if retrieval_enabled:
        corpus_jsonl_path = args.corpus_jsonl_path or str(agent.data_path / "corpus.jsonl")
        _run_corpus_consistency_precheck(theorem_repo_pairs, corpus_jsonl_path)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    all_rows: List[Dict[str, object]] = []
    summary: Dict[str, object] = {
        "url": args.url,
        "commit": args.commit,
        "build_deps": args.build_deps,
        "max_theorems": args.max_theorems,
        "variants": variants,
        "results": {},
    }

    for variant in variants:
        rows = run_variant(
            variant=variant,
            theorem_repo_pairs=theorem_repo_pairs,
            ret_ckpt_path=ret_ckpt_path if retrieval_enabled else None,
            gen_ckpt_path=args.gen_ckpt_path,
            corpus_jsonl_path=corpus_jsonl_path,
            max_theorems=args.max_theorems,
            build_deps=args.build_deps,
            summary_alpha=args.summary_alpha,
            summary_model=args.summary_model,
            summary_max_output_tokens=args.summary_max_output_tokens,
            summary_temperature=args.summary_temperature,
        )
        all_rows.extend(rows)
        summary["results"][variant] = summarize_rows(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "proving_results.csv", all_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

