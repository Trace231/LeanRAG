from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from pantograph import Server

from lean_dojo_v2.agent.lean_agent import LeanAgent
from lean_dojo_v2.lean_agent.common import Corpus
from lean_dojo_v2.lean_dojo.data_extraction.lean import LeanGitRepo, Pos
from lean_dojo_v2.lean_dojo.data_extraction.trace import get_traced_repo_path
from lean_dojo_v2.prover.retrieval_prover import RetrievalProver
from lean_dojo_v2.utils.constants import RAID_DIR


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
    ):
        super().__init__(ret_ckpt_path, gen_ckpt_path, indexed_corpus_path)
        self.variant = variant
        self._history_by_theorem: Dict[str, List[str]] = {}
        self._ctx_by_theorem: Dict[str, Dict[str, object]] = {}

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

        def patched_retrieve(
            state: List[str],
            file_name: List[str],
            theorem_full_name: List[str],
            theorem_pos: List[Pos],
            k: int,
        ):
            transformed: List[str] = []
            resolved_file_names: List[str] = []
            for s, thm_name in zip(state, theorem_full_name):
                ctx = self._ctx_by_theorem.get(thm_name, {})
                theorem_statement = str(ctx.get("theorem_statement", ""))
                recent_tactics = list(ctx.get("recent_tactics", []))
                transformed.append(
                    build_query_variant(
                        self.variant,
                        s,
                        theorem_statement=theorem_statement,
                        recent_tactics=recent_tactics,
                    )
                )
            for p in file_name:
                resolved_file_names.append(resolve_file_path(p))
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
        self._ctx_by_theorem[theorem_name] = {
            "theorem_statement": getattr(self.theorem, "theorem_statement", "") or "",
            "recent_tactics": self._history_by_theorem[theorem_name],
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
) -> List[Dict[str, object]]:
    prover = AblationRetrievalProver(
        ret_ckpt_path=ret_ckpt_path,
        gen_ckpt_path=gen_ckpt_path,
        indexed_corpus_path=corpus_jsonl_path,
        variant=variant,
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
        default="raw_state,goal_only,macro_context,temporal_context,denoised_state",
        help="Comma-separated query variants.",
    )
    parser.add_argument("--ret-ckpt-path", default="")
    parser.add_argument("--gen-ckpt-path", default=f"{RAID_DIR}/model_lightning.ckpt")
    parser.add_argument("--corpus-jsonl-path", default="")
    parser.add_argument("--seed", type=int, default=3407)
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

