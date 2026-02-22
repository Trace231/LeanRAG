from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Protocol

from openai import OpenAI

from leanrag_explorer.query_builders.base import BaseQueryBuilder
from leanrag_explorer.types import QueryContext


class SummaryGenerator(Protocol):
    """Protocol for LLM-style summarizers used by dual-track query builders."""

    def summarize(self, context: QueryContext) -> str:
        ...


@dataclass
class RuleBasedSummaryGenerator:
    """Safe default summarizer for early-stage experiments.

    This avoids OOD risk from injecting arbitrary NL while still providing a
    compact temporal summary. Later, replace this class with an LLM backend.
    """

    max_tactics: int = 5

    def summarize(self, context: QueryContext) -> str:
        tail = context.recent_tactics[-self.max_tactics :]
        if not tail:
            return "No prior tactics. Focus on current goal."
        return "Recent proof trajectory: " + " -> ".join(tail)


@dataclass
class OpenAISummaryGenerator:
    """LLM-based summarizer for dual-track query construction.

    This class generates an `nl_query` from recent tactics and current state.
    The summarization prompt is intentionally constrained to keep the output
    concise and retrieval-oriented rather than verbose.
    """

    model: str = "gpt-4o-mini"
    max_tactics: int = 8
    max_output_tokens: int = 120
    temperature: float = 0.1

    def __post_init__(self) -> None:
        self.client = OpenAI()

    def summarize(self, context: QueryContext) -> str:
        recent_tactics = context.recent_tactics[-self.max_tactics :]
        history = "\n".join(f"- {t}" for t in recent_tactics) if recent_tactics else "- (none)"
        prompt = (
            "You are helping retrieval for Lean theorem proving.\n"
            "Write ONE compact natural-language retrieval query that captures:\n"
            "1) what the proof is trying to do now,\n"
            "2) key algebraic/logic objects from the goal/hypotheses,\n"
            "3) useful lemma intent (e.g., rewriting, monotonicity, divisibility).\n"
            "Avoid extra narration. Output plain text only.\n\n"
            f"Current state:\n{context.current_state.strip()}\n\n"
            f"Recent tactics:\n{history}\n"
        )
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=self.max_output_tokens,
                temperature=self.temperature,
            )
            summary = (resp.output_text or "").strip()
            if summary:
                return summary
        except Exception:
            # Fall back to deterministic summary to keep the pipeline resilient.
            pass
        return RuleBasedSummaryGenerator(max_tactics=self.max_tactics).summarize(context)


class DualTrackLLMWeightedQueryBuilder(BaseQueryBuilder):
    """Strategy Base: formal track + NL track for late fusion."""

    def __init__(self, summary_generator: SummaryGenerator | None = None) -> None:
        self.summary_generator = summary_generator or RuleBasedSummaryGenerator()

    def build(self, context: QueryContext) -> Dict[str, str]:
        return {
            "formal_query": context.current_state.strip(),
            "nl_query": self.summary_generator.summarize(context).strip(),
        }


class MacroContextQueryBuilder(BaseQueryBuilder):
    """Strategy D: theorem statement + current proof state."""

    def build(self, context: QueryContext) -> Dict[str, str]:
        formal = f"{context.theorem_statement.strip()}\n\n{context.current_state.strip()}"
        return {"formal_query": formal, "nl_query": ""}


class TemporalContextQueryBuilder(BaseQueryBuilder):
    """Strategy B: recent tactic history + current state."""

    def __init__(self, max_tactics: int = 10) -> None:
        self.max_tactics = max_tactics

    def build(self, context: QueryContext) -> Dict[str, str]:
        history = "\n".join(context.recent_tactics[-self.max_tactics :]).strip()
        formal = (
            f"Recent tactics:\n{history}\n\nCurrent state:\n{context.current_state.strip()}"
            if history
            else context.current_state.strip()
        )
        return {"formal_query": formal, "nl_query": ""}


class KeywordFilterHelper:
    """Strategy A helper: lightweight term matching for local hypothesis filtering.

    This component is intentionally simple and deterministic so it can be used
    as a transparent preprocessing stage before dense retrieval.
    """

    _TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_'.]*")

    def _tokens(self, text: str) -> List[str]:
        return [t.lower() for t in self._TOKEN.findall(text)]

    def _goal_keywords(self, goal: str) -> set[str]:
        # Remove very short tokens to reduce noise from single-character vars.
        return {t for t in self._tokens(goal) if len(t) > 2}

    def rank_hypotheses(self, goal: str, hypotheses: Iterable[str]) -> List[str]:
        keywords = self._goal_keywords(goal)
        scored: List[tuple[int, str]] = []
        for h in hypotheses:
            h_tokens = self._tokens(h)
            overlap = sum(1 for t in h_tokens if t in keywords)
            scored.append((overlap, h))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [h for s, h in scored if s > 0]


class TwoStageDenoisedQueryBuilder(BaseQueryBuilder):
    """Strategy C: Goal + top-N relevant hypotheses (denoised state).

    Two stages:
      1) Local lightweight relevance scoring over hypotheses.
      2) Build compact formal query for global dense retrieval.
    """

    def __init__(self, helper: KeywordFilterHelper | None = None, top_n: int = 6) -> None:
        self.helper = helper or KeywordFilterHelper()
        self.top_n = top_n

    def build(self, context: QueryContext) -> Dict[str, str]:
        ranked = self.helper.rank_hypotheses(context.goal, context.hypotheses)
        selected = ranked[: self.top_n]
        compact_hyps = "\n".join(selected).strip()

        formal_query = f"Goal:\n{context.goal.strip()}"
        if compact_hyps:
            formal_query += f"\n\nRelevant hypotheses:\n{compact_hyps}"

        return {"formal_query": formal_query, "nl_query": ""}

