import json
import argparse
from pathlib import Path

def split_state(state: str):
    if "⊢" in state:
        left, right = state.split("⊢", 1)
        hyps = [x.strip() for x in left.splitlines() if x.strip()]
        return hyps, right.strip()
    return [], state.strip()

def extract_gold_from_annot(annot):
    """Robust extraction from multiple annotated_tactic formats."""
    provs = []

    # format A: [annotated_text, [prov...]]
    if isinstance(annot, list):
        if len(annot) >= 2 and isinstance(annot[1], list):
            provs = annot[1]
        else:
            # maybe directly list of prov dicts
            if annot and isinstance(annot[0], dict):
                provs = annot

    # format B: {"provenances":[...]} or {"refs":[...]}
    elif isinstance(annot, dict):
        if isinstance(annot.get("provenances"), list):
            provs = annot["provenances"]
        elif isinstance(annot.get("refs"), list):
            provs = annot["refs"]

    gold = sorted({
        p.get("full_name", "").strip()
        for p in provs
        if isinstance(p, dict) and p.get("full_name")
    })
    return gold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--history-len", type=int, default=3)
    args = ap.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out.open("w", encoding="utf-8") as f:
        for thm in rows:
            theorem_statement = thm.get("theorem_statement", "")
            file_path = thm.get("file_path", "")
            full_name = thm.get("full_name", "")

            # Case 1: theorem-level
            if isinstance(thm.get("traced_tactics"), list):
                traced_tactics = thm["traced_tactics"]
                tactic_texts = [t.get("tactic", "") for t in traced_tactics]

                for i, t in enumerate(traced_tactics):
                    state = t.get("state_before", "")
                    if not state:
                        continue
                    gold = extract_gold_from_annot(t.get("annotated_tactic"))
                    if not gold:
                        continue

                    hyps, goal = split_state(state)
                    recent = [x for x in tactic_texts[max(0, i-args.history_len):i] if x]

                    sample = {
                        "theorem_statement": theorem_statement,
                        "current_state": state,
                        "recent_tactics": recent,
                        "hypotheses": hyps,
                        "goal": goal,
                        "metadata": {
                            "theorem_id": f"{file_path}::{full_name}::{i}",
                            "file_path": file_path,
                            "full_name": full_name,
                            "tactic_idx": i,
                        },
                        "gold_premise_ids": gold,
                    }
                    f.write(json.dumps(sample, ensure_ascii=True) + "\n")
                    written += 1
                    if written >= args.max_samples:
                        break

            # Case 2: step-level (one row = one tactic step)
            else:
                state = thm.get("state_before", "") or thm.get("current_state", "")
                if not state:
                    continue
                gold = extract_gold_from_annot(thm.get("annotated_tactic"))
                if not gold:
                    continue

                hyps, goal = split_state(state)
                recent = thm.get("recent_tactics", [])
                if not isinstance(recent, list):
                    recent = []

                sample = {
                    "theorem_statement": theorem_statement,
                    "current_state": state,
                    "recent_tactics": recent[-args.history_len:],
                    "hypotheses": hyps,
                    "goal": goal,
                    "metadata": {
                        "theorem_id": str(thm.get("id", f"{file_path}::{full_name}")),
                        "file_path": file_path,
                        "full_name": full_name,
                        "tactic_idx": thm.get("tactic_idx", -1),
                    },
                    "gold_premise_ids": gold,
                }
                f.write(json.dumps(sample, ensure_ascii=True) + "\n")
                written += 1

            if written >= args.max_samples:
                break

    print(f"written={written}")
    print(f"output={out}")

if __name__ == "__main__":
    main()
