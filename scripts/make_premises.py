import json
from pathlib import Path

src = Path("data/corpus.jsonl")
dst = Path("data/premises.jsonl")
dst.parent.mkdir(parents=True, exist_ok=True)

num_files = 0
num_premises = 0
num_written = 0

with src.open("r", encoding="utf-8") as f, dst.open("w", encoding="utf-8") as out:
    for line in f:
        line = line.strip()
        if not line:
            continue
        num_files += 1
        row = json.loads(line)

        for p in row.get("premises", []):
            # corpus.jsonl 里常见字段：full_name, code
            premise_id = p.get("full_name")
            text = p.get("code", "")
            num_premises += 1

            if not premise_id or not text:
                continue

            out.write(json.dumps({
                "premise_id": premise_id,
                "text": text
            }, ensure_ascii=True) + "\n")
            num_written += 1

print(f"files_seen={num_files}")
print(f"premises_seen={num_premises}")
print(f"premises_written={num_written}")
print(f"output={dst}")
