from datasets import load_dataset
import json
from pathlib import Path

# 你要的数据集
ds = load_dataset("cat-searcher/leandojo-benchmark-4-random")

# 优先 test，没有就取第一个 split
split = "test" if "test" in ds else list(ds.keys())[0]
rows = [dict(x) for x in ds[split]]

out = Path("data/leandojo_test.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

print("saved:", out)
print("split:", split, "rows:", len(rows))

