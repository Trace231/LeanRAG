from datasets import load_dataset

ds = load_dataset("cat-searcher/leandojo-benchmark-4-random")
print(ds)

split = "test" if "test" in ds else list(ds.keys())[0]
row = ds[split][0]
print("keys:", row.keys())

# 如果是 theorem-level 结构，可能要看 traced_tactics[0]
if "traced_tactics" in row and row["traced_tactics"]:
    t0 = row["traced_tactics"][0]
    print("tactic keys:", t0.keys())
