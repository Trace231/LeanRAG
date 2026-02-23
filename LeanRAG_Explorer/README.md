# LeanRAG_Explorer

`LeanRAG_Explorer` 是一个围绕 LeanDojo-v2 / ReProver 检索增强证明（RAG）的研究工程仓库。  
它的目标不是“再写一个 proving 框架”，而是在 **不重写 LeanDojo 主流程** 的前提下，系统评估并改造“Query -> Retrieval -> Filter -> Proving”这条链路。

---

## 1. 背景与问题定义

在原始 ReProver 风格流程中，检索查询通常直接使用当前 proof state 文本。这个设计简单有效，但存在三个实际问题：

- **问题 A：Query 表示单一**  
  只看当前 state，忽略 theorem 宏观上下文、近期 tactic 轨迹、冗余假设噪声。
- **问题 B：Dense-only 检索偏置**  
  Dense 模型擅长语义，但对 Lean 中大量符号级精确匹配并不总是最优。
- **问题 C：工程评测断层**  
  很多工作只停在 Recall@K（且 GT 不唯一），但真正目标是最终 proving 成功率。

本仓库专门解决以上三点：先构建可复现实验框架，再在同一 proving 预算下做 query 检索消融。

---

## 2. 研究目标

本项目的核心研究问题是：

1. 不同 Query 结构是否能显著改变检索质量与 proving 成功率？
2. 在相同 theorem 集合和搜索预算下，哪种 Query 改造最稳定？
3. 在 noDeps 与 full deps 两种可访问前提空间下，结论是否一致？

---

## 3. 当前改造内容（你可以直接跑）

### 3.1 Query 侧改造（5 个变体）

本仓库目前实现并可消融的 Query 变体：

- `raw_state`：原始 proof state（基线）
- `goal_only`：仅使用 goal
- `macro_context`：theorem statement + current state
- `temporal_context`：recent tactics + current state
- `denoised_state`：goal + 关键词筛出的相关 hypotheses

这些变体都遵循同一原则：**只改检索查询，不改 prover 搜索器本体**。

### 3.2 检索与融合工程

仓库支持：

- Retriever 统一抽象（Dense / Sparse / LLM / Hybrid）
- Late fusion：Linear weighting / RRF
- 结果 provenance：记录 dense/sparse 分数与 rank，用于错误分析

### 3.3 评测工程

- `run_ablation.py`：检索层消融（Recall@K + 过滤前后对比）
- `run_proving_ablation.py`：端到端 proving 消融（success rate / steps / duration）

---

## 4. 与 LeanDojo-v2 的关系

本仓库刻意保持“轻侵入”：

- tracing、database、trainer、prover 主流程仍由 LeanDojo-v2 提供
- 我们主要在 query/retrieval 接口处插入策略层
- 消融时尽量保证 theorem 集合、搜索预算、模型权重不变

换句话说，这是一个 **LeanDojo-compatible 的研究层**，不是替代 LeanDojo。

---

## 5. 目前遇到的工程问题（真实现状）

在服务器环境中，`build_deps=True` 容易碰到 Lean toolchain / cache / dependency tracing 兼容问题，例如：

- `Init.olean incompatible header`
- `lake env lean --run ExtractData.lean` 在依赖 tracing 阶段失败

这类问题是生态工程问题，不是 query 策略问题。  
因此当前主线采用：

- **主分支默认 noDeps（稳定可复现）**
- **副分支再尝试 full deps（作为扩展实验）**

这个策略不会破坏主目标：在相同 setting 下比较 query 变体的相对效果。

---

## 6. 快速开始（主线：noDeps）

### 6.1 环境

```bash
conda create -n leanrag python=3.11 -y
conda activate leanrag
pip install -e ".[dev]"
pip install git+https://github.com/stanford-centaur/PyPantograph.git
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

设置 token：

```bash
export GITHUB_ACCESS_TOKEN="<your_token>"
export HF_TOKEN="<your_hf_token>"
```

### 6.2 生成 retriever 资产（默认 noDeps）

```bash
python scripts/bootstrap_retrieval_assets.py
```

### 6.3 跑 proving 消融

```bash
ROOT="$PWD"
RET_CKPT="$(python - <<'PY'
from lean_dojo_v2.utils.filesystem import find_latest_checkpoint
print(find_latest_checkpoint())
PY
)"

python LeanRAG_Explorer/run_proving_ablation.py \
  --url "https://github.com/durant42040/lean4-example" \
  --commit "005de00d03f1aaa32cb2923d5e3cbaf0b954a192" \
  --database-path "dynamic_database.json" \
  --output-dir "outputs/proving_ablation" \
  --max-theorems 3 \
  --ret-ckpt-path "$RET_CKPT" \
  --gen-ckpt-path "$ROOT/raid/model_lightning.ckpt" \
  --corpus-jsonl-path "$ROOT/raid/data/merged/corpus.jsonl"
```

> 建议先 `--max-theorems 3` 烟测，之后再扩大到 20/50。

---

## 7. 输出文件

### 检索层

- `outputs/ablation/metrics.csv`
- `outputs/ablation/error_cases.jsonl`
- `outputs/ablation/summary.json`

### proving 层

- `outputs/proving_ablation/proving_results.csv`
- `outputs/proving_ablation/summary.json`

---

## 8. 研究定位（当前阶段）

`LeanRAG_Explorer` 目前定位为：

- 一个强调可解释、可消融、可复现的研究工程平台
- 主打“query 改造对最终 proving 成功率的影响”
- 先在 noDeps 稳定 setting 下得出可靠结论，再向 full deps 扩展

它不是最终版产品，而是用于快速迭代研究假设与工程验证的实验平台。

