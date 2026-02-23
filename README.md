# LeanRAG_Explorer

`LeanRAG_Explorer` 是一个基于 LeanDojo-v2 的检索增强定理证明研究仓库，聚焦于 **Premise Selection RAG** 的可解释改造与端到端验证。  
项目目标不是替换 LeanDojo，而是在其原有 tracing/training/proving 框架上，以最小侵入方式系统评估“检索阶段改造”对最终证明成功率的影响。

---

## 1) 为什么做这个项目

在 ReProver 风格流水线中，检索通常是：

`current_state -> dense retriever -> top-k premises -> tactic generator`

这个流程有效，但在真实 Lean 任务里有三个痛点：

- **Query 单视角**：只看当前 state，缺失 theorem 宏观目标和证明轨迹信息。
- **Dense 单路偏置**：语义检索强，但符号精确匹配（如命名、类型构造器）经常被稀释。
- **评测不闭环**：Recall@K 不能完全代表 proving 能力，尤其在“同题多解”场景下。

因此，本仓库以“可复现实验”为核心，构建从检索到 proving 的完整消融闭环。

---

## 2) 核心方法：双路检索 + 级联搜索

本项目采用两个关键思想：

### A. 双路检索（Dual-Track Retrieval）

同一个证明时刻构造多视角 query，并路由到不同检索器：

- `formal_query` -> Dense / Sparse（结构化 proof-state 视角）
- `nl_query` -> LLM Retriever（语言概括视角）

然后做 late fusion（Linear/RRF）得到统一候选集合，并保留 provenance 便于误差分析。

### B. 级联搜索（Cascaded Retrieval + Proving）

采用“先粗检索、后约束、再证明”的级联流程：

1. Query 构造（raw / goal / temporal / denoised / macro）
2. 检索召回（Dense/Sparse/LLM/Hybrid）
3. 合法性过滤（DAG accessibility / no-future constraints）
4. 输入 tactic generator 做 proving 搜索

这种级联能把“检索改造”的效果独立出来，避免与 proving 策略混淆。

---

## 3) 当前实现内容

### 3.1 Query 改造（7 个主变体）

- `raw_state`：原始 state（基线）
- `goal_only`：只保留 goal
- `macro_context`：theorem statement + state
- `temporal_context`：近期 tactic 历史 + state
- `denoised_state`：goal + 关键词筛选 hypothesis
- `recent_states_context`：近期 3 条 state + 当前 state
- `dual_summary_fusion`：state 检索 + LLM 历史总结检索，归一化后加权融合

### 3.2 检索模块

- 统一抽象：`BaseRetriever`
- 实现类型：Dense / Sparse(BM25) / LLM / Hybrid
- 融合策略：Linear、RRF
- 结果追踪：dense_score、sparse_score、rank provenance

### 3.3 评测与实验脚本

- `run_ablation.py`：检索层（Recall@K / 过滤前后）
- `run_proving_ablation.py`：端到端 proving（success、steps、duration）
- `scripts/bootstrap_retrieval_assets.py`：trace + retriever资产准备

---

## 4) 与 LeanDojo-v2 的边界

本仓库遵循“轻侵入原则”：

- 保留 LeanDojo-v2 的 database/tracing/prover 主流程；
- 只在 query/retrieval 层做策略注入；
- 尽可能固定 theorem 集合与预算，保证对比公平。

这保证实验结论能直接映射到 LeanDojo 主线工程，而不是孤立 demo。

---

## 5) 工程现实与主线策略

在服务器环境中，`build_deps=True` 会遇到 toolchain/cache/依赖追踪兼容问题（例如 `.olean` header mismatch、`ExtractData.lean` 失败）。

因此当前采用双轨工程策略：

- **主线（main）**：默认 `noDeps`，优先保证实验可复现
- **扩展线（future branch）**：再推进 `build_deps=True` 全依赖实验

这不会改变研究核心问题：在同一 setting 下比较 query 改造的相对收益。

---

## 6) 快速开始（推荐主线：noDeps）

### 6.1 环境

```bash
conda create -n leanrag python=3.11 -y
conda activate leanrag
pip install -e ".[dev]"
pip install git+https://github.com/stanford-centaur/PyPantograph.git
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

```bash
export GITHUB_ACCESS_TOKEN="<your_token>"
export HF_TOKEN="<your_hf_token>"
```

### 6.2 资产准备（默认 noDeps）

```bash
python scripts/bootstrap_retrieval_assets.py
```

如需全依赖 tracing（实验扩展）：

```bash
python scripts/bootstrap_retrieval_assets.py --build-deps
```

### 6.3 proving 消融（先小样本烟测）

#### A) repo 模式（原始方式）

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

通过后再把 `--max-theorems` 提升到 20/50。

#### B) dataset 模式（推荐用于固定数据集复现实验）

先从 Hugging Face 下载并去重（去重键：`url+commit+file_path+full_name+start+end`）：

```bash
python scripts/download_hf_proving_dataset.py \
  --dataset "cat-searcher/leandojo-benchmark-4-random-sft" \
  --split train \
  --output outputs/datasets/leandojo_benchmark4_proving_dedup.jsonl
```

然后直接按 dataset 跑 proving 消融（不再需要 `--url/--commit`）：

```bash
ROOT="$PWD"
RET_CKPT="$(python - <<'PY'
from lean_dojo_v2.utils.filesystem import find_latest_checkpoint
print(find_latest_checkpoint())
PY
)"

python LeanRAG_Explorer/run_proving_ablation.py \
  --dataset-path "outputs/datasets/leandojo_benchmark4_proving_dedup.jsonl" \
  --database-path "dynamic_database.json" \
  --reuse-db \
  --output-dir "outputs/proving_ablation" \
  --max-theorems 50 \
  --ret-ckpt-path "$RET_CKPT" \
  --gen-ckpt-path "$ROOT/raid/model_lightning.ckpt" \
  --corpus-jsonl-path "$ROOT/raid/data/merged/corpus.jsonl"
```

若要跑纯生成组（不检索），将 `--ret-ckpt-path ""`，并省略 `--corpus-jsonl-path`。

---

## 7) 输出与分析文件

### 检索层

- `outputs/ablation/metrics.csv`
- `outputs/ablation/error_cases.jsonl`
- `outputs/ablation/summary.json`

### proving 层

- `outputs/proving_ablation/proving_results.csv`
- `outputs/proving_ablation/summary.json`

建议优先报告：

- success rate（主指标）
- avg steps / avg duration（效率指标）
- 不同 query 变体在失败样本中的模式差异（解释性指标）

---

## 8) 当前定位与后续计划

当前阶段定位：

- 一个可解释、可消融、可复现的 LeanRAG 研究平台；
- 主结论来自端到端 proving，而非仅检索 recall；
- 先在 noDeps setting 形成稳定结论，再向 full deps 扩展。

后续工作（路线图）：

1. 将 noDeps 结论迁移到 build_deps=True 设置；
2. 进一步引入 Reranker（cross-encoder/LLM judge）；
3. 引入动态反馈检索（tactic error-aware re-query）；
4. 做更大规模 repo 和 theorem 分层评测。

