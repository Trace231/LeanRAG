# LeanRAG_Explorer

`LeanRAG_Explorer` 是一个面向 Lean 4 自动定理证明的检索增强（RAG）研究仓库，核心目标是系统性研究 **Premise Selection**（前提检索）机制，并支持可复现实验与误差分析。

---

## 项目目的

当前 ReProver/LeanDojo 生态中，前提检索通常依赖 Dense Retriever（如 ByT5），但实际证明场景同时存在：

- 符号精确匹配需求（Sparse 更强）
- 语义泛化匹配需求（Dense 更强）
- 合法性约束需求（必须满足 Lean import DAG 可访问性）

本项目的核心思路是构建一个 **高度解耦的 Hybrid Retrieval Pipeline**：

- 多视角 Query 构建（formal + nl）
- 多检索器并行检索（Dense / Sparse / LLM）
- Late Fusion 融合（Linear / RRF）
- DAG 合法性过滤
- 统一评测与错误分析（Recall + provenance）

---

## 核心能力

- **Query Builder 策略化**：支持 A/B/C/D 与 Dual-Track 等多种 query 方案
- **Retriever 统一接口**：支持 mock 与真实实现平滑替换
- **Hybrid Fusion**：支持线性加权与 RRF
- **Provenance 追踪**：每个检索结果记录 dense/sparse 分数与 rank
- **Legality Filter**：支持过滤前/过滤后指标对比
- **Ablation Pipeline**：一键跑组合实验并导出分析文件

---

## 目录结构

```text
LeanRAG_Explorer/
├── README.md
├── requirements.txt
├── run_ablation.py
└── leanrag_explorer/
    ├── __init__.py
    ├── types.py
    ├── query_builders/
    │   ├── __init__.py
    │   ├── base.py
    │   └── strategies.py
    ├── retrievers/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── hybrid.py
    │   ├── mock.py
    │   └── real.py
    ├── filters/
    │   ├── __init__.py
    │   └── accessibility.py
    └── evaluation/
        ├── __init__.py
        └── evaluator.py
```

---

## 环境安装

```bash
conda create -n leanrag python=3.11 -y
conda activate leanrag
pip install -r requirements.txt
```

如果使用 OpenAI 相关组件：

```bash
export OPENAI_API_KEY="your_api_key"
```

---

## 输入数据格式

### 1) Dataset (`--dataset`, json/jsonl)

```json
{
  "theorem_statement": "...",
  "current_state": "...",
  "recent_tactics": ["simp", "rw [h]"],
  "hypotheses": ["h : a = b"],
  "goal": "a = b",
  "metadata": {"theorem_id": "T1"},
  "gold_premise_ids": ["Mathlib.foo.bar"]
}
```

### 2) Premises (`--premises`, json/jsonl)

```json
{
  "premise_id": "Mathlib.foo.bar",
  "text": "theorem Mathlib.foo.bar : ..."
}
```

### 3) Allowed map (`--allowed-map`, optional)

```json
{
  "T1": ["Mathlib.foo.bar", "Mathlib.baz.qux"]
}
```

---

## 运行实验

```bash
python run_ablation.py \
  --dataset data/eval_samples.jsonl \
  --premises data/premises.jsonl \
  --allowed-map data/allowed_map.json \
  --output-dir outputs/ablation \
  --top-k 100 \
  --error-k 10 \
  --alpha 0.6 \
  --rrf-k 60
```

---

## 输出结果

运行完成后会生成：

- `metrics.csv`：每个组合的 Recall@1/10/50（过滤前后）
- `error_cases.jsonl`：失败样本（含 provenance 字段）
- `summary.json`：运行摘要

其中 `error_cases.jsonl` 可用于分析：

- Dense/Sparse 分歧样本
- 过滤后召回下降原因
- Query strategy 导致的 miss 模式

---

## 研究路线建议

1. 先跑 `dense_only` / `sparse_only` 基线  
2. 加入 `hybrid_rrf_dense_sparse`  
3. 对比不同 QueryBuilder（macro / temporal / denoised / dual-track）  
4. 最后加入 `LLMRetriever` 与 `OpenAISummaryGenerator` 做成本更高实验

---

## 当前定位

这个仓库不是“直接追求 SOTA”的训练工程，而是一个可解释、可迭代、可做消融的研究平台。重点在于回答：

- 什么 query 表示最有效？
- 什么时候 Dense 强、什么时候 Sparse 强？
- DAG 约束对有效召回的真实影响有多大？
- 融合策略与 provenance 能否帮助定位检索失败根因？

