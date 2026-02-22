# Trainer 模块文档

`lean_dojo_v2/trainer/` 提供了四种针对 Lean 4 形式化定理证明的训练器，覆盖从监督微调、强化学习到检索模型训练、辅助回归模型训练的完整训练流程。

---

## 目录结构

```
trainer/
├── __init__.py           # 统一导出四个 Trainer
├── sft_trainer.py        # 监督微调 (SFT)
├── grpo_trainer.py       # 强化学习 (GRPO)
├── retrieval_trainer.py  # 前提检索模型训练 (RAG)
└── progress_trainer.py   # 证明进度预测模型训练 (Regression)
```

---

## 总览

| Trainer | 训练范式 | 基础框架 | 核心目标 |
|---|---|---|---|
| `SFTTrainer` | 监督微调 | `trl.SFTTrainer` | 让 LLM 学习给定证明状态输出下一步 tactic |
| `GRPOTrainer` | Group Relative Policy Optimization | `trl.GRPOTrainer` | 用奖励函数强化 LLM 的 tactic 生成策略 |
| `RetrievalTrainer` | 对比学习 / 检索微调 | `pytorch_lightning` | 微调 ByT5-based 前提检索器（RAG） |
| `ProgressTrainer` | 回归 | `transformers.Trainer` | 训练"剩余步数预测"辅助模型 |

---

## SFTTrainer

**文件**: `sft_trainer.py`

### 功能

对任意因果语言模型（CausalLM）做**监督微调**，使用 LeanDojo 提取的 traced tactics 数据，训练模型在给定证明状态（`state_before`）时输出下一步 tactic。

### 数据格式（`SFTDataset`）

从 traced tactics JSON 文件加载，按 tactic 粒度展开，每条样本为三轮对话：

```
System: 你是一个 Lean 4 tactic 生成器，只输出单行 tactic，禁用 sorry。
User:   <state_before>    ← 当前证明状态
Assistant: <tactic>       ← 目标输出（只取第一行）
```

- 自动跳过 `sorry` 标注的 tactic
- 使用 `completion_only_loss=True`，只对 Assistant 部分计算损失

### 核心参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model_name` | — | HuggingFace 模型 ID 或本地路径 |
| `output_dir` | `"outputs"` | 模型保存目录 |
| `epochs_per_repo` | `1` | 每个仓库训练轮数 |
| `batch_size` | `1` | 每设备 batch size |
| `lr` | `2e-5` | 学习率 |
| `lora_config` | `None` | 传入则启用 LoRA，否则全参数微调 |

### 训练流程

`train(repos, database, data_path)` 按**课程顺序**逐仓库迭代训练：

```
for repo in repos:
    累积已处理仓库 → database.export_merged_data() 生成合并数据集
    加载 train.json → 构建 SFTDataset → 调用 trl.SFTTrainer.train()
    保存模型（LoRA adapter 或完整权重）
```

### 使用示例

```python
from peft import LoraConfig
from lean_dojo_v2.trainer import SFTTrainer

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
)

trainer = SFTTrainer(
    model_name="deepseek-ai/DeepSeek-Prover-V2-7B",
    output_dir="outputs-sft",
    epochs_per_repo=1,
    batch_size=2,
    lr=2e-5,
    lora_config=lora_config,
)

trainer.train(repos, database, data_path)
```

---

## GRPOTrainer

**文件**: `grpo_trainer.py`

### 功能

使用 **GRPO（Group Relative Policy Optimization）** 对因果语言模型做强化学习微调。模型从相同 prompt 生成多条 tactic 候选，通过用户定义的奖励函数打分后更新策略，比 PPO 更轻量高效。

### 数据格式（`GRPODataset`）

每条样本包含：
- `problem`：当前证明状态字符串（用于奖励函数访问）
- `prompt`：System + User 两轮对话（供模型生成 tactic）

### 核心参数

与 `SFTTrainer` 基本一致，**额外**：

| 参数 | 说明 |
|---|---|
| `reward_func` | 必传，签名为 `(completions: List[str], **kwargs) -> Tensor`，返回每条生成的得分 |

### 奖励函数示例

```python
def lean_reward(completions, **kwargs):
    """验证 tactic 是否能被 Lean 接受，成功得 1.0，失败得 0.0。"""
    rewards = []
    for tactic in completions:
        # 调用 Lean 验证逻辑
        success = verify_tactic(tactic, kwargs["problem"])
        rewards.append(1.0 if success else 0.0)
    return torch.tensor(rewards)
```

### 训练流程

与 `SFTTrainer` 相同的课程迭代方式，底层替换为 `trl.GRPOTrainer`。

### 使用示例

```python
from lean_dojo_v2.trainer import GRPOTrainer

trainer = GRPOTrainer(
    model_name="deepseek-ai/DeepSeek-Prover-V2-7B",
    reward_func=lean_reward,
    output_dir="outputs-grpo",
    epochs_per_repo=1,
    batch_size=8,
    lr=2e-5,
)
trainer.train(repos, database, data_path)
```

---

## RetrievalTrainer

**文件**: `retrieval_trainer.py`

### 功能

训练/微调 **ByT5-based 前提检索器**（`PremiseRetriever`），使其能根据当前证明状态检索出最相关的前提（lemma/定义）。这是 RAG 证明流程的核心组件。

支持：
- **渐进式训练**（Lifelong Learning）：每个新仓库只额外训练 1 epoch，防止遗忘
- **EWC 正则化**（可选）：通过 Fisher Information Matrix 控制参数偏移
- **多 GPU 分布式训练**：基于 PyTorch Lightning DDP

### 训练流程

```
for repo in repos:
    导出累积数据集（corpus.jsonl + random/train.json + val.json）
    加载最新 checkpoint（自动查找或指定路径）
    构建 RetrievalDataModule
    调用 pl.Trainer.fit(model, datamodule)
    基于 R@10 保存最佳 checkpoint
```

### 评估指标

`evaluate()` 方法计算三个检索指标：

| 指标 | 含义 |
|---|---|
| **R@1** | 正确前提在 top-1 中的召回率 |
| **R@10** | 正确前提在 top-10 中的召回率（训练时的主要验证指标） |
| **MRR** | 平均倒数排名（Mean Reciprocal Rank） |

### 配置

通过 `TrainingConfig`（来自 `lean_agent/config.py`）控制：
- GPU 数量、seed、EWC lambda 值
- Checkpoint 目录、日志目录
- DataModule 相关参数

### 使用示例

```python
from lean_dojo_v2.trainer import RetrievalTrainer
from lean_dojo_v2.lean_agent.config import TrainingConfig

config = TrainingConfig(num_gpus=4, seed=42)
trainer = RetrievalTrainer(config=config)
trainer.train(repos, database, data_path="raid/data/merged")
trainer.evaluate()
```

---

## ProgressTrainer

**文件**: `progress_trainer.py`

### 功能

训练一个**证明进度预测**回归模型（`LeanProgress`），输入当前证明状态 + 候选 tactic，预测距证明结束还剩多少步（`steps_remaining`）。该预测值可用于在 best-first 搜索中引导节点选择。

### 数据格式（`ProgressDataset`）

从 JSONL 文件加载，每行格式：

```jsonl
{"goal": "⊢ n + 0 = n", "prefix": "induction n", "tactic": "rfl", "steps_remaining": 1}
```

| 字段 | 说明 |
|---|---|
| `goal` | 当前证明目标 |
| `prefix` | 已执行的 tactic 序列（可选） |
| `tactic` | 候选 tactic |
| `steps_remaining` | 回归标签，剩余步数 |

输入文本拼接格式：

```
Goal:
<goal>

Prefix:
<prefix>

Candidate tactic:
<tactic>
```

### 模型结构

使用 `AutoModelForSequenceClassification`（num_labels=1, problem_type="regression"），即在分类头的基础上做单值回归输出。

### 核心参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model_name` | `"bert-base-uncased"` | 骨干编码器 |
| `data_path` | — | JSONL 数据文件路径 |
| `output_dir` | `"outputs"` | 保存目录 |
| `max_length` | `512` | 最大 token 长度 |
| `batch_size` | `8` | Batch size |
| `epochs` | `3.0` | 训练轮数 |
| `learning_rate` | `1e-5` | 学习率 |
| `eval_ratio` | `0.2` | 验证集比例（自动切分） |

### 评估指标

| 指标 | 说明 |
|---|---|
| `mse` | 均方误差（主要选模指标） |
| `mae` | 平均绝对误差 |

### 使用示例

```python
from lean_dojo_v2.trainer import ProgressTrainer

trainer = ProgressTrainer(
    model_name="bert-base-uncased",
    data_path="raid/data/sample_leanprogress_dataset.jsonl",
    output_dir="outputs-progress",
    max_length=512,
    batch_size=8,
    epochs=3.0,
    learning_rate=1e-5,
)
trainer.train()
```

训练完成后设置环境变量即可在证明搜索中启用：

```bash
export LEANPROGRESS_MODEL=outputs-progress
```

---

## 模块间关系

```
DynamicDatabase
    └─ export_merged_data()
            ↓
    train.json / corpus.jsonl
            ↓
  ┌─────────────────────────────────┐
  │  SFTTrainer   GRPOTrainer       │  ← 训练 tactic 生成器 (LLM)
  │  RetrievalTrainer               │  ← 训练前提检索器 (RAG Retriever)
  │  ProgressTrainer                │  ← 训练证明进度预测器 (辅助)
  └─────────────────────────────────┘
            ↓
    lean_agent/prover/  ← 推理时集成上述模型进行 best-first 证明搜索
```

---

## 依赖关系

| 包 | 用于 |
|---|---|
| `trl` | SFTTrainer、GRPOTrainer |
| `peft` | LoRA 支持（SFT/GRPO） |
| `transformers` | 所有模型加载、ProgressTrainer |
| `pytorch_lightning` | RetrievalTrainer 分布式训练 |
| `datasets` | HF Dataset 构建 |
| `torch` | 所有模型 |
