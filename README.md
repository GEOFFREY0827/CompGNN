# CompGNN: QA dataset encoding

目前实现 encoder 数据准备阶段：加载 DecomposedHarm，严格筛选 `Task == "QA"`，分别编码每条原始任务和有序子任务。保留数据集原始 split、标签和其他元数据，不将标签编码为特征。暂未实现 GNN、边构建或 RL。

## 运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

先在 Hugging Face 上申请/接受该数据集访问条件，然后登录（也可使用 `HF_TOKEN` 环境变量，不要把 token 写入代码）：

```bash
hf auth login
compgnn-encode --limit 5 --output artifacts/qa_smoke
compgnn-encode --output artifacts/qa_embeddings
```

默认模型 `sentence-transformers/all-MiniLM-L6-v2`，384 维 float32，L2 归一化；可通过 `--model`、`--device cpu|cuda|mps`、`--batch-size`、`--no-normalize` 修改。每个任务与每个子任务单独编码；超过 encoder 的 `max_seq_length` 会按模型规则截断，实际长度写入 manifest。当前是冻结 encoder 的离线推理，输出不带梯度。

由于数据集受访问限制，尚未确认实际 CSV 字段。默认筛选字段是用户指定的 `Task`；原始文本和子任务字段会尝试识别 `original_task` / `original_prompt` / `Original Task` 与 `decomposed_subtasks` / `Decomposed Subtasks` / `subtasks`。不唯一或不匹配时，脚本会报告可用字段；根据实际 schema 显式指定，例如：

```bash
compgnn-encode --task-column Task --original-column original_task --subtasks-column decomposed_subtasks
```

子任务支持原生 `list[str]`，以及 CSV 内 JSON/Python list 字符串；不猜测自由文本中的编号或分隔符。空子任务列表允许保留。`--splits train test` 可以选择 split；`--limit` 对筛选后的每个 split 生效。输出目录非空时拒绝覆盖。

## 输出与 GNN 接口

每个 split 保存 `embeddings.npz` 和 `records.jsonl`，根目录保存模型、数据来源、字段映射、维度和样本数等 `manifest.json`。

```python
import numpy as np

data = np.load("artifacts/qa_embeddings/train/embeddings.npz", allow_pickle=False)
i = 0
start, end = data["subtask_offsets"][i:i + 2]
x = data["subtask_embeddings"][start:end]  # [num_subtasks_i, dim]，按原始顺序
task_embedding = data["original_embeddings"][i]  # [dim]
```

`records.jsonl` 的第 i 行对应第 i 个任务，包含原始 `source_row` 和完整 `data`。可用 `x` 作为后续 GNN 子任务节点特征；原始任务向量独立保存，在线防御时应根据观察范围决定是否使用。尚未自动构造图边。

数据集卡请求仅用于评估；本阶段只进行编码。后续用于 GNN/RL 训练前应确认数据使用许可。来源：[DecomposedHarm](https://huggingface.co/datasets/YuehHanChen/DecomposedHarm)、[SentenceTransformer API](https://sbert.net/docs/package_reference/sentence_transformer/model.html)。

## 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

测试覆盖 QA 筛选、子任务顺序与偏移、空列表、CSV list 解析和字段歧义。真实模型端到端运行需要安装依赖并具备数据集访问权限。
