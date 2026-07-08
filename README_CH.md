# RecForest

RecForest 是一个围绕 Gowalla 数据集构建的研究型推荐系统仓库。

仓库包含两条主要实验链路：

- `notebooks/gowalla/gowalla_DIN.ipynb`：`DINTrain` 基线
- `notebooks/gowalla/gowalla.ipynb`：多树 `Trm4Rec` + rerank 主方法

这个仓库主要由 notebook 驱动，没有标准的 `pytest`、`make` 或 CI 流程。

## 项目做什么

这个项目研究大规模序列推荐问题。

给定用户的历史 item 序列，模型尝试推荐未来的 item。

仓库当前重点包括：

- Gowalla 风格数据上的序列推荐
- 用 `DIN` 作为 `(用户历史, 候选 item)` 打分基线
- 用树结构 Transformer 检索模型 `Trm4Rec` 预测 item path，而不是直接对所有 item 打分

## 仓库结构

- `lib/`
  - 核心模型和预处理代码
- `notebooks/gowalla/`
  - 主入口 notebook
- `data/gowalla/`
  - 预处理后的训练 / 验证 / 测试数据、已保存模型、树文件
- `scripts/`
  - 小型分析脚本

重要文件：

- `lib/DIN_Model.py`：Deep Interest Network 实现
- `lib/DIN_trainer.py`：DIN 的训练与推理辅助代码
- `lib/Trm4Rec_trainer.py`：树结构 Transformer 的训练与推理
- `lib/Tree_Model.py`：树编码 / 解码逻辑
- `lib/generate_train_and_test_data.py`：原始数据到样本文件的预处理
- `lib/generate_training_batches.py`：样本文件读取与 batch 构造

## 数据布局

仓库默认使用已经预处理好的数据。

`data/gowalla/` 中已包含：

- `train_instances_0..9`
- `test_instances`
- `validation_instances`
- `user_item_num.txt`
- `DIN_MODEL.pt`
- `model/`
- `tree/`

未包含：

- 原始 `data/gowalla/gowalla.txt`

`user_item_num.txt` 保存两个数字：

- 第 1 行：用户数
- 第 2 行：item 数

## 样本格式

训练样本（`train_instances_*`）格式：

```text
user|history_1,...,history_69|label
```

评估样本（`test_instances`、`validation_instances`）格式：

```text
user|history_1,...,history_69|future_label_1,...,future_label_k
```

也就是说：

- 训练时使用单个 next-item 标签
- 评估时使用未来 item 集合作为目标

## 如何运行

notebook 假设当前工作目录是 `notebooks/gowalla`。

例如：

```bash
jupyter notebook
```

然后打开：

- `notebooks/gowalla/gowalla_DIN.ipynb`
- `notebooks/gowalla/gowalla.ipynb`

## 依赖

仓库没有完整保存原始项目环境文件。

从代码看，至少依赖：

- `torch`
- `transformers`
- `numpy`
- `joblib`
- `tqdm`
- `pandas`
- `matplotlib`
- Jupyter / IPython

## 重要说明

- notebook 默认走 `have_processed_data=True` 路径。
- `Train_instance.training_batches()`、`test_batches()`、`validation_batches()`、`generate_training_records()` 都是无限生成器。
- `DINTrain.update_DIN()` 内部会自己执行 `backward()`、`step()`、`zero_grad()`。
- `Trm4Rec.update_model()` 只返回 loss，优化器更新由 notebook 外层负责。
- `gowalla.ipynb` 依赖已生成好的树文件。当前仓库中的 `tree/` 不包含完全复现实验所需的全部文件。
- `lib/__init__.py` 可以容忍缺失的 `JTM_variant.py`，当前仓库中这个文件并不存在。

## 关于原始 Gowalla 的现状

预处理代码假设原始文件是一个自定义 5 列格式：

```text
user_id,item_id,cat_id,behavior,timestamp
```

这和常见公开 Gowalla raw check-in 格式并不一致。

这意味着：

仓库大概率不是直接使用标准公开 Gowalla raw，而是先使用了一个中间转换版本。

## 推荐阅读顺序

如果你想快速理解这个项目，推荐顺序：

1. `notebooks/gowalla/gowalla_DIN.ipynb`
2. `lib/generate_training_batches.py`
3. `lib/DIN_trainer.py`
4. `lib/DIN_Model.py`
5. `lib/generate_train_and_test_data.py`
6. `notebooks/gowalla/gowalla.ipynb`
7. `lib/Trm4Rec_trainer.py`
8. `lib/Tree_Model.py`
