# 图神经网络 Beginner：DGL 完整实现

本项目使用 **DGL + PyTorch** 完成原始作业的四项任务，并统一支持 GPU、固定随机种子、早停、检查点、JSON 结果、训练耗时和峰值显存记录。

## 已实现内容

| 任务 | 数据集 | 模型/方法 | 评测 |
|---|---|---|---|
| 节点分类 | Cora、Citeseer、Flickr | GCN、GAT、GraphSAGE、GIN；全图与 NeighborSampler | Accuracy、耗时、显存 |
| 链路预测 | Cora、Citeseer、Flickr | 四种 GNN + 点积解码；全图与边批采样 | ROC-AUC、AP、耗时、显存 |
| 图级预测 | TUDataset（MUTAG/PROTEINS/ENZYMES）、ZINC | 四种 GNN；Avg/Max/Min Pooling；全批与小批 | 分类 Accuracy；ZINC 回归 MAE |
| 知识图谱补全 | FB15k-237、FB15k、WN18 | TransE、RotatE、ConvE | filtered MRR、MR、Hits@1/3/10 |

> ZINC 的真实任务是分子属性回归，因此本实现使用 MSE 训练、MAE 评测，不把它错误地当作分类数据集。

## 1. 环境安装（Windows + NVIDIA GPU）

已针对 Python 3.11、DGL 2.2.1、PyTorch 2.1.2、CUDA 12.1 固定版本。PowerShell 在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows_gpu.ps1
.\.venv\Scripts\python.exe .\scripts\check_environment.py
```

CPU 备用环境：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows_cpu.ps1
```

所有依赖都安装在项目内的 `.venv`，不会修改全局 Python。首次运行各数据集时，DGL 会将数据下载到对应任务的 `data/` 目录。

## 2. 最快验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe .\scripts\run_benchmarks.py --suite smoke --device cuda
.\.venv\Scripts\python.exe .\scripts\generate_report.py
```

`smoke` 只验证完整流水线。正式实验使用：

```powershell
# 四种模型的核心对比
.\.venv\Scripts\python.exe .\scripts\run_benchmarks.py --suite core --device cuda

# 原作业全部数据集、模型、训练模式、池化与参数研究（运行时间较长）
.\.venv\Scripts\python.exe .\scripts\run_benchmarks.py --suite full --device cuda
```

可用 `--task task1`（或 task2/task3/task4）只运行单项，也可先加 `--dry-run` 查看将执行的命令。

## 3. 单项训练示例

```powershell
# 任务一：Flickr 上采样训练 GraphSAGE
.\.venv\Scripts\python.exe .\task1_node_classification\code\train.py --dataset flickr --model graphsage --mode sampled --fanout 15,10 --device cuda

# 任务二：Cora 全图链路预测
.\.venv\Scripts\python.exe .\task2_link_prediction\code\train.py --dataset cora --model gat --mode full --device cuda

# 任务三：MUTAG + GIN + MinPooling
.\.venv\Scripts\python.exe .\task3_graph_classification\code\train.py --dataset mutag --model gin --pooling min --mode mini --device cuda

# 任务三：ZINC 回归
.\.venv\Scripts\python.exe .\task3_graph_classification\code\train.py --dataset zinc --model gcn --pooling avg --device cuda

# 任务四：FB15k-237 RotatE；正式报告使用完整 filtered evaluation
.\.venv\Scripts\python.exe .\task4_knowledge_graph\code\train.py --dataset fb15k237 --model rotate --eval-limit 0 --device cuda
```

每个任务的 README 给出了完整训练与测试命令。训练结果写入各自的 `results/`：

- `*.json`：配置、指标、逐 epoch 历史、耗时和显存；
- `*.pt`：模型与优化器检查点；
- `reports/RESULTS.md`：由所有 JSON 自动汇总的表格和观察。

## 4. 目录结构

```text
graph_beginner_dgl/
├── common/                         # 四种通用 DGL GNN、池化、工具
├── task1_node_classification/
│   ├── data/  code/  results/  README.md
├── task2_link_prediction/
│   ├── data/  code/  results/  README.md
├── task3_graph_classification/
│   ├── data/  code/  results/  README.md
├── task4_knowledge_graph/
│   ├── data/  code/  results/  README.md
├── scripts/                        # 环境检查、实验矩阵、报告生成
├── tests/                          # 无需下载数据的单元测试
├── reports/                        # 实验方法和自动结果报告
├── requirements.txt                # CPU 环境
└── requirements-gpu-cu121.txt      # Windows CUDA 12.1 环境
```

## 5. 实验公平性和实现细节

- 所有对比保存完整超参数和随机种子；比较模型时应保持数据切分、epoch、隐藏维度一致。
- 节点分类的 sampled 模式使用 DGL `MultiLayerNeighborSampler`。
- 链路预测按唯一无向边对切分；一条边及其反向边始终属于同一 split，防止信息泄漏。采样训练会从消息图中同时排除目标边及反向边。
- 图分类的 `full` 表示一次批入全部训练图，`mini` 表示常规小批；这是图级任务中“全批/分批”可比的定义。
- MinPooling 等价实现为 `-MaxPooling(-h)`。
- 知识图谱加入 reciprocal relations；头实体预测转化为逆关系的尾实体预测。filtered 评测会屏蔽训练、验证和测试集中其他真实实体。
- `--eval-limit 2000` 是快速开发评测；正式报告必须使用 `--eval-limit 0`。
- `scripts/run_benchmarks.py --suite full` 还包含学习率与层数的受控参数实验。

## 6. 报告

方法、指标口径和分析清单见 [reports/EXPERIMENT_REPORT.md](reports/EXPERIMENT_REPORT.md)。完成实验后运行：

```powershell
.\.venv\Scripts\python.exe .\scripts\generate_report.py
```

不要手工复制终端数字；自动报告直接读取 JSON，可避免模型、数据集或训练模式错配。
