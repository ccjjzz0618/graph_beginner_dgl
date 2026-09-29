# 任务三：图分类与 ZINC 回归

支持 TUDataset 的 MUTAG、PROTEINS、ENZYMES，以及 ZINC。模型为 GCN、GAT、GraphSAGE、GIN；池化为 Avg、Max、Min。

- TUDataset：分类，损失为 Cross Entropy，指标为 Accuracy。
- ZINC：受限溶解度回归，损失为 MSE，指标为 MAE。
- `full`：所有训练图组成一个大 batch；`mini`：常规小批训练。

## 训练

```powershell
# TUDataset 图分类
.\.venv\Scripts\python.exe .\task3_graph_classification\code\train.py --dataset mutag --model gin --pooling avg --mode mini --epochs 200 --device cuda

# 更换池化方法
.\.venv\Scripts\python.exe .\task3_graph_classification\code\train.py --dataset proteins --model gcn --pooling min --mode full --device cuda

# ZINC 图回归
.\.venv\Scripts\python.exe .\task3_graph_classification\code\train.py --dataset zinc --model graphsage --pooling max --mode mini --batch-size 128 --device cuda
```

## 测试检查点

```powershell
.\.venv\Scripts\python.exe .\task3_graph_classification\code\evaluate.py --checkpoint .\task3_graph_classification\results\<run>.pt --device cuda
```

比较池化方法时，必须保持同一数据切分、模型、学习率、层数和随机种子。

