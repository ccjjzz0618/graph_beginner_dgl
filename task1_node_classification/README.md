# 任务一：节点分类

支持 Cora、Citeseer、Flickr，模型为 GCN、GAT、GraphSAGE、GIN。`full` 在完整图上传播，`sampled` 使用 DGL `MultiLayerNeighborSampler` 逐层采样邻居。

## 训练

从项目根目录运行：

```powershell
# 全图训练
.\.venv\Scripts\python.exe .\task1_node_classification\code\train.py --dataset cora --model gcn --mode full --epochs 200 --device cuda

# 子图采样训练；fanout 数量须等于层数（单个值会自动复制）
.\.venv\Scripts\python.exe .\task1_node_classification\code\train.py --dataset flickr --model graphsage --mode sampled --layers 2 --fanout 15,10 --batch-size 1024 --device cuda
```

可选关键参数：`--learning-rate`、`--layers`、`--hidden-dim`、`--dropout`、`--patience`、`--seed`。

## 测试检查点

```powershell
.\.venv\Scripts\python.exe .\task1_node_classification\code\evaluate.py --checkpoint .\task1_node_classification\results\<run>.pt --device cuda
```

主要指标为测试集 Accuracy。JSON 同时记录最佳验证轮、训练总耗时、平均 epoch 耗时和峰值 CUDA 显存。

