# 任务二：图上的链路预测

支持 Cora、Citeseer、Flickr 和 GCN/GAT/GraphSAGE/GIN。GNN 产生节点嵌入，点积解码器输出边分数。

实现会先对唯一无向边对做 train/validation/test 切分，反向边不会跨 split；负样本保证不是原图中的真实边。

## 训练

```powershell
# 全图编码
.\.venv\Scripts\python.exe .\task2_link_prediction\code\train.py --dataset cora --model gat --mode full --epochs 200 --device cuda

# EdgeDataLoader + NeighborSampler
.\.venv\Scripts\python.exe .\task2_link_prediction\code\train.py --dataset flickr --model graphsage --mode sampled --fanout 15,10 --batch-size 2048 --device cuda
```

采样模式利用 DGL edge-prediction sampler，同时从消息流图排除目标边和反向边。

## 测试检查点

```powershell
.\.venv\Scripts\python.exe .\task2_link_prediction\code\evaluate.py --checkpoint .\task2_link_prediction\results\<run>.pt --device cuda
```

报告 ROC-AUC 与 Average Precision；两者均越高越好。

