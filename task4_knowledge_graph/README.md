# 任务四：知识图谱补全

支持 DGL 内置 FB15k-237、FB15k、WN18，模型为 TransE、RotatE、ConvE。训练时添加 reciprocal relations，使用负采样和 logistic loss。

## 训练

```powershell
# 快速开发评测（只评 2000 条）
.\.venv\Scripts\python.exe .\task4_knowledge_graph\code\train.py --dataset fb15k237 --model transe --epochs 100 --eval-limit 2000 --device cuda

# 正式完整 filtered evaluation
.\.venv\Scripts\python.exe .\task4_knowledge_graph\code\train.py --dataset fb15k237 --model rotate --epochs 100 --eval-limit 0 --device cuda

# ConvE；embedding-dim 必须能被 embedding-height 整除
.\.venv\Scripts\python.exe .\task4_knowledge_graph\code\train.py --dataset fb15k237 --model conve --embedding-dim 200 --embedding-height 10 --device cuda
```

## 测试检查点

```powershell
.\.venv\Scripts\python.exe .\task4_knowledge_graph\code\evaluate.py --checkpoint .\task4_knowledge_graph\results\<run>.pt --split test --eval-limit 0 --device cuda
```

报告 filtered Mean Rank、MRR、Hits@1、Hits@3、Hits@10。除 Mean Rank 越低越好外，其余越高越好。正式结果必须确认 JSON 中 `evaluated_triples` 等于完整 split 大小。

