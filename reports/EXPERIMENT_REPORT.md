# DGL 图学习实验报告

## 1. 实验目标

本实验统一比较 GCN、GAT、GraphSAGE、GIN 在节点分类、链路预测和图级预测上的性能，并比较全批与采样/小批训练的精度、运行时间与显存。图级任务额外比较 AvgPooling、MaxPooling、MinPooling。知识图谱部分比较 TransE、RotatE、ConvE。

## 2. 模型原理摘要

- **GCN**：使用归一化邻接矩阵聚合邻居，结构简单、全图训练效率高。
- **GAT**：学习邻居注意力权重，表达力较强，但多头注意力增加计算和显存。
- **GraphSAGE**：对邻居采样并聚合，天然适合大图小批训练。
- **GIN**：采用 sum aggregation 与 MLP，在图同构判别意义上具有较强表达能力。
- **TransE**：用平移约束 `h + r ≈ t`，参数少、速度快，但对复杂关系受限。
- **RotatE**：在复数空间把关系建模为旋转，能表达对称、反对称、逆和组合模式。
- **ConvE**：将实体与关系嵌入重排后做二维卷积，再与候选实体打分，交互能力强。

## 3. 数据与切分

- Cora/Citeseer/Flickr 的节点分类使用 DGL 官方 mask。
- 链路预测对唯一无向边对随机切分；训练图只含训练边，反向边与原边同 split。
- TU 数据集按固定 seed 做 80%/10%/10% 图级切分。
- ZINC 使用官方 train/valid/test split。
- FB15k-237/FB15k/WN18 使用 DGL 官方三划分；filtered 评测过滤所有已知真三元组。

## 4. 控制变量

每组对比至少固定：数据 split、seed、epoch/早停规则、隐藏维度、dropout 与优化器。模型主对比建议运行 3–5 个 seed，报告均值 ± 标准差。

参数研究采用 GCN/Cora 全图作为基线：

- 学习率：0.001、0.01、0.05；
- 网络层数：2、3、4；
- 其余设置保持不变。

## 5. 结果生成

```powershell
.\.venv\Scripts\python.exe .\scripts\run_benchmarks.py --suite full --device cuda
.\.venv\Scripts\python.exe .\scripts\generate_report.py
```

生成的 `reports/RESULTS.md` 是结果表。不要在实验前填写数字，也不要把 `smoke` 或 `eval-limit > 0` 的开发结果描述为完整基准。

## 6. 必答分析清单

1. 在同一数据集上，四种 GNN 的指标差异是否大于不同 seed 的波动？
2. 学习率过大是否导致验证指标震荡，过小是否在固定 epoch 内欠拟合？
3. 网络加深后是否出现过平滑或优化困难？
4. 全图与采样训练的单 epoch 时间、总时间、峰值显存和最终指标如何变化？
5. GraphSAGE 是否在 sampled 模式中体现更好的速度/精度折中？
6. Avg/Max/MinPooling 对不同图数据集的影响是否一致？
7. ZINC 的 MAE 与分类数据的 Accuracy 不可横向比较。
8. TransE、RotatE、ConvE 的 MRR/Hits 与训练代价如何权衡？

## 7. 结论写作模板

先陈述受控实验中可复现的数值差异，再解释潜在机制；最后说明 seed 数量、抽样评测或运行预算等限制。不要仅凭单次 smoke 结果断言某模型普遍优于另一模型。

