# 实验结果汇总

> 本文件由 `scripts/generate_report.py` 从各任务的 JSON 结果自动生成。

> 当前包含 smoke 流程验证结果；其 epoch 很少，不能作为正式模型优劣结论。

## 主要观察

- 节点分类核心对比最佳为 cora/gcn/full，测试准确率 0.8260。
- 链路预测核心对比最佳为 cora/graphsage/full，测试 AUC 0.8919。
- 节点分类核心实验中，full/sampled 平均指标为 0.7898/0.7865，平均总训练时间为 0.350/0.749 秒；在 Cora 这类小图上，采样开销没有转化为速度优势。
- 链路预测核心实验中，full/sampled 平均指标为 0.7670/0.7465，平均总训练时间为 0.469/4.367 秒；在 Cora 这类小图上，采样开销没有转化为速度优势。
- 图分类当前最佳池化为 max（gin），测试准确率 0.7222。
- 受控参数研究最佳为学习率 0.01、2 层，测试准确率 0.8260。

## 全图与采样模式汇总（核心实验均值）

| task   | mode    |   mean_metric |   mean_seconds |   mean_peak_MB |
|:-------|:--------|--------------:|---------------:|---------------:|
| 节点分类   | full    |        0.7898 |         0.3498 |        62.1006 |
| 节点分类   | sampled |        0.7865 |         0.7485 |        62.8047 |
| 链路预测   | full    |        0.7670 |         0.4686 |        67.7122 |
| 链路预测   | sampled |        0.7465 |         4.3671 |        81.2856 |

## 任务一：节点分类

| run   | dataset   | model     | mode    |     lr |   layers |   test_acc |   seconds |   peak_MB |
|:------|:----------|:----------|:--------|-------:|---------:|-----------:|----------:|----------:|
| core  | cora      | gat       | full    | 0.0100 |        2 |     0.8130 |    0.4234 |   67.8887 |
| core  | cora      | gat       | sampled | 0.0100 |        2 |     0.7690 |    0.7718 |   60.3486 |
| core  | cora      | gcn       | full    | 0.0100 |        2 |     0.8260 |    0.4153 |   50.3359 |
| core  | cora      | gcn       | sampled | 0.0100 |        2 |     0.8140 |    0.4979 |   56.0137 |
| core  | cora      | gin       | full    | 0.0100 |        2 |     0.7170 |    0.2942 |   77.7017 |
| core  | cora      | gin       | sampled | 0.0100 |        2 |     0.7530 |    0.8891 |   84.7476 |
| core  | cora      | graphsage | full    | 0.0100 |        2 |     0.8030 |    0.2662 |   52.4761 |
| core  | cora      | graphsage | sampled | 0.0100 |        2 |     0.8100 |    0.8353 |   50.1089 |
| smoke | citeseer  | gcn       | full    | 0.0100 |        2 |     0.2370 |    0.1764 |  115.4434 |
| smoke | cora      | gcn       | full    | 0.0100 |        2 |     0.5360 |    0.2538 |   50.3359 |
| smoke | cora      | gcn       | sampled | 0.0100 |        2 |     0.6830 |    0.1814 |   55.8960 |
| smoke | flickr    | graphsage | sampled | 0.0100 |        2 |     0.4233 |    1.0511 |  420.7876 |
| sweep | cora      | gcn       | full    | 0.0010 |        2 |     0.7850 |    0.4172 |   50.3359 |
| sweep | cora      | gcn       | full    | 0.0010 |        3 |     0.7360 |    0.4009 |   51.8936 |
| sweep | cora      | gcn       | full    | 0.0010 |        4 |     0.7790 |    0.4894 |   53.4512 |
| sweep | cora      | gcn       | full    | 0.0100 |        2 |     0.8260 |    0.3944 |   50.3359 |
| sweep | cora      | gcn       | full    | 0.0100 |        3 |     0.7980 |    0.4272 |   51.8936 |
| sweep | cora      | gcn       | full    | 0.0100 |        4 |     0.7970 |    0.4987 |   53.4512 |
| sweep | cora      | gcn       | full    | 0.0500 |        2 |     0.8060 |    0.3849 |   50.3359 |
| sweep | cora      | gcn       | full    | 0.0500 |        3 |     0.7420 |    0.5316 |   51.8936 |
| sweep | cora      | gcn       | full    | 0.0500 |        4 |     0.7660 |    0.6061 |   53.4512 |

## 学习率与层数研究

|     lr |   layers |   test_acc |   seconds |   peak_MB |
|-------:|---------:|-----------:|----------:|----------:|
| 0.0010 |   2.0000 |     0.7850 |    0.4172 |   50.3359 |
| 0.0010 |   3.0000 |     0.7360 |    0.4009 |   51.8936 |
| 0.0010 |   4.0000 |     0.7790 |    0.4894 |   53.4512 |
| 0.0100 |   2.0000 |     0.8260 |    0.3944 |   50.3359 |
| 0.0100 |   3.0000 |     0.7980 |    0.4272 |   51.8936 |
| 0.0100 |   4.0000 |     0.7970 |    0.4987 |   53.4512 |
| 0.0500 |   2.0000 |     0.8060 |    0.3849 |   50.3359 |
| 0.0500 |   3.0000 |     0.7420 |    0.5316 |   51.8936 |
| 0.0500 |   4.0000 |     0.7660 |    0.6061 |   53.4512 |

## 任务二：链路预测

| run   | dataset   | model     | mode    |   test_auc |   test_AP |   seconds |   peak_MB |
|:------|:----------|:----------|:--------|-----------:|----------:|----------:|----------:|
| core  | cora      | gat       | full    |     0.5874 |    0.5809 |    0.3996 |   76.2334 |
| core  | cora      | gat       | sampled |     0.5393 |    0.5415 |    2.6662 |   93.3198 |
| core  | cora      | gcn       | full    |     0.7539 |    0.7486 |    0.4333 |   57.1641 |
| core  | cora      | gcn       | sampled |     0.7955 |    0.7855 |    5.1261 |   68.6235 |
| core  | cora      | gin       | full    |     0.8347 |    0.8239 |    0.5018 |   78.4751 |
| core  | cora      | gin       | sampled |     0.7815 |    0.7793 |    4.4503 |   93.1650 |
| core  | cora      | graphsage | full    |     0.8919 |    0.8664 |    0.5395 |   58.9761 |
| core  | cora      | graphsage | sampled |     0.8698 |    0.8453 |    5.2260 |   70.0342 |
| smoke | citeseer  | gcn       | full    |     0.6624 |    0.6962 |    0.1767 |  116.2847 |
| smoke | cora      | graphsage | full    |     0.5826 |    0.5847 |    0.2015 |   58.9761 |
| smoke | cora      | graphsage | sampled |     0.6667 |    0.6729 |    0.3099 |   69.8525 |
| smoke | flickr    | graphsage | sampled |     0.5508 |    0.5314 |   12.2926 |  619.9595 |

## 任务三：图分类/回归

| run   | dataset   | task_type      | model     | pooling   | mode   | metric   |   test_metric |   seconds |   peak_MB |
|:------|:----------|:---------------|:----------|:----------|:-------|:---------|--------------:|----------:|----------:|
| core  | mutag     | classification | gat       | avg       | mini   | accuracy |        0.5556 |    7.3955 |   30.8521 |
| core  | mutag     | classification | gat       | max       | mini   | accuracy |        0.5556 |    7.0099 |   30.8521 |
| core  | mutag     | classification | gat       | min       | mini   | accuracy |        0.5556 |    6.7889 |   30.8521 |
| core  | mutag     | classification | gcn       | avg       | mini   | accuracy |        0.5556 |    7.0894 |   19.3618 |
| core  | mutag     | classification | gcn       | max       | mini   | accuracy |        0.5556 |    6.8777 |   19.3618 |
| core  | mutag     | classification | gcn       | min       | mini   | accuracy |        0.5556 |    6.7571 |   19.3618 |
| core  | mutag     | classification | gin       | avg       | mini   | accuracy |        0.5556 |    5.7374 |   20.2168 |
| core  | mutag     | classification | gin       | max       | mini   | accuracy |        0.7222 |   11.3057 |   20.2168 |
| core  | mutag     | classification | gin       | min       | mini   | accuracy |        0.5556 |    6.4968 |   20.2842 |
| core  | mutag     | classification | graphsage | avg       | mini   | accuracy |        0.5556 |    6.0073 |   20.0488 |
| core  | mutag     | classification | graphsage | max       | mini   | accuracy |        0.5556 |    6.4837 |   20.0488 |
| core  | mutag     | classification | graphsage | min       | mini   | accuracy |        0.5556 |    5.9943 |   20.0488 |
| smoke | mutag     | classification | gin       | avg       | mini   | accuracy |        0.5556 |    0.9905 |   20.1099 |
| smoke | zinc      | regression     | gcn       | avg       | mini   | mae      |        1.5186 |   12.2765 |   34.7021 |

## 任务四：知识图谱补全

| run   | dataset   | model   |   eval_triples |    MRR |   Hits@1 |   Hits@3 |   Hits@10 |   seconds |
|:------|:----------|:--------|---------------:|-------:|---------:|---------:|----------:|----------:|
| smoke | fb15k237  | conve   |              5 | 0.0155 |   0.0000 |   0.0000 |    0.1000 |    2.4959 |
| smoke | fb15k237  | rotate  |              5 | 0.0003 |   0.0000 |   0.0000 |    0.0000 |    2.0687 |
| smoke | fb15k237  | transe  |             20 | 0.0483 |   0.0250 |   0.0500 |    0.1250 |    2.2015 |

## 解释口径

- 速度仅在相同机器、相同数据与相同 epoch 设置下比较。
- ZINC 是图回归数据集，因此报告 MAE（越低越好），而不是分类准确率。
- 知识图谱结果只有在 `eval_triples` 等于完整测试集大小时才是完整 filtered 指标。
- 多次随机种子实验应报告均值与标准差；单次结果只用于流程验证。
