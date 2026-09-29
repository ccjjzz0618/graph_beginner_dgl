from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "RESULTS.md"


def load_results() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(PROJECT_ROOT.glob("task*/results/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        payload["_path"] = str(path.relative_to(PROJECT_ROOT))
        results.append(payload)
    return results


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_暂无结果。_"
    return frame.to_markdown(index=False, floatfmt=".4f")


def run_kind(item: dict[str, Any]) -> str:
    path = item.get("_path", "")
    if "benchmark-smoke" in path:
        return "smoke"
    if "benchmark-sweep" in path:
        return "sweep"
    return "core"


def node_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in results:
        if item.get("task") != "node_classification":
            continue
        config = item["config"]
        rows.append(
            {
                "run": run_kind(item),
                "dataset": config["dataset"],
                "model": config["model"],
                "mode": config["mode"],
                "lr": config["learning_rate"],
                "layers": config["layers"],
                "test_acc": item["test_accuracy"],
                "seconds": item["total_train_seconds"],
                "peak_MB": item["peak_cuda_memory_mb"],
            }
        )
    return pd.DataFrame(rows)


def link_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in results:
        if item.get("task") != "link_prediction":
            continue
        config = item["config"]
        rows.append(
            {
                "run": run_kind(item),
                "dataset": config["dataset"],
                "model": config["model"],
                "mode": config["mode"],
                "test_auc": item["test"]["auc"],
                "test_AP": item["test"]["average_precision"],
                "seconds": item["total_train_seconds"],
                "peak_MB": item["peak_cuda_memory_mb"],
            }
        )
    return pd.DataFrame(rows)


def graph_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in results:
        if item.get("task") != "graph_prediction":
            continue
        config = item["config"]
        metric_name = "accuracy" if item["task_type"] == "classification" else "mae"
        rows.append(
            {
                "run": run_kind(item),
                "dataset": config["dataset"],
                "task_type": item["task_type"],
                "model": config["model"],
                "pooling": config["pooling"],
                "mode": config["mode"],
                "metric": metric_name,
                "test_metric": item["test"][metric_name],
                "seconds": item["total_train_seconds"],
                "peak_MB": item["peak_cuda_memory_mb"],
            }
        )
    return pd.DataFrame(rows)


def kg_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in results:
        if item.get("task") != "knowledge_graph_completion":
            continue
        config = item["config"]
        rows.append(
            {
                "run": run_kind(item),
                "dataset": config["dataset"],
                "model": config["model"],
                "eval_triples": item["test"]["evaluated_triples"],
                "MRR": item["test"]["mrr"],
                "Hits@1": item["test"]["hits_at_1"],
                "Hits@3": item["test"]["hits_at_3"],
                "Hits@10": item["test"]["hits_at_10"],
                "seconds": item["total_train_seconds"],
            }
        )
    return pd.DataFrame(rows)


def observations(nodes: pd.DataFrame, links: pd.DataFrame, graphs: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    core_nodes = nodes[nodes["run"] == "core"] if not nodes.empty else nodes
    core_links = links[links["run"] == "core"] if not links.empty else links
    core_graphs = graphs[graphs["run"] == "core"] if not graphs.empty else graphs
    sweeps = nodes[nodes["run"] == "sweep"] if not nodes.empty else nodes
    if not core_nodes.empty:
        best = core_nodes.loc[core_nodes["test_acc"].idxmax()]
        notes.append(
            f"节点分类核心对比最佳为 {best['dataset']}/{best['model']}/{best['mode']}，"
            f"测试准确率 {best['test_acc']:.4f}。"
        )
    if not core_links.empty:
        best = core_links.loc[core_links["test_auc"].idxmax()]
        notes.append(
            f"链路预测核心对比最佳为 {best['dataset']}/{best['model']}/{best['mode']}，"
            f"测试 AUC {best['test_auc']:.4f}。"
        )
    for label, frame, metric in (
        ("节点分类", core_nodes, "test_acc"),
        ("链路预测", core_links, "test_auc"),
    ):
        modes = frame.groupby("mode")[[metric, "seconds", "peak_MB"]].mean() if not frame.empty else pd.DataFrame()
        if {"full", "sampled"}.issubset(modes.index):
            full = modes.loc["full"]
            sampled = modes.loc["sampled"]
            notes.append(
                f"{label}核心实验中，full/sampled 平均指标为 "
                f"{full[metric]:.4f}/{sampled[metric]:.4f}，平均总训练时间为 "
                f"{full['seconds']:.3f}/{sampled['seconds']:.3f} 秒；"
                "在 Cora 这类小图上，采样开销没有转化为速度优势。"
            )
    if not core_graphs.empty:
        classification = core_graphs[core_graphs["task_type"] == "classification"]
        if not classification.empty:
            best = classification.loc[classification["test_metric"].idxmax()]
            notes.append(
                f"图分类当前最佳池化为 {best['pooling']}（{best['model']}），"
                f"测试准确率 {best['test_metric']:.4f}。"
            )
    if not sweeps.empty:
        best = sweeps.loc[sweeps["test_acc"].idxmax()]
        notes.append(
            f"受控参数研究最佳为学习率 {best['lr']:g}、{int(best['layers'])} 层，"
            f"测试准确率 {best['test_acc']:.4f}。"
        )
    if not notes:
        notes.append("尚无可汇总实验；先运行 `python scripts/run_benchmarks.py --suite smoke`。")
    return notes


def mode_summary(nodes: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, frame, metric in (
        ("节点分类", nodes, "test_acc"),
        ("链路预测", links, "test_auc"),
    ):
        core = frame[frame["run"] == "core"] if not frame.empty else frame
        if core.empty:
            continue
        for mode, group in core.groupby("mode"):
            rows.append(
                {
                    "task": label,
                    "mode": mode,
                    "mean_metric": group[metric].mean(),
                    "mean_seconds": group["seconds"].mean(),
                    "mean_peak_MB": group["peak_MB"].mean(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Markdown tables from JSON results")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    results = load_results()
    nodes = node_frame(results)
    links = link_frame(results)
    graphs = graph_frame(results)
    knowledge = kg_frame(results)
    modes = mode_summary(nodes, links)
    sweeps = nodes[nodes["run"] == "sweep"] if not nodes.empty else nodes
    has_smoke_results = any("benchmark-smoke" in item.get("_path", "") for item in results)
    lines = [
        "# 实验结果汇总",
        "",
        "> 本文件由 `scripts/generate_report.py` 从各任务的 JSON 结果自动生成。",
        *(
            [
                "",
                "> 当前包含 smoke 流程验证结果；其 epoch 很少，不能作为正式模型优劣结论。",
            ]
            if has_smoke_results
            else []
        ),
        "",
        "## 主要观察",
        "",
        *[f"- {note}" for note in observations(nodes, links, graphs)],
        "",
        "## 全图与采样模式汇总（核心实验均值）",
        "",
        markdown_table(modes),
        "",
        "## 任务一：节点分类",
        "",
        markdown_table(nodes),
        "",
        "## 学习率与层数研究",
        "",
        markdown_table(sweeps[["lr", "layers", "test_acc", "seconds", "peak_MB"]] if not sweeps.empty else sweeps),
        "",
        "## 任务二：链路预测",
        "",
        markdown_table(links),
        "",
        "## 任务三：图分类/回归",
        "",
        markdown_table(graphs),
        "",
        "## 任务四：知识图谱补全",
        "",
        markdown_table(knowledge),
        "",
        "## 解释口径",
        "",
        "- 速度仅在相同机器、相同数据与相同 epoch 设置下比较。",
        "- ZINC 是图回归数据集，因此报告 MAE（越低越好），而不是分类准确率。",
        "- 知识图谱结果只有在 `eval_triples` 等于完整测试集大小时才是完整 filtered 指标。",
        "- 多次随机种子实验应报告均值与标准差；单次结果只用于流程验证。",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
