from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Run:
    task: str
    name: str
    arguments: tuple[str, ...]

    @property
    def script(self) -> Path:
        scripts = {
            "task1": PROJECT_ROOT / "task1_node_classification" / "code" / "train.py",
            "task2": PROJECT_ROOT / "task2_link_prediction" / "code" / "train.py",
            "task3": PROJECT_ROOT / "task3_graph_classification" / "code" / "train.py",
            "task4": PROJECT_ROOT / "task4_knowledge_graph" / "code" / "train.py",
        }
        return scripts[self.task]

    @property
    def result_path(self) -> Path:
        directories = {
            "task1": "task1_node_classification",
            "task2": "task2_link_prediction",
            "task3": "task3_graph_classification",
            "task4": "task4_knowledge_graph",
        }
        return PROJECT_ROOT / directories[self.task] / "results" / f"benchmark-{self.name}.json"


def smoke_runs() -> list[Run]:
    shared_node_data = str(PROJECT_ROOT / "task1_node_classification" / "data")
    return [
        Run("task1", "smoke-cora-gcn-full", ("--dataset", "cora", "--model", "gcn", "--mode", "full", "--epochs", "2", "--patience", "0")),
        Run("task1", "smoke-cora-gcn-sampled", ("--dataset", "cora", "--model", "gcn", "--mode", "sampled", "--epochs", "2", "--patience", "0")),
        Run("task1", "smoke-citeseer-gcn-full", ("--dataset", "citeseer", "--model", "gcn", "--mode", "full", "--epochs", "1", "--patience", "0")),
        Run("task1", "smoke-flickr-sage-sampled", ("--dataset", "flickr", "--model", "graphsage", "--mode", "sampled", "--epochs", "1", "--batch-size", "2048", "--patience", "0")),
        Run("task2", "smoke-cora-sage-full", ("--dataset", "cora", "--model", "graphsage", "--mode", "full", "--epochs", "2", "--patience", "0")),
        Run("task2", "smoke-cora-sage-sampled", ("--dataset", "cora", "--model", "graphsage", "--mode", "sampled", "--epochs", "2", "--patience", "0")),
        Run("task2", "smoke-citeseer-gcn-full", ("--dataset", "citeseer", "--model", "gcn", "--mode", "full", "--epochs", "1", "--patience", "0", "--data-dir", shared_node_data)),
        Run("task2", "smoke-flickr-sage-sampled", ("--dataset", "flickr", "--model", "graphsage", "--mode", "sampled", "--epochs", "1", "--batch-size", "4096", "--patience", "0", "--data-dir", shared_node_data)),
        Run("task3", "smoke-mutag-gin-avg", ("--dataset", "mutag", "--model", "gin", "--pooling", "avg", "--mode", "mini", "--epochs", "2", "--patience", "0")),
        Run("task3", "smoke-zinc-gcn-avg", ("--dataset", "zinc", "--model", "gcn", "--pooling", "avg", "--mode", "mini", "--epochs", "1", "--batch-size", "256", "--eval-batch-size", "512", "--patience", "0")),
        Run("task4", "smoke-fb15k237-transe", ("--dataset", "fb15k237", "--model", "transe", "--embedding-dim", "50", "--negative-samples", "4", "--epochs", "1", "--eval-every", "1", "--eval-limit", "20", "--patience", "0")),
        Run("task4", "smoke-fb15k237-rotate", ("--dataset", "fb15k237", "--model", "rotate", "--embedding-dim", "20", "--negative-samples", "1", "--batch-size", "8192", "--epochs", "1", "--eval-every", "1", "--eval-batch-size", "5", "--eval-limit", "5", "--patience", "0")),
        Run("task4", "smoke-fb15k237-conve", ("--dataset", "fb15k237", "--model", "conve", "--embedding-dim", "20", "--embedding-height", "5", "--conve-channels", "4", "--negative-samples", "1", "--batch-size", "8192", "--epochs", "1", "--eval-every", "1", "--eval-batch-size", "5", "--eval-limit", "5", "--patience", "0")),
    ]


def core_runs() -> list[Run]:
    runs: list[Run] = []
    models = ("gcn", "gat", "graphsage", "gin")
    for task in ("task1", "task2"):
        for model, mode in itertools.product(models, ("full", "sampled")):
            runs.append(
                Run(
                    task,
                    f"cora-{model}-{mode}",
                    ("--dataset", "cora", "--model", model, "--mode", mode, "--epochs", "100"),
                )
            )
    for model, pooling in itertools.product(models, ("avg", "max", "min")):
        runs.append(
            Run(
                "task3",
                f"mutag-{model}-{pooling}-mini",
                ("--dataset", "mutag", "--model", model, "--pooling", pooling, "--mode", "mini", "--epochs", "100"),
            )
        )
    for model in ("transe", "rotate", "conve"):
        runs.append(
            Run(
                "task4",
                f"fb15k237-{model}",
                ("--dataset", "fb15k237", "--model", model, "--epochs", "50", "--eval-limit", "2000"),
            )
        )
    return runs


def sweep_runs() -> list[Run]:
    """Controlled learning-rate/layer study required by the assignment."""
    runs: list[Run] = []
    for learning_rate, layers in itertools.product(
        ("0.001", "0.01", "0.05"), ("2", "3", "4")
    ):
        runs.append(
            Run(
                "task1",
                f"sweep-cora-gcn-lr{learning_rate}-layers{layers}",
                (
                    "--dataset",
                    "cora",
                    "--model",
                    "gcn",
                    "--mode",
                    "full",
                    "--learning-rate",
                    learning_rate,
                    "--layers",
                    layers,
                    "--fanout",
                    ",".join(["10"] * int(layers)),
                ),
            )
        )
    return runs


def full_runs() -> list[Run]:
    runs: list[Run] = []
    models = ("gcn", "gat", "graphsage", "gin")
    for dataset, model, mode in itertools.product(
        ("cora", "citeseer", "flickr"), models, ("full", "sampled")
    ):
        runs.append(
            Run(
                "task1",
                f"{dataset}-{model}-{mode}",
                ("--dataset", dataset, "--model", model, "--mode", mode),
            )
        )
        runs.append(
            Run(
                "task2",
                f"{dataset}-{model}-{mode}",
                ("--dataset", dataset, "--model", model, "--mode", mode),
            )
        )
    for dataset, model, pooling, mode in itertools.product(
        ("mutag", "proteins", "enzymes", "zinc"),
        models,
        ("avg", "max", "min"),
        ("full", "mini"),
    ):
        runs.append(
            Run(
                "task3",
                f"{dataset}-{model}-{pooling}-{mode}",
                ("--dataset", dataset, "--model", model, "--pooling", pooling, "--mode", mode),
            )
        )
    for model in ("transe", "rotate", "conve"):
        runs.append(
            Run(
                "task4",
                f"fb15k237-{model}-full-eval",
                ("--dataset", "fb15k237", "--model", model, "--eval-limit", "0"),
            )
        )
    runs.extend(sweep_runs())
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible experiment matrices")
    parser.add_argument(
        "--suite", choices=("smoke", "core", "sweep", "full"), default="smoke"
    )
    parser.add_argument("--task", choices=("all", "task1", "task2", "task3", "task4"), default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    runs = {
        "smoke": smoke_runs,
        "core": core_runs,
        "sweep": sweep_runs,
        "full": full_runs,
    }[args.suite]()
    if args.task != "all":
        runs = [run for run in runs if run.task == args.task]

    failures: list[str] = []
    for index, run in enumerate(runs, start=1):
        command = [
            sys.executable,
            str(run.script),
            *run.arguments,
            "--device",
            args.device,
            "--output",
            str(run.result_path),
        ]
        print(f"[{index}/{len(runs)}] {run.name}")
        print(subprocess.list2cmdline(command))
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            failures.append(run.name)
            if not args.continue_on_error:
                raise SystemExit(completed.returncode)
    if failures:
        raise SystemExit(f"Failed runs: {', '.join(failures)}")


if __name__ == "__main__":
    main()
