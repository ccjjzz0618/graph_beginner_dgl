from __future__ import annotations

import argparse
import json
import os
import platform

os.environ.setdefault("DGLBACKEND", "pytorch")

import dgl
import numpy as np
import sklearn
import torch
from dgl.nn import GraphConv


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify PyTorch, DGL and CUDA")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    if not cuda_available and not args.allow_cpu:
        raise SystemExit(
            "CUDA is unavailable in PyTorch. Install requirements-gpu-cu121.txt "
            "or rerun with --allow-cpu for a CPU-only environment."
        )
    device = torch.device("cuda" if cuda_available else "cpu")
    graph = dgl.add_self_loop(dgl.graph(([0, 1], [1, 0]))).to(device)
    features = torch.randn(graph.num_nodes(), 4, device=device)
    output = GraphConv(4, 2, allow_zero_in_degree=True).to(device)(graph, features)
    output.sum().backward()
    details = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "dgl": dgl.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda_available": cuda_available,
        "device": torch.cuda.get_device_name(0) if cuda_available else "CPU",
        "dgl_graphconv_device": str(output.device),
        "status": "ok",
    }
    print(json.dumps(details, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
