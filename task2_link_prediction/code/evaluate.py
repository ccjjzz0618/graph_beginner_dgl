from __future__ import annotations

import argparse

from train import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved link-prediction model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.device)


if __name__ == "__main__":
    main()

