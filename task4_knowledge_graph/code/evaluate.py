from __future__ import annotations

import argparse

from train import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Filtered evaluation of a saved KGE model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--eval-limit", type=int, default=0, help="0 evaluates the complete split"
    )
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.device, args.split, args.eval_limit)


if __name__ == "__main__":
    main()

