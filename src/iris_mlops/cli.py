"""
Command-line interface for training.

WHY THIS MODULE EXISTS:
    A CLI gives a stable, scriptable interface. CI systems, cron jobs,
    and Prefect flows all invoke this. The CLI is the entry point;
    everything else is a library.
"""

import argparse
import logging
from iris_mlops.train import train

def main() -> None:
    # argparse: standard library for CLI parsing
    parser = argparse.ArgumentParser(description="Train Iris classifier")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Clear existing handlers to prevent duplicate logs in some environments
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    acc = train(seed=args.seed, C=args.C, max_iter=args.max_iter)

    # print for the human at the terminal; logger for the audit trail
    print(f"Final accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
