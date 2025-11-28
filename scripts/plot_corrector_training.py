#!/usr/bin/env python3
"""Plot corrector training curves from saved metrics files."""

import argparse
import glob
import json
import os
from typing import List

import matplotlib.pyplot as plt


def load_metrics(paths: List[str]):
    data = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        data.append((path, payload))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics_glob",
        type=str,
        required=True,
        help='Glob pattern to locate "*_metrics.json" files (e.g., "results/corrector_train/*_metrics.json").',
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the plot as a PNG. If omitted, the plot is shown interactively.",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.metrics_glob))
    if not files:
        print(f"No metric files matched pattern: {args.metrics_glob}")
        return

    runs = load_metrics(files)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for path, payload in runs:
        hist = payload.get("history", {})
        meta = payload.get("meta", {})
        label = f"{meta.get('corrector_type', 'unknown')} (lr={meta.get('lr', '?')}, lambda={meta.get('lambda_reg', '?')})"
        epochs = hist.get("epoch", [])
        axes[0].plot(epochs, hist.get("train_loss", []), label=label)
        axes[1].plot(epochs, hist.get("train_mse", []), label=label)
        axes[2].plot(epochs, hist.get("train_delta_norm", []), label=label)

    axes[0].set_ylabel("Train loss")
    axes[1].set_ylabel("Train MSE")
    axes[2].set_ylabel("Mean |delta_a|")
    axes[2].set_xlabel("Epoch")
    axes[0].set_title("Corrector training curves")
    for ax in axes:
        ax.legend()
        ax.grid(True)
    plt.tight_layout()

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        plt.savefig(args.output, dpi=200)
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
