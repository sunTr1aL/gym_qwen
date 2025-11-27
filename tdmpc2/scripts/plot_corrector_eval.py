import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt


def load_results(csv_path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["mean_return"] = float(row.get("mean_return", 0.0))
            row["exec_horizon"] = int(row.get("exec_horizon", 0))
            records.append(row)
    return records


def plot_horizon(records: List[Dict], output_path: str) -> None:
    if not records:
        print("No records to plot.")
        return
    grouped: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        corr = rec.get("corrector_type") or "none"
        grouped[corr][rec["exec_horizon"]].append(rec["mean_return"])
    horizons = sorted({rec["exec_horizon"] for rec in records})
    plt.figure(figsize=(8, 5))
    for corr_type, values in grouped.items():
        ys = [sum(values.get(h, [0])) / max(len(values.get(h, [])), 1) for h in horizons]
        plt.plot(horizons, ys, marker="o", label=corr_type)
    plt.xlabel("Execution horizon (steps)")
    plt.ylabel("Mean return (avg across model sizes)")
    plt.title("Performance vs execution horizon")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved horizon plot to {output_path}")


def plot_model_sizes(records: List[Dict], output_path: str) -> None:
    if not records:
        print("No records to plot.")
        return
    focus_h = 3
    filtered = [r for r in records if r["exec_horizon"] == focus_h]
    if not filtered:
        print("No horizon-3 evaluations to plot for model sizes.")
        return
    variants = ["open_loop_3", "corrected_two_tower_3", "corrected_temporal_3", "baseline_replan"]
    model_sizes = sorted({r["model_size"] for r in filtered})
    width = 0.18
    positions = range(len(model_sizes))
    plt.figure(figsize=(10, 5))
    for idx, var in enumerate(variants):
        vals = []
        for size in model_sizes:
            match = next((r for r in filtered if r["model_size"] == size and r["variant"] == var), None)
            vals.append(match.get("mean_return") if match else 0.0)
        offsets = [p + (idx - len(variants) / 2) * width for p in positions]
        plt.bar(offsets, vals, width=width, label=var)
    plt.xticks(list(positions), model_sizes)
    plt.xlabel("Model size")
    plt.ylabel("Mean return (horizon=3)")
    plt.title("Performance vs model size")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved model size plot to {output_path}")


def plot_improvement(records: List[Dict], output_path: str) -> None:
    improvements: List[tuple] = []
    for horizon in (2, 3):
        base = {(r["model_size"], horizon): r for r in records if r["exec_horizon"] == horizon and r["corrector_type"] in (None, "", "none")}
        for corr_type in ("two_tower", "temporal"):
            for rec in records:
                if rec["exec_horizon"] != horizon or rec.get("corrector_type") != corr_type:
                    continue
                key = (rec["model_size"], horizon)
                base_return = base.get(key, {}).get("mean_return", 0.0)
                improvements.append((corr_type, rec["model_size"], horizon, rec["mean_return"] - base_return))
    if not improvements:
        print("No improvement data to plot.")
        return
    plt.figure(figsize=(10, 5))
    labels = [f"{size}-h{h}" for _, size, h, _ in improvements]
    values = [imp for *_, imp in improvements]
    colors = ["tab:blue" if corr == "two_tower" else "tab:orange" for corr, *_ in improvements]
    plt.bar(range(len(values)), values, color=colors)
    plt.xticks(range(len(values)), labels, rotation=45)
    plt.ylabel("Return gain vs no corrector")
    plt.title("Corrector improvement over open-loop")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved improvement plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_csv", type=str, required=True, help="Aggregated CSV from eval_corrector.py")
    parser.add_argument("--output_dir", type=str, default="results/corrector_eval/plots")
    args = parser.parse_args()

    records = load_results(args.results_csv)
    if not records:
        print(f"No records loaded from {args.results_csv}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    plot_horizon(records, os.path.join(args.output_dir, "performance_vs_horizon.png"))
    plot_model_sizes(records, os.path.join(args.output_dir, "performance_vs_model_size.png"))
    plot_improvement(records, os.path.join(args.output_dir, "corrector_improvement.png"))


if __name__ == "__main__":
    main()
