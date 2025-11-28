import argparse
import os
import re
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def load_results(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _order_model_sizes(sizes: List[str]) -> List[str]:
    def size_key(val: str) -> float:
        match = re.match(r"([0-9]+(?:\.[0-9]+)?)([KkMmGg]?)", val)
        if not match:
            return float("inf")
        num = float(match.group(1))
        suffix = match.group(2).lower()
        if suffix == "k":
            num /= 1000.0
        elif suffix == "g":
            num *= 1000.0
        # Default assumes millions
        return num

    return sorted(sizes, key=size_key)


def plot_horizon(df: pd.DataFrame, output_path: str) -> None:
    if df.empty:
        print("No records to plot.")
        return
    df = df.copy()
    df["corrector_label"] = df["corrector_type"].fillna("none")
    grouped = df.groupby(["corrector_label", "exec_horizon"])["mean_return"].mean().unstack(fill_value=0)
    horizons = grouped.columns.tolist()
    plt.figure(figsize=(8, 5))
    for corr_type, row in grouped.iterrows():
        plt.plot(horizons, row.values, marker="o", label=corr_type)
    plt.xlabel("Execution horizon (steps)")
    plt.ylabel("Mean return (avg across models)")
    plt.title("Performance vs execution horizon")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Corrector")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved horizon plot to {output_path}")


def plot_model_name_size(df: pd.DataFrame, output_path: str, horizon: int = 3) -> None:
    subset = df[df["exec_horizon"] == horizon]
    if subset.empty:
        print(f"No horizon-{horizon} evaluations to plot for model sizes.")
        return
    variants = ["baseline_replan", "open_loop_3", "corrected_two_tower_3", "corrected_temporal_3"]
    model_names = sorted(subset["model_name"].dropna().unique())
    n_rows = len(model_names)
    plt.figure(figsize=(10, 4 * max(n_rows, 1)))
    for idx, model_name in enumerate(model_names):
        ax = plt.subplot(max(n_rows, 1), 1, idx + 1)
        df_name = subset[subset["model_name"] == model_name]
        sizes = _order_model_sizes(df_name["model_size"].dropna().unique().tolist())
        for var in variants:
            series = []
            for size in sizes:
                match = df_name[(df_name["model_size"] == size) & (df_name["variant"] == var)]
                series.append(match["mean_return"].mean() if not match.empty else 0.0)
            ax.plot(sizes, series, marker="o", label=var)
        ax.set_title(f"{model_name} - horizon {horizon}")
        ax.set_xlabel("Model size")
        ax.set_ylabel("Mean return")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved model-name/size plot to {output_path}")


def plot_improvement(df: pd.DataFrame, output_path: str) -> None:
    if df.empty:
        print("No records to plot.")
        return
    df = df.copy()
    df["corrector_label"] = df["corrector_type"].fillna("none")
    base = df[df["corrector_label"].isin(["none", "", None])]
    improvements: List[tuple] = []
    for _, row in df[df["corrector_label"].isin(["two_tower", "temporal"])].iterrows():
        match = base[
            (base["model_id"] == row["model_id"]) & (base["exec_horizon"] == row["exec_horizon"])
        ]
        base_return = match["mean_return"].mean() if not match.empty else 0.0
        improvements.append(
            (
                row["corrector_label"],
                row["model_name"],
                row["model_size"],
                row["exec_horizon"],
                row["mean_return"] - base_return,
            )
        )
    if not improvements:
        print("No improvement data to plot.")
        return
    labels = [f"{mn}-{ms}-h{h}" for _, mn, ms, h, _ in improvements]
    values = [val for *_, val in improvements]
    colors = ["tab:blue" if corr == "two_tower" else "tab:orange" for corr, *_ in improvements]
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(values)), values, color=colors)
    plt.xticks(range(len(values)), labels, rotation=45, ha="right")
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

    df = load_results(args.results_csv)
    if df.empty:
        print(f"No records loaded from {args.results_csv}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    plot_horizon(df, os.path.join(args.output_dir, "performance_vs_horizon.png"))
    plot_model_name_size(df, os.path.join(args.output_dir, "performance_vs_model_name_size.png"))
    plot_improvement(df, os.path.join(args.output_dir, "corrector_improvement.png"))


if __name__ == "__main__":
    main()
