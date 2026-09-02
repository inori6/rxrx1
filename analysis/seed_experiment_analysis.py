"""Analyze reproducibility across random seeds from an experiments CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

REQUIRED_COLUMNS = (
    "experiment_group",
    "seed",
    "final_train_acc",
    "final_train_loss",
    "final_val_acc",
    "final_val_loss",
    "best_train_acc",
    "best_train_loss",
    "best_val_acc",
    "best_val_loss",
    "best_epoch",
)
NUMERIC_COLUMNS = tuple(column for column in REQUIRED_COLUMNS if column != "experiment_group")
GAP_COLUMNS = (
    "final_acc_train_val_gap",
    "best_acc_train_val_gap",
    "final_loss_val_train_gap",
    "best_loss_val_train_gap",
)
SUMMARY_METRICS = NUMERIC_COLUMNS[1:] + GAP_COLUMNS


def load_experiments(csv_path: Path) -> pd.DataFrame:
    """Load the experiment table, validate its schema, and coerce numeric fields."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Experiments CSV not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    frame = frame.copy()
    frame["experiment_group"] = frame["experiment_group"].astype("string")
    frame[list(NUMERIC_COLUMNS)] = frame[list(NUMERIC_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid = frame[list(NUMERIC_COLUMNS)].isna().sum()
    invalid = invalid.loc[invalid > 0]
    if not invalid.empty:
        details = ", ".join(f"{column}={count}" for column, count in invalid.items())
        raise ValueError(f"Missing or non-numeric values in required columns: {details}")
    return frame


def filter_seed_experiments(frame: pd.DataFrame, group_pattern: str = "seed") -> pd.DataFrame:
    """Keep experiment groups whose names match a case-insensitive regex."""
    mask = frame["experiment_group"].str.contains(
        group_pattern, case=False, regex=True, na=False
    )
    filtered = frame.loc[mask].copy()
    if filtered.empty:
        raise ValueError(
            f"No rows matched experiment_group pattern {group_pattern!r}. "
            "Pass --group-pattern with the naming convention used by your CSV."
        )
    return filtered.sort_values(["experiment_group", "seed"], kind="stable")


def add_train_val_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    """Add gaps where a positive value consistently means validation is worse."""
    enriched = frame.copy()
    enriched["final_acc_train_val_gap"] = (
            enriched["final_train_acc"] - enriched["final_val_acc"]
    )
    enriched["best_acc_train_val_gap"] = (
            enriched["best_train_acc"] - enriched["best_val_acc"]
    )
    enriched["final_loss_val_train_gap"] = (
            enriched["final_val_loss"] - enriched["final_train_loss"]
    )
    enriched["best_loss_val_train_gap"] = (
            enriched["best_val_loss"] - enriched["best_train_loss"]
    )
    return enriched


def summarize_by_group(frame: pd.DataFrame) -> pd.DataFrame:
    """Return long-form group statistics using sample standard deviation (ddof=1)."""
    records: list[dict[str, float | int | str]] = []
    for group, group_frame in frame.groupby("experiment_group", sort=True):
        for metric in SUMMARY_METRICS:
            values = group_frame[metric].dropna()
            count = int(values.size)
            mean = float(values.mean()) if count else np.nan
            std = float(values.std(ddof=1)) if count > 1 else np.nan
            cv = std / abs(mean) if np.isfinite(std) and not np.isclose(mean, 0.0) else np.nan
            minimum = float(values.min()) if count else np.nan
            maximum = float(values.max()) if count else np.nan
            records.append(
                {
                    "experiment_group": group,
                    "metric": metric,
                    "n": count,
                    "mean": mean,
                    "std": std,
                    "cv": cv,
                    "cv_percent": cv * 100 if np.isfinite(cv) else np.nan,
                    "min": minimum,
                    "max": maximum,
                    "range": maximum - minimum if count else np.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def select_median_seed(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the observed seed closest to each group's median best validation accuracy."""
    records = []
    for group, group_frame in frame.groupby("experiment_group", sort=True):
        target = float(group_frame["best_val_acc"].median())
        ranked = group_frame.assign(
            distance_to_group_median=(group_frame["best_val_acc"] - target).abs()
        ).sort_values(["distance_to_group_median", "seed"], kind="stable")
        selected = ranked.iloc[0]
        records.append(
            {
                "experiment_group": group,
                "seed": selected["seed"],
                "best_val_acc": selected["best_val_acc"],
                "group_median_best_val_acc": target,
                "distance_to_group_median": selected["distance_to_group_median"],
            }
        )
    return pd.DataFrame.from_records(records)


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_seed_analysis(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Save three compact diagnostic figures and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = list(frame.groupby("experiment_group", sort=True))
    colors = plt.get_cmap("tab10")
    paths: list[Path] = []

    fig, axes = plt.subplots(
        len(groups), 1, figsize=(9, max(3.2, 2.8 * len(groups))), squeeze=False
    )
    for index, (group, group_frame) in enumerate(groups):
        ax = axes[index, 0]
        ordered = group_frame.sort_values("seed")
        positions = np.arange(len(ordered))
        values = ordered["best_val_acc"]
        mean, std = values.mean(), values.std(ddof=1)
        ax.plot(positions, values, "o-", color=colors(index % 10), label="best val acc")
        ax.axhline(mean, color="black", linestyle="--", linewidth=1, label="mean")
        if np.isfinite(std):
            ax.axhspan(mean - std, mean + std, color="black", alpha=0.08, label="mean ± std")
        ax.set(title=str(group), ylabel="Accuracy", xticks=positions)
        ax.set_xticklabels(ordered["seed"].map(lambda value: f"{value:g}"))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, ncol=3)
    axes[-1, 0].set_xlabel("Seed")
    path = output_dir / "best_val_acc_by_seed.png"
    _save_figure(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for index, (group, group_frame) in enumerate(groups):
        ax.scatter(
            group_frame["best_train_acc"],
            group_frame["best_val_acc"],
            s=55,
            alpha=0.8,
            color=colors(index % 10),
            label=str(group),
        )
    accuracy_values = frame[["best_train_acc", "best_val_acc"]]
    limits = np.array([accuracy_values.min().min(), accuracy_values.max().max()])
    padding = max((limits[1] - limits[0]) * 0.08, 1e-3)
    limits += np.array([-padding, padding])
    ax.plot(limits, limits, "--", color="0.35", linewidth=1, label="train = val")
    ax.set(
        title="Best train vs. validation accuracy",
        xlabel="Best train accuracy",
        ylabel="Best validation accuracy",
        xlim=limits,
        ylim=limits,
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    path = output_dir / "best_train_vs_val_acc.png"
    _save_figure(fig, path)
    paths.append(path)

    fig, axes = plt.subplots(
        len(groups), 1, figsize=(9, max(3.2, 2.8 * len(groups))), squeeze=False
    )
    for index, (group, group_frame) in enumerate(groups):
        ax = axes[index, 0]
        ordered = group_frame.sort_values("seed")
        positions = np.arange(len(ordered))
        ax.plot(positions, ordered["final_acc_train_val_gap"], "o-", label="final acc gap")
        ax.plot(positions, ordered["best_acc_train_val_gap"], "s--", label="best acc gap")
        ax.axhline(0, color="0.35", linewidth=1)
        ax.set(title=str(group), ylabel="Train − val", xticks=positions)
        ax.set_xticklabels(ordered["seed"].map(lambda value: f"{value:g}"))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    axes[-1, 0].set_xlabel("Seed")
    path = output_dir / "accuracy_gaps_by_seed.png"
    _save_figure(fig, path)
    paths.append(path)
    return paths


def run_analysis(
        csv_path: Path,
        output_dir: Path,
        group_pattern: str = "seed",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the analysis and write reusable tables and figures."""
    seed_runs = add_train_val_gaps(
        filter_seed_experiments(load_experiments(csv_path), group_pattern)
    )
    summary = summarize_by_group(seed_runs)
    median_seeds = select_median_seed(seed_runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_runs.to_csv(output_dir / "seed_runs_enriched.csv", index=False)
    summary.to_csv(output_dir / "seed_summary.csv", index=False)
    median_seeds.to_csv(output_dir / "median_seed.csv", index=False)
    plot_seed_analysis(seed_runs, output_dir)
    return seed_runs, summary, median_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/experiments.csv"),
        help="Experiments CSV path relative to the project root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis/seed"),
        help="Output directory relative to the project root.",
    )
    parser.add_argument(
        "--group-pattern",
        default="seed",
        help="Case-insensitive regex matched against experiment_group.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_runs, summary, median_seeds = run_analysis(
        Path(__file__).resolve().parents[1] / args.input, Path(__file__).resolve().parents[1] / args.output_dir,
        args.group_pattern
    )
    key_summary = summary.loc[summary["metric"] == "best_val_acc"]
    print(f"Analyzed {len(seed_runs)} runs across {seed_runs['experiment_group'].nunique()} group(s).")
    print("\nBest validation accuracy summary:")
    print(key_summary.to_string(index=False))
    print("\nMedian-performing seed(s):")
    print(median_seeds.to_string(index=False))
    print(f"\nSaved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
