"""Analyze the effect of input image size on model performance and compute cost."""

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
    "image_size",
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

OPTIONAL_COLUMNS = (
    "runtime_seconds",
    "runtime_minutes",
    "runtime_per_epoch_minutes",
)

GAP_COLUMNS = (
    "final_acc_train_val_gap",
    "best_acc_train_val_gap",
    "final_loss_val_train_gap",
    "best_loss_val_train_gap",
)

RECOMMENDED_IMAGE_SIZE = 384
REFERENCE_IMAGE_SIZE = 256


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

    numeric_columns = [
        column
        for column in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS)
        if column != "experiment_group" and column in frame.columns
    ]

    frame[numeric_columns] = frame[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    required_numeric = [
        column for column in REQUIRED_COLUMNS if column != "experiment_group"
    ]

    invalid = frame[required_numeric].isna().sum()
    invalid = invalid.loc[invalid > 0]

    if not invalid.empty:
        details = ", ".join(
            f"{column}={count}"
            for column, count in invalid.items()
        )
        raise ValueError(
            f"Missing or non-numeric values in required columns: {details}"
        )

    return frame


def filter_image_size_experiments(
    frame: pd.DataFrame,
    group_pattern: str = "image_size|size",
) -> pd.DataFrame:
    """Keep experiment groups corresponding to image-size experiments."""
    mask = frame["experiment_group"].str.contains(
        group_pattern,
        case=False,
        regex=True,
        na=False,
    )

    filtered = frame.loc[mask].copy()

    if filtered.empty:
        raise ValueError(
            f"No rows matched experiment_group pattern {group_pattern!r}. "
            "Pass --group-pattern with the naming convention used by your CSV."
        )

    return filtered.sort_values("image_size", kind="stable").reset_index(drop=True)


def add_derived_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add generalization-gap and approximate resolution-cost metrics."""
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

    enriched["pixel_count"] = enriched["image_size"] ** 2
    enriched["relative_pixel_cost"] = (
        enriched["image_size"] / REFERENCE_IMAGE_SIZE
    ) ** 2

    return enriched


def build_image_size_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Build one compact comparison table ordered by input resolution."""
    columns = [
        "image_size",
        "best_val_acc",
        "final_val_acc",
        "best_val_loss",
        "best_acc_train_val_gap",
        "relative_pixel_cost",
    ]

    for column in OPTIONAL_COLUMNS:
        if column in frame.columns:
            columns.append(column)

    summary = frame[columns].copy()
    summary = summary.sort_values("image_size").reset_index(drop=True)

    best_accuracy = summary["best_val_acc"].max()
    summary["best_val_acc_gap_from_best"] = (
        best_accuracy - summary["best_val_acc"]
    )

    if "runtime_per_epoch_minutes" in summary.columns:
        baseline_runtime = summary.loc[
            summary["image_size"] == REFERENCE_IMAGE_SIZE,
            "runtime_per_epoch_minutes",
        ]

        if not baseline_runtime.empty:
            summary["runtime_relative_to_256"] = (
                summary["runtime_per_epoch_minutes"]
                / baseline_runtime.iloc[0]
            )

    summary["recommended"] = (
        summary["image_size"] == RECOMMENDED_IMAGE_SIZE
    )

    return summary


def select_recommended_size(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the predefined resolution selected for subsequent experiments."""
    selected = frame.loc[
        frame["image_size"] == RECOMMENDED_IMAGE_SIZE
    ].copy()

    if selected.empty:
        raise ValueError(
            f"Recommended image size {RECOMMENDED_IMAGE_SIZE} "
            "was not found in the image-size experiments."
        )

    columns = [
        "image_size",
        "best_val_acc",
        "final_val_acc",
        "best_val_loss",
        "best_acc_train_val_gap",
        "relative_pixel_cost",
    ]

    for column in OPTIONAL_COLUMNS:
        if column in selected.columns:
            columns.append(column)

    return selected[columns].reset_index(drop=True)


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_image_size_analysis(
    frame: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Save image-size performance and efficiency diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = frame.sort_values("image_size")

    paths: list[Path] = []

    # Validation accuracy
    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.plot(
        ordered["image_size"],
        ordered["best_val_acc"],
        "o-",
        label="Best validation accuracy",
    )
    ax.plot(
        ordered["image_size"],
        ordered["final_val_acc"],
        "s--",
        label="Final validation accuracy",
    )

    selected = ordered.loc[
        ordered["image_size"] == RECOMMENDED_IMAGE_SIZE
    ]

    if not selected.empty:
        ax.scatter(
            selected["image_size"],
            selected["best_val_acc"],
            s=120,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label=f"Selected: {RECOMMENDED_IMAGE_SIZE}",
        )

    ax.set(
        title="Validation performance by image size",
        xlabel="Image size",
        ylabel="Accuracy",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    path = output_dir / "validation_accuracy_by_image_size.png"
    _save_figure(fig, path)
    paths.append(path)

    # Train-validation gap
    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.plot(
        ordered["image_size"],
        ordered["best_acc_train_val_gap"],
        "o-",
        label="Best train − val accuracy gap",
    )
    ax.plot(
        ordered["image_size"],
        ordered["final_acc_train_val_gap"],
        "s--",
        label="Final train − val accuracy gap",
    )

    ax.axhline(0, linewidth=1)

    ax.set(
        title="Generalization gap by image size",
        xlabel="Image size",
        ylabel="Train − validation accuracy",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    path = output_dir / "accuracy_gap_by_image_size.png"
    _save_figure(fig, path)
    paths.append(path)

    # Runtime
    if "runtime_per_epoch_minutes" in ordered.columns:
        valid_runtime = ordered.dropna(
            subset=["runtime_per_epoch_minutes"]
        )

        if not valid_runtime.empty:
            fig, ax = plt.subplots(figsize=(7.5, 5))

            ax.plot(
                valid_runtime["image_size"],
                valid_runtime["runtime_per_epoch_minutes"],
                "o-",
            )

            ax.set(
                title="Runtime per epoch by image size",
                xlabel="Image size",
                ylabel="Runtime per epoch (minutes)",
            )
            ax.grid(alpha=0.25)

            path = output_dir / "runtime_by_image_size.png"
            _save_figure(fig, path)
            paths.append(path)

    # Accuracy vs theoretical pixel cost
    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.plot(
        ordered["relative_pixel_cost"],
        ordered["best_val_acc"],
        "o-",
    )

    for _, row in ordered.iterrows():
        ax.annotate(
            f"{int(row['image_size'])}",
            (
                row["relative_pixel_cost"],
                row["best_val_acc"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set(
        title="Validation accuracy vs. approximate input cost",
        xlabel=f"Relative pixel count ({REFERENCE_IMAGE_SIZE} = 1×)",
        ylabel="Best validation accuracy",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(alpha=0.25)

    path = output_dir / "accuracy_vs_resolution_cost.png"
    _save_figure(fig, path)
    paths.append(path)

    return paths


def build_conclusion() -> str:
    """Return the experiment-level decision used by subsequent experiments."""
    return (
        f"Image size {RECOMMENDED_IMAGE_SIZE} is selected for subsequent experiments. "
        "The decision is based on the overall trade-off between validation performance, "
        "computational cost, and suitability for the RxRx1 target task rather than on "
        "validation accuracy alone. Increasing resolution preserves more cellular "
        "morphological information, but computational cost grows approximately with "
        "the number of input pixels. Larger resolutions therefore substantially "
        "increase training cost, while their potential validation improvement does "
        "not justify the additional compute at the current experimental stage. "
        f"{RECOMMENDED_IMAGE_SIZE} provides a practical intermediate resolution that "
        "retains more spatial information than lower-resolution inputs while keeping "
        "the cost manageable for repeated augmentation, normalization, model, and "
        "hyperparameter experiments."
    )


def run_analysis(
    csv_path: Path,
    output_dir: Path,
    group_pattern: str = "image_size|size",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the complete image-size analysis and save reusable outputs."""
    image_size_runs = add_derived_metrics(
        filter_image_size_experiments(
            load_experiments(csv_path),
            group_pattern,
        )
    )

    summary = build_image_size_summary(image_size_runs)
    recommended = select_recommended_size(image_size_runs)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_size_runs.to_csv(
        output_dir / "image_size_runs_enriched.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "image_size_summary.csv",
        index=False,
    )
    recommended.to_csv(
        output_dir / "recommended_image_size.csv",
        index=False,
    )

    plot_image_size_analysis(image_size_runs, output_dir)

    conclusion = build_conclusion()
    (output_dir / "conclusion.txt").write_text(
        conclusion,
        encoding="utf-8",
    )

    return image_size_runs, summary, recommended


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
        default=Path("results/analysis/image_size"),
        help="Output directory relative to the project root.",
    )
    parser.add_argument(
        "--group-pattern",
        default="image_size|size",
        help="Case-insensitive regex matched against experiment_group.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]

    image_size_runs, summary, recommended = run_analysis(
        project_root / args.input,
        project_root / args.output_dir,
        args.group_pattern,
    )

    print(
        f"Analyzed {len(image_size_runs)} "
        "image-size experiment(s)."
    )

    print("\nImage-size comparison:")
    print(summary.to_string(index=False))

    print("\nSelected image size:")
    print(recommended.to_string(index=False))

    print("\nConclusion:")
    print(build_conclusion())

    print(f"\nSaved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()