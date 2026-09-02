import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def read_training_log(log_path):
    pattern = re.compile(
        r"Epoch (\d+) \| "
        r"train_loss=([\d.]+) \| "
        r"train_acc=([\d.]+) \| "
        r"val_loss=([\d.]+) \| "
        r"val_acc=([\d.]+)"
    )

    records = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                records.append(
                    {
                        "epoch": int(match.group(1)),
                        "train_loss": float(match.group(2)),
                        "train_acc": float(match.group(3)),
                        "val_loss": float(match.group(4)),
                        "val_acc": float(match.group(5)),
                    }
                )

    if not records:
        raise ValueError(f"No epoch records found in {log_path}")

    return pd.DataFrame(records)


def add_legends(ax, experiment_names, colors):
    experiment_handles = [
        Line2D(
            [0],
            [0],
            color=colors[i % len(colors)],
            linewidth=2,
            label=experiment_name,
        )
        for i, experiment_name in enumerate(experiment_names)
    ]

    experiment_legend = ax.legend(
        handles=experiment_handles,
        title="Experiments",
        loc="upper left",
    )

    ax.add_artist(experiment_legend)

    style_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            marker="o",
            label="Train",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            marker="s",
            label="Validation",
        ),
    ]

    ax.legend(
        handles=style_handles,
        title="Split",
        loc="upper right",
    )


def plot_experiments(
    experiment_names,
    save_name,
    show=True,
    overwrite=False,
    xlim=None,
    xticks=None,
    loss_ylim=None,
    loss_yticks=None,
    accuracy_ylim=None,
    accuracy_yticks=None,
    linewidth=2.0,
    alpha=0.85,
    markersize=6,
    legend_fields=None,
):
    """
    Plot train/validation loss and accuracy curves for multiple experiments.

    Metrics are loaded from experiment log files and truncated to the epochs
    shared by all selected experiments. Figures are saved under
    `results/figures/{save_name}/`.

    Supports custom axis ranges/ticks, line styles, marker sizes, concise
    legend labels, and optional overwriting of existing figures.
    """

    experiments_path = PROJECT_ROOT / "results" / "experiments.csv"
    log_dir = PROJECT_ROOT / "outputs" / "logs"
    figure_dir = PROJECT_ROOT / "results" / "figures" / save_name

    figure_dir.mkdir(parents=True, exist_ok=True)

    experiments = pd.read_csv(experiments_path)

    missing_experiments = [
        name
        for name in experiment_names
        if name not in experiments["experiment_name"].values
    ]

    if missing_experiments:
        raise ValueError(
            f"Experiments not found: {missing_experiments}"
        )

    selected_experiments = (
        experiments[
            experiments["experiment_name"].isin(experiment_names)
        ]
        .copy()
        .set_index("experiment_name")
        .loc[experiment_names]
        .reset_index()
    )

    histories = {}

    for experiment_name in experiment_names:
        log_path = log_dir / f"{experiment_name}.log"

        if not log_path.exists():
            raise FileNotFoundError(
                f"Log file not found: {log_path}"
            )

        histories[experiment_name] = read_training_log(log_path)

    # Limit comparison to epochs shared by all experiments.
    common_max_epoch = min(
        int(history["epoch"].max())
        for history in histories.values()
    )

    for experiment_name in histories:
        histories[experiment_name] = (
            histories[experiment_name][
                histories[experiment_name]["epoch"] <= common_max_epoch
            ]
            .copy()
            .reset_index(drop=True)
        )

    print(f"Comparison limited to epochs 1-{common_max_epoch}")

    # Automatically generate concise legend labels.
    display_names = {
        "seed": "seed",
        "image_size": "size",
        "model": "model",
        "batch_size": "batch",
        "epochs": "epochs",
        "optimizer": "optimizer",
        "lr": "lr",
        "weight_decay": "wd",
        "augmentation": "aug",
        "normalization": "norm",
    }

    if legend_fields is None:
        legend_fields = []

        if "experiment_group" in selected_experiments.columns:
            groups = (
                selected_experiments["experiment_group"]
                .dropna()
                .unique()
            )

            if len(groups) == 1:
                group = groups[0]

                if group in selected_experiments.columns:
                    legend_fields = [group]

        if not legend_fields:
            candidate_fields = [
                "model",
                "seed",
                "image_size",
                "batch_size",
                "epochs",
                "optimizer",
                "lr",
                "weight_decay",
                "augmentation",
                "normalization",
            ]

            legend_fields = [
                field
                for field in candidate_fields
                if field in selected_experiments.columns
                and selected_experiments[field].nunique(
                    dropna=False
                ) > 1
            ]

    legend_labels = {}

    for _, row in selected_experiments.iterrows():
        experiment_name = row["experiment_name"]

        if legend_fields:
            parts = []

            for field in legend_fields:
                value = row[field]
                label = display_names.get(field, field)

                if pd.notna(value):
                    parts.append(f"{label}={value}")

            legend_labels[experiment_name] = ", ".join(parts)
        else:
            legend_labels[experiment_name] = experiment_name

    # X-axis settings.
    if xticks is None:
        current_xticks = list(range(1, common_max_epoch + 1))
    else:
        current_xticks = xticks

    if xlim is None:
        current_xlim = (0.8, common_max_epoch + 0.2)
    else:
        current_xlim = xlim

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plots = {
        "loss": {
            "train_metric": "train_loss",
            "val_metric": "val_loss",
            "ylabel": "Loss",
            "train_title": "Train Loss",
            "val_title": "Validation Loss",
            "ylim": loss_ylim,
            "yticks": loss_yticks,
        },
        "accuracy": {
            "train_metric": "train_acc",
            "val_metric": "val_acc",
            "ylabel": "Accuracy",
            "train_title": "Train Accuracy",
            "val_title": "Validation Accuracy",
            "ylim": accuracy_ylim,
            "yticks": accuracy_yticks,
        },
    }

    for plot_name, config in plots.items():
        save_path = figure_dir / f"{save_name}_{plot_name}.png"

        if save_path.exists() and not overwrite:
            print(f"Skipped: {save_path}")
            continue

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14, 5),
            sharex=True,
        )

        train_ax, val_ax = axes

        for i, (experiment_name, history) in enumerate(
            histories.items()
        ):
            color = colors[i % len(colors)]
            legend_label = legend_labels[experiment_name]

            train_ax.plot(
                history["epoch"],
                history[config["train_metric"]],
                color=color,
                marker="o",
                markersize=markersize,
                linewidth=linewidth,
                alpha=alpha,
                label=legend_label,
            )

            val_ax.plot(
                history["epoch"],
                history[config["val_metric"]],
                color=color,
                marker="o",
                markersize=markersize,
                linewidth=linewidth,
                alpha=alpha,
                label=legend_label,
            )

        train_ax.set_title(config["train_title"])
        val_ax.set_title(config["val_title"])

        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.set_ylabel(config["ylabel"])
            ax.set_xlim(*current_xlim)
            ax.set_xticks(current_xticks)
            ax.grid(alpha=0.3)

            if config["ylim"] is not None:
                ax.set_ylim(*config["ylim"])

            if config["yticks"] is not None:
                ax.set_yticks(config["yticks"])

        handles, labels = train_ax.get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            title="Experiments",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=min(len(experiment_names), 5),
        )

        fig.suptitle(save_name, y=1.16)
        fig.tight_layout()

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

if __name__ == "__main__":
    print(Path(__file__))
    print('PROJECT_ROOT:',PROJECT_ROOT)