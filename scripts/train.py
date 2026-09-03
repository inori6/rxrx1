from pathlib import Path
import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from rxrx1.data.dataset import RxRxDataset
from rxrx1.data.manifest import create_label_to_index, read_manifest
from rxrx1.data.transforms import prepare_transforms
from rxrx1.data.normalization import build_normalizer
from rxrx1.models.efficientnet import build_efficientnet
from rxrx1.training.criterion import build_criterion
from rxrx1.training.optimizers import build_optimizer
from rxrx1.training.trainer import fit_model
from rxrx1.training.validation import filter_and_validate_val_manifest
from rxrx1.utils.paths import get_image_root
from rxrx1.utils.seed import set_seed
from rxrx1.utils.logger import (
    setup_logger,
    log_label_revised,
    log_training_started,
    log_training_finished,
    log_training_failed,
)
from rxrx1.utils.tracking import (
    setup_wandb,
    update_wandb_git_info,
    finish_wandb,
    fail_wandb,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_normalizer_by_scope(normalizer):
    """Route a normalizer to Dataset or to the loader-batch training hook."""

    if normalizer is None:
        return None, None

    apply_to = getattr(normalizer, "apply_to", "image")
    if apply_to == "image":
        return normalizer, None
    if apply_to == "batch":
        return None, normalizer
    raise ValueError(f"Unknown normalizer apply_to scope: {apply_to!r}.")


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])

    project_root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger_name = config["experiment"]["name"]
    log_file = project_root / config["logging"]["log_dir"] / f"{logger_name}.log"
    logger = setup_logger(logger_name, log_file, config["logging"]["level"])

    run = setup_wandb(config, project_root)
    update_wandb_git_info(
        run,
        os.getenv("RXRX1_GIT_COMMIT"),
        os.getenv("RXRX1_GIT_REF"),
    )

    try:
        train_manifest = read_manifest(project_root / config["data"]["train_manifest"])
        val_manifest = read_manifest(project_root / config["data"]["val_manifest"])

        val_manifest, train_labels, original_val_labels, val_labels = (
            filter_and_validate_val_manifest(train_manifest, val_manifest)
        )
        log_label_revised(logger, train_labels, original_val_labels, val_labels)

        label_to_index = create_label_to_index(train_manifest)
        image_root = get_image_root("train")

        train_transform, val_transform = prepare_transforms(config)
        train_normalizer = build_normalizer(
            config,
            split="train",
            project_root=project_root,
            logger=logger,
        )

        normalization_config = (
                config.get("normalization")
                or {}
        )

        reference_config = (
                normalization_config.get("reference")
                or {}
        )

        train_reference_stats = getattr(
            train_normalizer,
            "stats",
            None,
        )

        split_policy = str(
            reference_config.get(
                "split_policy",
                "train_only",
            )
        ).lower()

        # train_only:
        #     val directly needs train global stats
        #
        # all:
        #     val needs train global stats + val stats
        #
        # val_only:
        #     val does not need train stats
        share_train_stats_with_val = (
                train_reference_stats is not None
                and split_policy in {
                    "train_only",
                    "all",
                }
        )

        val_normalizer = build_normalizer(
            config,
            stats=(
                train_reference_stats
                if share_train_stats_with_val
                else None
            ),
            split="val",
            project_root=project_root,
            logger=logger,
        )
        train_image_normalizer, train_batch_normalizer = split_normalizer_by_scope(
            train_normalizer
        )
        val_image_normalizer, val_batch_normalizer = split_normalizer_by_scope(
            val_normalizer
        )

        train_dataset = RxRxDataset(
            train_manifest,
            image_root,
            label_to_index,
            transform=train_transform,
            normalizer=train_image_normalizer,
        )
        val_dataset = RxRxDataset(
            val_manifest,
            image_root,
            label_to_index,
            transform=val_transform,
            normalizer=val_image_normalizer,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=config["data"]["shuffle_train"],
            num_workers=config["data"]["num_workers"],
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=config["data"]["shuffle_val"],
            num_workers=config["data"]["num_workers"],
            pin_memory=torch.cuda.is_available(),
        )

        model = build_efficientnet(
            name=config["model"]["name"],
            num_classes=len(label_to_index),
            pretrained=config["model"]["pretrained"],
        ).to(device)

        criterion = build_criterion(config).to(device)
        optimizer = build_optimizer(model, config)

        checkpoint_path = (
            project_root
            / config["checkpoint"]["dir"]
            / config["experiment"]["name"]
            / "best.pt"
        )

        log_training_started(logger, device, len(train_dataset), len(val_dataset))

        results = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=config["training"]["epochs"],
            checkpoint_enabled=config["checkpoint"]["enabled"],
            checkpoint_path=checkpoint_path,
            logger=logger,
            run=run,
            train_batch_normalizer=train_batch_normalizer,
            val_batch_normalizer=val_batch_normalizer,
        )

        log_training_finished(logger, results)
        finish_wandb(run, results)

    except Exception as error:
        log_training_failed(logger, error)
        fail_wandb(run)
        raise


if __name__ == "__main__":
    main()

