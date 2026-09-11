from pathlib import Path
import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from rxrx1.data.dataset import RxRxDataset
from rxrx1.data.sampler import PKBatchSampler
from rxrx1.data.manifest import create_label_to_index, read_manifest
from rxrx1.data.transforms import prepare_transforms, build_batch_transform
from rxrx1.data.normalization import build_normalizer
from rxrx1.models.factory import build_model
from rxrx1.training.criterion import build_criterion
from rxrx1.training.optimizers import build_optimizer
from rxrx1.training.schedulers import build_scheduler
from rxrx1.training.trainer import fit_model
from rxrx1.training.checkpoint import load_checkpoint
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
from rxrx1.utils.tracking import setup_wandb, update_wandb_git_info, finish_wandb, fail_wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_normalizer_by_scope(normalizer):
    if normalizer is None: return None, None
    apply_to = getattr(normalizer, "apply_to", "image")
    if apply_to == "image": return normalizer, None
    if apply_to == "batch": return None, normalizer
    raise ValueError(f"Unknown normalizer apply_to scope: {apply_to!r}.")


def run_training(config, epoch_callback=None):
    metric_config = config.get("metric") or {}
    criterion = build_criterion(config)
    set_seed(config["experiment"]["seed"])

    project_root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger_name = config["experiment"]["name"]
    log_file = project_root / config["logging"]["log_dir"] / f"{logger_name}.log"
    logger = setup_logger(logger_name, log_file, config["logging"]["level"])

    run = setup_wandb(config, project_root)
    update_wandb_git_info(run, os.getenv("RXRX1_GIT_COMMIT"), os.getenv("RXRX1_GIT_REF"))

    try:
        training_config = config.get("training") or {}
        validation_enabled = bool(training_config.get("validation", True))

        amp_config = training_config.get("amp") or {}
        amp_enabled = bool(amp_config.get("enabled", False))
        amp_dtype = str(amp_config.get("dtype", "bf16")).lower()
        if amp_enabled and amp_dtype != "bf16": raise ValueError(f"Unsupported AMP dtype: {amp_dtype}")
        if amp_enabled and device.type == "cuda" and not torch.cuda.is_bf16_supported(): raise RuntimeError(
            "CUDA device does not support BF16.")
        logger.info("AMP | enabled=%s | dtype=%s", amp_enabled, amp_dtype if amp_enabled else "fp32")

        train_manifest = read_manifest(project_root / config["data"]["train_manifest"])

        if validation_enabled:
            val_manifest = read_manifest(project_root / config["data"]["val_manifest"])
            val_manifest, train_labels, original_val_labels, val_labels = filter_and_validate_val_manifest(
                train_manifest, val_manifest
            )
            log_label_revised(logger, train_labels, original_val_labels, val_labels)
        else:
            val_manifest = None

        label_to_index = create_label_to_index(train_manifest)
        image_root = get_image_root("train")
        train_transform, val_transform = prepare_transforms(config)

        transform_config = config.get("transform") or {}
        batch_config = (transform_config.get("batch") or []) if transform_config.get("switch") else []
        train_batch_transform = build_batch_transform(batch_config, len(label_to_index))
        train_acc_definition = "soft_target_distribution_match" if batch_config else "hard_label_accuracy"

        logger.info("Training batch transforms: %s", batch_config)
        logger.info("train/acc definition: %s", train_acc_definition)

        if run is not None:
            run.config.update({
                "effective_batch_transforms": batch_config,
                "train_acc_definition": train_acc_definition,
                "validation_enabled": validation_enabled,
            })

        train_normalizer = build_normalizer(
            config,
            split="train",
            project_root=project_root,
            logger=logger,
        )
        train_image_normalizer, train_batch_normalizer = split_normalizer_by_scope(train_normalizer)

        if validation_enabled:
            reference_config = (config.get("normalization") or {}).get("reference") or {}
            train_reference_stats = getattr(train_normalizer, "stats", None)
            split_policy = str(reference_config.get("split_policy", "train_only")).lower()

            shared_stats = (
                train_reference_stats
                if train_reference_stats is not None and split_policy in {"train_only", "all"}
                else None
            )

            val_normalizer = build_normalizer(
                config,
                stats=shared_stats,
                split="val",
                project_root=project_root,
                logger=logger,
            )
            val_image_normalizer, val_batch_normalizer = split_normalizer_by_scope(val_normalizer)
        else:
            val_image_normalizer = None
            val_batch_normalizer = None

        train_dataset = RxRxDataset(
            train_manifest,
            image_root,
            label_to_index,
            transform=train_transform,
            normalizer=train_image_normalizer,
        )

        val_dataset = None
        if validation_enabled:
            val_dataset = RxRxDataset(
                val_manifest,
                image_root,
                label_to_index,
                transform=val_transform,
                normalizer=val_image_normalizer,
            )

        sampler_config = config["data"].get("pk_sampler") or {}

        if sampler_config.get("enabled", False):
            train_sampling = {
                "batch_sampler": PKBatchSampler(
                    train_dataset.manifest["sirna"].tolist(),
                    config["data"]["batch_size"],
                    k=sampler_config.get("k", 4),
                    seed=config["experiment"]["seed"],
                )
            }
        else:
            train_sampling = {
                "batch_size": config["data"]["batch_size"],
                "shuffle": config["data"]["shuffle_train"],
            }

        train_loader = DataLoader(
            train_dataset,
            **train_sampling,
            num_workers=config["data"]["num_workers"],
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = None
        if validation_enabled:
            val_loader = DataLoader(
                val_dataset,
                batch_size=config["data"]["batch_size"],
                shuffle=config["data"].get("shuffle_val", False),
                num_workers=config["data"]["num_workers"],
                pin_memory=torch.cuda.is_available(),
            )

        model = build_model(
            model_config=config["model"],
            num_classes=len(label_to_index),
            metric=metric_config,
        ).to(device)

        criterion = criterion.to(device)
        optimizer = build_optimizer(model, config)

        scheduler = build_scheduler(
            optimizer=optimizer,
            config=config,
            epochs=training_config["epochs"],
            steps_per_epoch=len(train_loader),
        )

        checkpoint_dir = project_root / config["checkpoint"]["dir"] / config["experiment"]["name"]
        best_checkpoint_path = checkpoint_dir / "best.pt"
        last_checkpoint_path = checkpoint_dir / "last.pt"

        start_epoch = 0
        resume_from = training_config.get("resume_from")

        if resume_from:
            resume_path = project_root / resume_from
            ckpt = load_checkpoint(
                path=resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                total_epochs=training_config["epochs"],
                steps_per_epoch=len(train_loader),
                device=device,
            )
            start_epoch = int(ckpt["epoch"])
            logger.info(
                "Resumed checkpoint | path=%s | completed_epoch=%d | next_epoch=%d",
                resume_path,
                start_epoch,
                start_epoch + 1,
            )

        log_training_started(
            logger,
            device,
            len(train_dataset),
            len(val_dataset) if val_dataset is not None else None,
        )

        results = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=training_config["epochs"],
            checkpoint_enabled=config["checkpoint"]["enabled"],
            checkpoint_path=best_checkpoint_path,
            last_checkpoint_path=last_checkpoint_path,
            logger=logger,
            run=run,
            scheduler=scheduler,
            train_batch_normalizer=train_batch_normalizer,
            val_batch_normalizer=val_batch_normalizer,
            epoch_callback=epoch_callback,
            train_batch_transform=train_batch_transform,
            start_epoch=start_epoch,
            stop_after_epoch=training_config.get("stop_after_epoch"),
            train_acc_checkpoint_config=training_config.get("train_acc_checkpoint"),
            amp_enabled=amp_enabled,
        )

        log_training_finished(logger, results)
        finish_wandb(run, results)
        return results

    except Exception as error:
        log_training_failed(logger, error)
        fail_wandb(run)
        raise


def main():
    args = parse_args()
    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()