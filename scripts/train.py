from pathlib import Path
import argparse
from functools import partial

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from rxrx1.data.dataset import RxRxDataset
from rxrx1.data.transforms import resize_image
from rxrx1.data.manifest import create_label_to_index, read_manifest
from rxrx1.models.resnet import build_resnet
from rxrx1.training.trainer import train_one_epoch, validate_one_epoch
from rxrx1.training.criterion import build_criterion
from rxrx1.training.optimizers import build_optimizer
from rxrx1.training.checkpoint import save_checkpoint
from rxrx1.utils.paths import get_image_root
from rxrx1.utils.seed import set_seed
from rxrx1.utils.logger import setup_logger
from rxrx1.utils.tracking import setup_wandb




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,)
    return parser.parse_args()

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["experiment"]["seed"])

    project_root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger_name = config["experiment"]["name"]
    log_file = (
        project_root
        / config["logging"]["log_dir"]
        / f"{config['experiment']['name']}.log"
    )
    log_level = config["logging"]["level"]
    logger = setup_logger(logger_name, log_file, log_level)

    run = setup_wandb(config, project_root)

    train_manifest = read_manifest(
        project_root / config["data"]["train_manifest"]
    )
    val_manifest = read_manifest(
        project_root / config["data"]["val_manifest"]
    )

    train_labels = set(train_manifest["sirna"].unique())
    original_val_labels = set(val_manifest["sirna"].unique())

    val_manifest = (
        val_manifest[val_manifest["sirna"].isin(train_labels)]
        .copy()
        .reset_index(drop=True)
    )

    val_labels = set(val_manifest["sirna"].unique())

    if train_labels != val_labels:
        raise ValueError(
            "Train/val label mismatch after filtering | "
            f"train={len(train_labels)} | "
            f"val={len(val_labels)} | "
            f"train_only={len(train_labels - val_labels)} | "
            f"val_only={len(val_labels - train_labels)}"
        )

    logger.info(
        "Label check passed | train_labels=%d | "
        "original_val_labels=%d | val_labels=%d | removed_val_labels=%d",
        len(train_labels),
        len(original_val_labels),
        len(val_labels),
        len(original_val_labels - val_labels),
    )

    label_to_index = create_label_to_index(train_manifest)
    image_root = get_image_root("train")

    input_size = config["data"]["image_size"]
    if input_size != 512:
        train_transform = partial(resize_image, size=input_size)
        val_transform = partial(resize_image, size=input_size)
    else:
        train_transform = None
        val_transform = None

    train_dataset = RxRxDataset(
        train_manifest,
        image_root,
        label_to_index,
        transform=train_transform,
    )
    val_dataset = RxRxDataset(
        val_manifest,
        image_root,
        label_to_index,
        transform=val_transform,
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

    model = build_resnet(
        name=config["model"]["name"],
        num_classes=len(label_to_index),
        pretrained=config["model"]["pretrained"],
    ).to(device)

    criterion = build_criterion(config).to(device)
    optimizer = build_optimizer(model, config)

    best_val_acc = float("-inf")

    checkpoint_path = (
        project_root
        / config["checkpoint"]["dir"]
        / config["experiment"]["name"]
        / "best.pt"
    )

    logger.info(
        "Training started | device=%s | train_samples=%d | val_samples=%d",
        device,
        len(train_dataset),
        len(val_dataset),
    )

    best_val_acc = float("-inf")
    best_epoch = 0
    best_train_loss = None
    best_train_acc = None
    best_val_loss = None

    final_train_loss = None
    final_train_acc = None
    final_val_loss = None
    final_val_acc = None

    for epoch in tqdm(
            range(config["training"]["epochs"]),
            desc="Epoch",
    ):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        logger.info(
            "Epoch %d | train_loss=%.4f | train_acc=%.4f | "
            "val_loss=%.4f | val_acc=%.4f",
            epoch + 1,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        if run is not None:
            run.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                }
            )

        final_train_loss = train_loss
        final_train_acc = train_acc
        final_val_loss = val_loss
        final_val_acc = val_acc

        if val_acc > best_val_acc:
            best_epoch = epoch + 1
            best_train_loss = train_loss
            best_train_acc = train_acc
            best_val_loss = val_loss
            best_val_acc = val_acc

            if config["checkpoint"]["enabled"]:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    val_acc=val_acc,
                    path=checkpoint_path,
                )

                logger.info(
                    "Best checkpoint saved | epoch=%d | "
                    "val_acc=%.4f | val_loss=%.4f",
                    best_epoch,
                    best_val_acc,
                    best_val_loss,
                )

    logger.info(
        "Training finished | "
        "final_train_acc=%.4f | final_train_loss=%.4f | "
        "final_val_acc=%.4f | final_val_loss=%.4f | "
        "best_epoch=%d | best_train_acc=%.4f | best_train_loss=%.4f | "
        "best_val_acc=%.4f | best_val_loss=%.4f",
        final_train_acc,
        final_train_loss,
        final_val_acc,
        final_val_loss,
        best_epoch,
        best_train_acc,
        best_train_loss,
        best_val_acc,
        best_val_loss,
    )

    if run is not None:
        run.summary["final_train_acc"] = final_train_acc
        run.summary["final_train_loss"] = final_train_loss
        run.summary["final_val_acc"] = final_val_acc
        run.summary["final_val_loss"] = final_val_loss

        run.summary["best_epoch"] = best_epoch
        run.summary["best_train_acc"] = best_train_acc
        run.summary["best_train_loss"] = best_train_loss
        run.summary["best_val_acc"] = best_val_acc
        run.summary["best_val_loss"] = best_val_loss

        run.finish()


if __name__ == "__main__":
    main()