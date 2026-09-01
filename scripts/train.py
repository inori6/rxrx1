from pathlib import Path
import argparse

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from rxrx1.data.dataset import RxRxDataset
from rxrx1.data.manifest import create_label_to_index, read_manifest
from rxrx1.models.resnet import build_resnet
from rxrx1.training.trainer import train_one_epoch, validate_one_epoch
from rxrx1.training.criterion import build_criterion
from rxrx1.training.optimizers import build_optimizer
from rxrx1.training.checkpoint import save_checkpoint
from rxrx1.utils.paths import get_image_root
from rxrx1.utils.seed import set_seed
from rxrx1.utils.logger import setup_logger




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

    logger_name = config["experiment"]["name"]
    log_file = (
        Path(config["logging"]["log_dir"])
        / f"{config['experiment']['name']}.log"
    )
    log_level = config["logging"]["level"]
    logger = setup_logger(logger_name, log_file, log_level)

    set_seed(config["experiment"]["seed"])

    project_root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_manifest = read_manifest(
        project_root / config["data"]["train_manifest"]
    )
    val_manifest = read_manifest(
        project_root / config["data"]["val_manifest"]
    )

    label_to_index = create_label_to_index(train_manifest)
    image_root = get_image_root("train")

    train_dataset = RxRxDataset(
        train_manifest,
        image_root,
        label_to_index,
    )
    val_dataset = RxRxDataset(
        val_manifest,
        image_root,
        label_to_index,
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
        Path(config["checkpoint"]["dir"])
        / config["experiment"]["name"]
        / "best.pt"
    )

    logger.info("Training started")

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

        if config["checkpoint"]["enabled"] and val_acc > best_val_acc:
            best_val_acc = val_acc

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                val_acc=val_acc,
                path=checkpoint_path,
            )

            logger.info(
                "Best checkpoint saved | epoch=%d | val_acc=%.4f",
                epoch + 1,
                val_acc,
            )

    logger.info("Training finished | best_val_acc=%.4f", best_val_acc)

if __name__ == "__main__":
    main()