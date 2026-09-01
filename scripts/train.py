from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rxrx1.data.dataset import RxRxDataset
from rxrx1.data.manifest import create_label_to_index, read_manifest
from rxrx1.models.resnet import build_resnet
from rxrx1.training.trainer import train_one_epoch, validate_one_epoch
from rxrx1.utils.paths import get_image_root
from rxrx1.utils.seed import set_seed




def main():
    set_seed(42)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_manifest = read_manifest(
        PROJECT_ROOT / "data/processed/splits/d1_train_treatment.csv"
    )
    val_manifest = read_manifest(
        PROJECT_ROOT / "data/processed/splits/d1_val_treatment.csv"
    )

    label_to_index = create_label_to_index(train_manifest)
    image_root = get_image_root("train")

    train_dataset = RxRxDataset(
        train_manifest, image_root, label_to_index
    )
    val_dataset = RxRxDataset(
        val_manifest, image_root, label_to_index
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_resnet(
        name="resnet18",
        num_classes=len(label_to_index),
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    for epoch in tqdm(range(5), desc="Epoch"):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}"
        )


if __name__ == "__main__":
    main()