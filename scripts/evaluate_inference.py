from pathlib import Path
import argparse
import logging

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from rxrx1.data.dataset import RxRxDataset
from rxrx1.data.manifest import create_label_to_index, read_manifest
from rxrx1.data.normalization import build_normalizer
from rxrx1.data.transforms import prepare_transforms
from rxrx1.inference.lsa import lsa_predict
from rxrx1.inference.tta import tta_logits
from rxrx1.models.factory import build_model
from rxrx1.training.validation import filter_and_validate_val_manifest
from rxrx1.utils.paths import get_image_root
from rxrx1.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_normalizer(normalizer):
    if normalizer is None:
        return None, None
    apply_to = getattr(normalizer, "apply_to", "image")
    if apply_to == "image":
        return normalizer, None
    if apply_to == "batch":
        return None, normalizer
    raise ValueError(f"Unknown normalizer scope: {apply_to}")


def prepare_metadata(batch, device):
    return {
        "cell_type_idx": batch["cell_type_idx"].to(device),
        "well_position": batch["well_position"].to(device),
    }


def unwrap_logits(outputs):
    return outputs[0] if isinstance(outputs, tuple) else outputs


@torch.inference_mode()
def collect_predictions(model, loader, device, batch_normalizer, tta_config):
    model.eval()
    records = []
    tta_enabled = tta_config.get("enabled", False)
    tta_mode = tta_config.get("mode", "d4")

    for batch in tqdm(loader, desc="Inference"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        metadata = prepare_metadata(batch, device)

        if batch_normalizer is not None:
            images = batch_normalizer(images, batch)

        raw = unwrap_logits(model(images, metadata))
        tta = (
            tta_logits(model, images, metadata=metadata, mode=tta_mode)
            if tta_enabled else raw
        )

        raw = raw.cpu().numpy()
        tta = tta.cpu().numpy()
        labels = labels.cpu().numpy()

        for i, id_code in enumerate(batch["id_code"]):
            records.append({
                "id_code": id_code,
                "experiment": batch["experiment"][i],
                "plate": int(batch["plate"][i]),
                "site": int(batch["site"][i]),
                "label": int(labels[i]),
                "raw_logits": raw[i],
                "tta_logits": tta[i],
            })

    return pd.DataFrame(records)


def aggregate_sites(predictions, logits_column):
    rows = []

    for id_code, group in predictions.groupby("id_code", sort=False):
        for column in ("experiment", "plate", "label"):
            if group[column].nunique() != 1:
                raise ValueError(f"{id_code}: inconsistent {column} across sites.")

        rows.append({
            "id_code": id_code,
            "experiment": group["experiment"].iloc[0],
            "plate": int(group["plate"].iloc[0]),
            "label": int(group["label"].iloc[0]),
            "logits": np.stack(group[logits_column].to_numpy()).mean(axis=0),
        })

    return pd.DataFrame(rows)


def accuracy(predictions, logits_column="logits"):
    logits = np.stack(predictions[logits_column].to_numpy())
    labels = predictions["label"].to_numpy()
    return float((logits.argmax(axis=1) == labels).mean())


def evaluate_lsa(predictions, train_manifest, label_to_index):
    lsa_input = predictions[["id_code", "experiment", "plate", "logits"]].copy()
    submission, assignments = lsa_predict(
        lsa_input,
        train_manifest,
        label_to_index,
    )

    scored = submission.merge(
        predictions[["id_code", "label"]],
        on="id_code",
        validate="one_to_one",
    )

    pred = scored["sirna"].map(label_to_index).to_numpy()
    target = scored["label"].to_numpy()

    return float((pred == target).mean()), assignments


def main():
    args = parse_args()
    config = load_config(args.config)
    inference = config.get("inference") or {}
    tta_config = inference.get("tta") or {}
    lsa_config = inference.get("lsa") or {}

    if inference.get("site_aggregation", "mean_logits") != "mean_logits":
        raise ValueError("Only site_aggregation=mean_logits is supported.")

    project_root = Path(__file__).resolve().parents[1]
    set_seed(config["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_manifest = read_manifest(project_root / config["data"]["train_manifest"])
    val_manifest = read_manifest(project_root / config["data"]["val_manifest"])
    val_manifest, *_ = filter_and_validate_val_manifest(train_manifest, val_manifest)
    label_to_index = create_label_to_index(train_manifest)

    _, val_transform = prepare_transforms(config)
    logger = logging.getLogger("inference")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    train_normalizer = build_normalizer(
        config,
        split="train",
        project_root=project_root,
        logger=logger,
    )

    stats = getattr(train_normalizer, "stats", None)
    reference = (config.get("normalization") or {}).get("reference") or {}
    split_policy = str(reference.get("split_policy", "train_only")).lower()
    shared_stats = stats if stats is not None and split_policy in {"train_only", "all"} else None

    val_normalizer = build_normalizer(
        config,
        stats=shared_stats,
        split="val",
        project_root=project_root,
        logger=logger,
    )

    val_image_normalizer, val_batch_normalizer = split_normalizer(val_normalizer)

    dataset = RxRxDataset(
        val_manifest,
        get_image_root("train"),
        label_to_index,
        transform=val_transform,
        normalizer=val_image_normalizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(
        model_config=config["model"],
        num_classes=len(label_to_index),
        metric=config.get("metric") or {},
    ).to(device)

    checkpoint = args.checkpoint or inference.get("checkpoint")
    if checkpoint is None:
        raise ValueError("Provide --checkpoint or inference.checkpoint in YAML.")

    checkpoint = Path(checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = project_root / checkpoint

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    print(f"Checkpoint: {checkpoint}")
    print(f"Epoch:      {state.get('epoch', 'unknown')}")
    print(f"Saved val:  {state.get('val_acc', 'unknown')}")
    print(f"TTA:        {tta_config.get('enabled', False)}")
    print(f"LSA:        {lsa_config.get('enabled', False)}")

    predictions = collect_predictions(
        model,
        loader,
        device,
        val_batch_normalizer,
        tta_config,
    )

    raw_site_acc = accuracy(
        predictions.rename(columns={"raw_logits": "logits"})
    )
    raw_well = aggregate_sites(predictions, "raw_logits")
    raw_well_acc = accuracy(raw_well)

    print(f"\nRaw site acc:      {raw_site_acc:.6f}")
    print(f"Raw site-mean acc: {raw_well_acc:.6f}")

    current = raw_well

    if tta_config.get("enabled", False):
        tta_site_acc = accuracy(
            predictions.rename(columns={"tta_logits": "logits"})
        )
        tta_well = aggregate_sites(predictions, "tta_logits")
        tta_well_acc = accuracy(tta_well)

        print(f"TTA site acc:      {tta_site_acc:.6f}")
        print(f"TTA site-mean acc: {tta_well_acc:.6f}")
        print(f"TTA delta:         {tta_well_acc - raw_well_acc:+.6f}")
        current = tta_well

    if lsa_config.get("enabled", False):
        lsa_acc, assignments = evaluate_lsa(
            current,
            train_manifest,
            label_to_index,
        )
        print(f"LSA acc:           {lsa_acc:.6f}")
        print(f"LSA delta:         {lsa_acc - accuracy(current):+.6f}")
        print(f"Plate assignments: {len(assignments)}")


if __name__ == "__main__":
    main()