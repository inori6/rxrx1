from pathlib import Path
import argparse

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from rxrx1.data.dataset import RxRxDataset
from rxrx1.data.manifest import create_label_to_index, read_manifest
from rxrx1.data.normalization import build_normalizer
from rxrx1.data.transforms import prepare_transforms
from rxrx1.models.factory import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def split_normalizer(normalizer):
    if normalizer is None:
        return None, None

    apply_to = getattr(normalizer, "apply_to", "image")

    if apply_to == "image":
        return normalizer, None

    if apply_to == "batch":
        return None, normalizer

    raise ValueError(f"Unknown normalizer scope: {apply_to}")


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Normalization statistics must still come from the TRAIN images.
    config["data"]["image_root"] = "data/raw/train"

    train_manifest = read_manifest(
        root / config["data"]["train_manifest"]
    )

    label_to_index = create_label_to_index(train_manifest)
    index_to_label = {
        index: label
        for label, index in label_to_index.items()
    }

    # Original Kaggle test metadata is one row per well.
    test = pd.read_csv(root / "data/raw/test.csv")
    test["cell_type"] = (
        test["experiment"].str.split("-").str[0]
    )

    # Expand each well into site 1 + site 2.
    test = test.merge(
        pd.DataFrame({"site": [1, 2]}),
        how="cross",
    )

    # Some official test wells contain only one available site.
    # Keep only sites for which all 6 channel images exist.
    test_root = root / "data/raw/test"

    def site_complete(row):
        base = test_root / row["experiment"] / f"Plate{int(row['plate'])}"
        return all(
            (base / f"{row['well']}_s{int(row['site'])}_w{ch}.png").is_file()
            for ch in range(1, 7)
        )

    site_mask = test.apply(site_complete, axis=1)

    missing_sites = test.loc[
        ~site_mask,
        ["id_code", "experiment", "plate", "well", "site"],
    ]

    if not missing_sites.empty:
        print("Skipping unavailable test sites:")
        print(missing_sites.to_string(index=False))

    test = test.loc[site_mask].reset_index(drop=True)

    # RxRxDataset requires a known label.
    # During inference this label is never used for prediction.
    dummy_label = next(iter(label_to_index))
    test["sirna"] = dummy_label

    _, test_transform = prepare_transforms(config)

    # Fit/load the same train-derived normalization statistics
    # used by training.
    train_normalizer = build_normalizer(
        config,
        split="train",
        project_root=root,
    )

    train_stats = getattr(
        train_normalizer,
        "stats",
        None,
    )

    # train_only normalization:
    # apply TRAIN statistics to TEST images.
    test_normalizer = build_normalizer(
        config,
        stats=train_stats,
        split="val",
        project_root=root,
    )

    image_normalizer, batch_normalizer = split_normalizer(
        test_normalizer
    )

    dataset = RxRxDataset(
        manifest=test,
        image_root=root / "data/raw/test",
        label_to_index=label_to_index,
        transform=test_transform,
        normalizer=image_normalizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = build_model(
        model_config=config["model"],
        num_classes=len(label_to_index),
        metric=config.get("metric") or {},
    ).to(device)

    checkpoint_path = root / args.checkpoint

    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(state["model_state_dict"])
    model.eval()

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Epoch:      {state.get('epoch')}")
    print(f"Device:     {device}")

    logit_sums = {}
    site_counts = {}

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Test inference"):
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            if batch_normalizer is not None:
                images = batch_normalizer(
                    images,
                    batch,
                )

            metadata = {
                "cell_type_idx":
                    batch["cell_type_idx"].to(
                        device,
                        non_blocking=True,
                    ),
                "well_position":
                    batch["well_position"].to(
                        device,
                        non_blocking=True,
                    ),
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                outputs = model(images, metadata)
                logits = (
                    outputs[0]
                    if isinstance(outputs, tuple)
                    else outputs
                )

            logits = logits.float().cpu()

            for i, id_code in enumerate(batch["id_code"]):
                if id_code not in logit_sums:
                    logit_sums[id_code] = logits[i].clone()
                    site_counts[id_code] = 1
                else:
                    logit_sums[id_code] += logits[i]
                    site_counts[id_code] += 1

    predictions = {}

    for id_code, total_logits in logit_sums.items():
        mean_logits = (
            total_logits / site_counts[id_code]
        )

        class_index = int(mean_logits.argmax())
        label = index_to_label[class_index]

        if isinstance(label, str) and label.startswith("sirna_"):
            sirna = int(label.removeprefix("sirna_"))
        else:
            sirna = int(label)

        predictions[id_code] = sirna

    sample = pd.read_csv(
        root / "data/raw/sample_submission.csv"
    )

    submission = sample[["id_code"]].copy()

    submission["sirna"] = (
        submission["id_code"]
        .map(predictions)
    )

    if submission["sirna"].isna().any():
        missing = submission.loc[
            submission["sirna"].isna(),
            "id_code",
        ].tolist()[:10]

        raise RuntimeError(
            f"Missing predictions for: {missing}"
        )

    submission["sirna"] = (
        submission["sirna"]
        .astype(int)
    )

    output_path = root / args.output
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        output_path,
        index=False,
    )

    print()
    print(submission.head())
    print(f"Rows:       {len(submission)}")
    print(f"Submission: {output_path}")


if __name__ == "__main__":
    main()
