from functools import partial
from pathlib import Path
import argparse

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from rxrx1.data.dataset import RxRxDataset
from rxrx1.data.manifest import create_label_to_index, read_manifest
from rxrx1.data.normalization import (
    ReferenceZScoreNormalizer,
    build_normalizer,
    fit_normalization_stats,
)
from rxrx1.data.transforms import prepare_transforms, resize_image
from rxrx1.models.factory import build_model


# These two rows are known invalid/problematic RxRx1 test wells and should
# never enter normalization statistics, inference, or final submissions.
KNOWN_INVALID_TEST_IDS = {
    "HUVEC-18_3_D23",
    "RPE-09_2_J16",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--norm-stats-source",
        choices=("train", "test"),
        default="train",
        help=(
            "Statistics used for reference z-score normalization. "
            "'train' preserves the original train-derived behavior; "
            "'test' fits mean/std directly from the target test images."
        ),
    )
    return p.parse_args()


def split_normalizer(normalizer):
    if normalizer is None:
        return None, None

    apply_to = getattr(normalizer, "apply_to", "image")

    if apply_to == "image":
        return normalizer, None

    if apply_to == "batch":
        return None, normalizer

    raise ValueError(
        f"Unknown normalizer scope: {apply_to}"
    )


def get_normalization_options(config):
    section = config.get("normalization") or {}

    enabled = bool(
        section.get(
            "switch",
            section.get("enabled", False),
        )
    )

    method = str(
        section.get(
            "method",
            section.get("mode", "none"),
        )
    ).lower()

    statistics = section.get("statistics") or {}

    source = str(
        statistics.get("source", "reference")
    ).lower()

    grouping = str(
        statistics.get(
            "grouping",
            section.get("grouping", ""),
        )
    ).lower()

    spatial = str(
        statistics.get(
            "spatial",
            section.get("spatial", "global"),
        )
    ).lower()

    channel = str(
        statistics.get(
            "channel",
            section.get("channel", "per_channel"),
        )
    ).lower()

    std_type = str(
        statistics.get(
            "std_type",
            section.get("std_type", "population"),
        )
    ).lower()

    application = section.get("application") or {}

    position = str(
        application.get("position", "before_resize")
    ).lower()

    eps = float(section.get("eps", 1.0e-6))

    missing_group = str(
        section.get("missing_group", "error")
    ).lower()

    return {
        "enabled": enabled,
        "method": method,
        "source": source,
        "grouping": grouping,
        "spatial": spatial,
        "channel": channel,
        "std_type": std_type,
        "position": position,
        "eps": eps,
        "missing_group": missing_group,
    }


def filter_valid_test_wells(test, sample):
    if "id_code" not in test.columns:
        raise ValueError(
            "data/raw/test.csv is missing id_code."
        )

    if "id_code" not in sample.columns:
        raise ValueError(
            "sample_submission.csv is missing id_code."
        )

    if sample["id_code"].duplicated().any():
        duplicated = (
            sample.loc[
                sample["id_code"].duplicated(),
                "id_code",
            ]
            .astype(str)
            .tolist()[:10]
        )

        raise ValueError(
            f"Duplicate id_code values in sample submission: {duplicated}"
        )

    original_test_rows = len(test)
    original_sample_rows = len(sample)

    invalid_test = sorted(
        set(test["id_code"].astype(str))
        & KNOWN_INVALID_TEST_IDS
    )

    invalid_sample = sorted(
        set(sample["id_code"].astype(str))
        & KNOWN_INVALID_TEST_IDS
    )

    test = test.loc[
        ~test["id_code"]
        .astype(str)
        .isin(KNOWN_INVALID_TEST_IDS)
    ].copy()

    sample = sample.loc[
        ~sample["id_code"]
        .astype(str)
        .isin(KNOWN_INVALID_TEST_IDS)
    ].copy()

    target_ids = set(
        sample["id_code"].astype(str)
    )

    raw_test_ids = set(
        test["id_code"].astype(str)
    )

    missing_metadata = sorted(
        target_ids - raw_test_ids
    )

    if missing_metadata:
        raise RuntimeError(
            "Target submission contains id_code values "
            "missing from test.csv: "
            f"{missing_metadata[:10]}"
        )

    extra_metadata = sorted(
        raw_test_ids - target_ids
    )

    if extra_metadata:
        print(
            "Ignoring test.csv rows not required by "
            f"submission target: {len(extra_metadata)}"
        )

        print(
            "Extra id_code preview:",
            extra_metadata[:10],
        )

    test = test.loc[
        test["id_code"]
        .astype(str)
        .isin(target_ids)
    ].copy()

    print(
        "Test well filtering | "
        f"raw_test={original_test_rows} "
        f"raw_sample={original_sample_rows} "
        f"invalid_test={invalid_test} "
        f"invalid_sample={invalid_sample} "
        f"target={len(sample)}"
    )

    return (
        test.reset_index(drop=True),
        sample.reset_index(drop=True),
    )


def expand_complete_test_sites(test, test_root):
    test = test.merge(
        pd.DataFrame({"site": [1, 2]}),
        how="cross",
    )

    def site_complete(row):
        base = (
            test_root
            / row["experiment"]
            / f"Plate{int(row['plate'])}"
        )

        return all(
            (
                base
                / (
                    f"{row['well']}"
                    f"_s{int(row['site'])}"
                    f"_w{channel}.png"
                )
            ).is_file()
            for channel in range(1, 7)
        )

    site_mask = test.apply(
        site_complete,
        axis=1,
    )

    missing_sites = test.loc[
        ~site_mask,
        [
            "id_code",
            "experiment",
            "plate",
            "well",
            "site",
        ],
    ]

    if not missing_sites.empty:
        print(
            "Skipping unavailable test sites:"
        )

        print(
            missing_sites.to_string(
                index=False
            )
        )

    test = (
        test.loc[site_mask]
        .reset_index(drop=True)
    )

    wells_with_site = set(
        test["id_code"].astype(str)
    )

    all_wells = set(
        (
            test["id_code"].astype(str)
            if len(test)
            else pd.Series(dtype=str)
        )
    )

    if test.empty:
        raise RuntimeError(
            "No complete test sites were found."
        )

    return test


def build_test_stats_normalizer(
    config,
    test_manifest,
    label_to_index,
    root,
):
    options = get_normalization_options(
        config
    )

    if not options["enabled"]:
        print(
            "--norm-stats-source=test ignored: "
            "normalization is disabled."
        )

        return build_normalizer(
            config,
            split="val",
            project_root=root,
        )

    if options["method"] != "zscore":
        print(
            "--norm-stats-source=test ignored: "
            f"normalization method={options['method']!r} "
            "does not use fitted reference z-score statistics."
        )

        return build_normalizer(
            config,
            split="val",
            project_root=root,
        )

    if options["source"] == "sample":
        print(
            "--norm-stats-source=test ignored: "
            "statistics.source='sample' computes mean/std "
            "independently for each image."
        )

        return build_normalizer(
            config,
            split="val",
            project_root=root,
        )

    if options["source"] != "reference":
        raise ValueError(
            "Unsupported normalization statistics source: "
            f"{options['source']!r}"
        )

    if options["grouping"] == "loader_batch":
        print(
            "--norm-stats-source=test ignored: "
            "grouping='loader_batch' computes mean/std "
            "from each inference mini-batch."
        )

        return build_normalizer(
            config,
            split="val",
            project_root=root,
        )

    if options["grouping"] not in {
        "global",
        "experiment",
        "plate",
    }:
        raise ValueError(
            "Test-derived statistics only support "
            "global / experiment / plate grouping, "
            f"got {options['grouping']!r}."
        )

    position = options["position"]

    if position == "before_resize":
        fit_transform = None

    elif position == "after_resize":
        fit_transform = partial(
            resize_image,
            size=int(
                config["data"]["image_size"]
            ),
        )

    else:
        raise ValueError(
            "Unsupported normalization position: "
            f"{position!r}"
        )

    stats_dataset = RxRxDataset(
        manifest=test_manifest,
        image_root=root / "data/raw/test",
        label_to_index=label_to_index,
        transform=fit_transform,
        normalizer=None,
    )

    stats_loader = DataLoader(
        stats_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=False,
    )

    print()
    print("=" * 70)
    print("FITTING NORMALIZATION STATISTICS FROM TEST IMAGES")
    print("=" * 70)
    print(
        f"images:      {len(stats_dataset)}"
    )
    print(
        f"grouping:    {options['grouping']}"
    )
    print(
        f"spatial:     {options['spatial']}"
    )
    print(
        f"channel:     {options['channel']}"
    )
    print(
        f"std type:    {options['std_type']}"
    )
    print(
        f"position:    {options['position']}"
    )
    print(
        "population:  target test images"
    )
    print("=" * 70)

    stats = fit_normalization_stats(
        loader=stats_loader,
        grouping=options["grouping"],
        spatial=options["spatial"],
        channel=options["channel"],
        std_type=options["std_type"],
    )

    if "global" in stats:
        global_stats = stats["global"]

        mean_values = (
            global_stats.mean
            .detach()
            .cpu()
            .flatten()
            .tolist()
        )

        std_values = (
            global_stats.std
            .detach()
            .cpu()
            .flatten()
            .tolist()
        )

        print(
            "test global mean:",
            mean_values,
        )

        print(
            "test global std: ",
            std_values,
        )

    normalizer = ReferenceZScoreNormalizer(
        stats=stats,
        grouping=options["grouping"],
        eps=options["eps"],
        missing_group=options[
            "missing_group"
        ],
    )

    normalizer.position = position

    return normalizer


def build_inference_normalizer(
    config,
    test_manifest,
    label_to_index,
    root,
    norm_stats_source,
):
    if norm_stats_source == "train":
        # Preserve the historical behavior:
        # fit/load TRAIN statistics and apply
        # those statistics to test images.
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

        test_normalizer = build_normalizer(
            config,
            stats=train_stats,
            split="val",
            project_root=root,
        )

        print(
            "Normalization statistics source: TRAIN"
        )

        return test_normalizer

    if norm_stats_source == "test":
        print(
            "Normalization statistics source: TEST"
        )

        return build_test_stats_normalizer(
            config=config,
            test_manifest=test_manifest,
            label_to_index=label_to_index,
            root=root,
        )

    raise ValueError(
        f"Unknown norm stats source: {norm_stats_source}"
    )


def main():
    args = parse_args()

    root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    # Automatic reference-stat fitting for TRAIN
    # must still resolve the training images.
    config["data"]["image_root"] = (
        "data/raw/train"
    )

    train_manifest = read_manifest(
        root
        / config["data"]["train_manifest"]
    )

    label_to_index = (
        create_label_to_index(
            train_manifest
        )
    )

    index_to_label = {
        index: label
        for label, index
        in label_to_index.items()
    }

    # --------------------------------------------------------
    # Build the exact submission target first.
    # --------------------------------------------------------

    sample = pd.read_csv(
        root
        / "data/raw/sample_submission.csv"
    )

    raw_test = pd.read_csv(
        root
        / "data/raw/test.csv"
    )

    raw_test, sample = (
        filter_valid_test_wells(
            raw_test,
            sample,
        )
    )

    raw_test["cell_type"] = (
        raw_test["experiment"]
        .str.split("-")
        .str[0]
    )

    test_root = (
        root
        / "data/raw/test"
    )

    test = expand_complete_test_sites(
        raw_test,
        test_root,
    )

    # Every target well must retain at least one site.
    target_ids = set(
        sample["id_code"].astype(str)
    )

    available_ids = set(
        test["id_code"].astype(str)
    )

    unavailable_wells = sorted(
        target_ids - available_ids
    )

    if unavailable_wells:
        raise RuntimeError(
            "No complete image site exists for "
            "submission wells: "
            f"{unavailable_wells[:10]}"
        )

    # RxRxDataset requires a known training label,
    # but this dummy label is never used in inference.
    dummy_label = next(
        iter(label_to_index)
    )

    test["sirna"] = dummy_label

    _, test_transform = (
        prepare_transforms(config)
    )

    test_normalizer = (
        build_inference_normalizer(
            config=config,
            test_manifest=test,
            label_to_index=label_to_index,
            root=root,
            norm_stats_source=(
                args.norm_stats_source
            ),
        )
    )

    (
        image_normalizer,
        batch_normalizer,
    ) = split_normalizer(
        test_normalizer
    )

    dataset = RxRxDataset(
        manifest=test,
        image_root=test_root,
        label_to_index=label_to_index,
        transform=test_transform,
        normalizer=image_normalizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=config["data"][
            "batch_size"
        ],
        shuffle=False,
        num_workers=config["data"][
            "num_workers"
        ],
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = build_model(
        model_config=config["model"],
        num_classes=len(label_to_index),
        metric=(
            config.get("metric")
            or {}
        ),
    ).to(device)

    checkpoint_path = (
        root
        / args.checkpoint
    )

    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        state["model_state_dict"]
    )

    model.eval()

    print()
    print(
        f"Checkpoint: {checkpoint_path}"
    )
    print(
        f"Epoch:      {state.get('epoch')}"
    )
    print(
        f"Device:     {device}"
    )
    print(
        "Norm stats: "
        f"{args.norm_stats_source}"
    )
    print(
        f"Target wells: {len(sample)}"
    )
    print(
        f"Test sites:   {len(dataset)}"
    )
    print()

    logit_sums = {}
    site_counts = {}

    amp_enabled = (
        device.type == "cuda"
    )

    amp_dtype = torch.bfloat16

    if (
        device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        amp_dtype = torch.float16

        print(
            "BF16 unsupported on this GPU; "
            "using FP16 inference."
        )

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc="Test inference",
        ):
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
                    batch[
                        "cell_type_idx"
                    ].to(
                        device,
                        non_blocking=True,
                    ),
                "well_position":
                    batch[
                        "well_position"
                    ].to(
                        device,
                        non_blocking=True,
                    ),
            }

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                outputs = model(
                    images,
                    metadata,
                )

                logits = (
                    outputs[0]
                    if isinstance(
                        outputs,
                        tuple,
                    )
                    else outputs
                )

            logits = (
                logits
                .float()
                .cpu()
            )

            for i, id_code in enumerate(
                batch["id_code"]
            ):
                id_code = str(id_code)

                if id_code not in target_ids:
                    raise RuntimeError(
                        "Inference produced "
                        "unexpected id_code: "
                        f"{id_code}"
                    )

                if id_code not in logit_sums:
                    logit_sums[id_code] = (
                        logits[i].clone()
                    )

                    site_counts[id_code] = 1

                else:
                    logit_sums[
                        id_code
                    ] += logits[i]

                    site_counts[
                        id_code
                    ] += 1

    predicted_ids = set(
        logit_sums
    )

    missing_predictions = sorted(
        target_ids
        - predicted_ids
    )

    extra_predictions = sorted(
        predicted_ids
        - target_ids
    )

    if missing_predictions:
        raise RuntimeError(
            "Missing predictions for "
            f"{len(missing_predictions)} wells: "
            f"{missing_predictions[:10]}"
        )

    if extra_predictions:
        raise RuntimeError(
            "Unexpected extra predictions for "
            f"{len(extra_predictions)} wells: "
            f"{extra_predictions[:10]}"
        )

    predictions = {}

    for (
        id_code,
        total_logits,
    ) in logit_sums.items():

        mean_logits = (
            total_logits
            / site_counts[id_code]
        )

        class_index = int(
            mean_logits.argmax()
        )

        label = index_to_label[
            class_index
        ]

        if (
            isinstance(label, str)
            and label.startswith(
                "sirna_"
            )
        ):
            sirna = int(
                label.removeprefix(
                    "sirna_"
                )
            )

        else:
            sirna = int(label)

        predictions[
            id_code
        ] = sirna

    submission = (
        sample[["id_code"]]
        .copy()
    )

    submission["sirna"] = (
        submission["id_code"]
        .astype(str)
        .map(predictions)
    )

    if submission[
        "sirna"
    ].isna().any():

        missing = (
            submission.loc[
                submission[
                    "sirna"
                ].isna(),
                "id_code",
            ]
            .tolist()[:10]
        )

        raise RuntimeError(
            "Missing final submission "
            f"predictions for: {missing}"
        )

    submission[
        "sirna"
    ] = (
        submission["sirna"]
        .astype(int)
    )

    if len(submission) != len(
        target_ids
    ):
        raise RuntimeError(
            "Submission row count "
            "does not match unique "
            "target id count."
        )

    if (
        submission["id_code"]
        .astype(str)
        .isin(
            KNOWN_INVALID_TEST_IDS
        )
        .any()
    ):
        raise RuntimeError(
            "Known invalid test IDs "
            "entered final submission."
        )

    output_path = (
        root
        / args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        submission.head()
    )
    print()
    print(
        f"Rows:       {len(submission)}"
    )
    print(
        "Unique IDs: "
        f"{submission['id_code'].nunique()}"
    )
    print(
        "Pred classes: "
        f"{submission['sirna'].nunique()}"
    )
    print(
        f"Submission: {output_path}"
    )


if __name__ == "__main__":
    main()
