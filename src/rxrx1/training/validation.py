def compare_labels(train_labels, val_labels):
    if train_labels != val_labels:
        raise ValueError(
            "Train/val label mismatch after filtering | "
            f"train={len(train_labels)} | "
            f"val={len(val_labels)} | "
            f"train_only={len(train_labels - val_labels)} | "
            f"val_only={len(val_labels - train_labels)}"
        )


def filter_and_validate_val_manifest(
    train_manifest,
    val_manifest,
):
    train_labels = set(train_manifest["sirna"].unique())
    original_val_labels = set(val_manifest["sirna"].unique())

    val_manifest = (
        val_manifest[
            val_manifest["sirna"].isin(train_labels)
        ]
        .copy()
        .reset_index(drop=True)
    )

    val_labels = set(val_manifest["sirna"].unique())

    compare_labels(
        train_labels,
        val_labels,
    )

    return (
        val_manifest,
        train_labels,
        original_val_labels,
        val_labels,
    )