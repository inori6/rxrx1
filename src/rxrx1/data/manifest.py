from pathlib import Path

import pandas as pd


def read_manifest(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return pd.read_csv(path)


def build_manifest(metadata_path, sites=(1, 2)):
    manifest = pd.read_csv(metadata_path)

    required = {"experiment", "plate", "well", "sirna"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

    manifest["cell_type"] = manifest["experiment"].str.split("-").str[0]
    manifest = manifest.merge(pd.DataFrame({"site": sites}), how="cross")
    return manifest


def create_label_to_index(manifest):
    sirna_list = sorted(manifest["sirna"].unique())
    return {sirna: index for index, sirna in enumerate(sirna_list)}