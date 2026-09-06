from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


CELL_TYPE_TO_IDX = {
    "HEPG2": 0,
    "HUVEC": 1,
    "RPE": 2,
    "U2OS": 3,
}

WELL_ROW_MIN = ord("B")
WELL_ROW_MAX = ord("O")
WELL_COL_MIN = 2
WELL_COL_MAX = 23


def encode_cell_type(cell_type):
    if cell_type not in CELL_TYPE_TO_IDX:
        raise ValueError(f"Unknown cell type: {cell_type}")

    return CELL_TYPE_TO_IDX[cell_type]


def encode_well_position(well):
    row = ord(well[0].upper())
    col = int(well[1:])

    relative_row = (
        (row - WELL_ROW_MIN)
        / (WELL_ROW_MAX - WELL_ROW_MIN)
    )
    relative_col = (
        (col - WELL_COL_MIN)
        / (WELL_COL_MAX - WELL_COL_MIN)
    )

    return relative_row, relative_col

class RxRxDataset(Dataset):
    """Load one six-channel RxRx1 site from a manifest row."""

    def __init__(
            self,
            manifest: pd.DataFrame,
            image_root: str | Path,
            label_to_index: dict,
            transform: Optional[Callable] = None,
            normalizer: Optional[Callable] = None,
            validate_paths: bool = False,
    ):
        self.manifest = manifest.copy().reset_index(drop=True)
        self.image_root = Path(image_root)
        self.label_to_index = label_to_index
        self.transform = transform
        self.normalizer = normalizer

        required_columns = {
            "experiment",
            "plate",
            "well",
            "site",
            "sirna",
            "cell_type",
        }
        missing_columns = required_columns - set(self.manifest.columns)

        if missing_columns:
            raise ValueError(
                "Manifest is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        unknown_labels = set(self.manifest["sirna"]) - set(self.label_to_index)

        if unknown_labels:
            raise ValueError(
                f"Manifest contains {len(unknown_labels)} "
                "labels that are not present in label_to_index."
            )

        if validate_paths:
            self._validate_all_paths()

        labels = set(self.manifest["sirna"].unique())
        unknown_labels = labels - set(self.label_to_index)

        if unknown_labels:
            raise ValueError(
                f"Manifest contains {len(unknown_labels)} labels "
                "not present in label_to_index."
            )

    def __len__(self):
        return len(self.manifest)

    def _image_path(self, row: pd.Series, channel: int) -> Path:
        return (
            self.image_root
            / str(row["experiment"])
            / f"Plate{int(row['plate'])}"
            / f"{row['well']}_s{int(row['site'])}_w{channel}.png"
        )

    def _load_six_channels(self, row: pd.Series) -> np.ndarray:
        channels = []

        for channel in range(1, 7):
            image_path = self._image_path(row, channel)
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

            if image is None:
                raise FileNotFoundError(
                    f"Unable to read image: {image_path}"
                )

            if image.ndim != 2:
                raise ValueError(
                    "Expected a single-channel image, "
                    f"but received shape {image.shape}: {image_path}"
                )

            channels.append(image)

        channel_shapes = {image.shape for image in channels}

        if len(channel_shapes) != 1:
            raise ValueError(
                "The six channels have different shapes: "
                f"{sorted(channel_shapes)}"
            )

        # Output shape: height × width × 6
        return np.stack(channels, axis=-1)

    def _validate_all_paths(self):
        missing_paths = []

        for _, row in self.manifest.iterrows():
            for channel in range(1, 7):
                image_path = self._image_path(row, channel)

                if not image_path.exists():
                    missing_paths.append(str(image_path))

        if missing_paths:
            preview = "\n".join(missing_paths[:10])

            raise FileNotFoundError(
                f"{len(missing_paths)} images are missing.\n"
                f"First missing paths:\n{preview}"
            )

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]

        # ========================================================
        # 1. Load raw six-channel image
        # ========================================================

        image = self._load_six_channels(row)

        # HWC -> CHW
        image = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .float()
        )

        # ========================================================
        # 2. Resolve image-level normalization position
        # ========================================================

        normalizer_position = None

        if self.normalizer is not None:

            apply_to = getattr(
                self.normalizer,
                "apply_to",
                "image",
            )

            if apply_to != "image":
                raise ValueError(
                    "RxRxDataset can only apply image-level normalizers. "
                    f"Received normalizer.apply_to={apply_to!r}. "
                    "Batch-level normalizers must be applied in the trainer."
                )

            normalizer_position = getattr(
                self.normalizer,
                "position",
                "before_resize",
            )

            if normalizer_position not in {
                "before_resize",
                "after_resize",
            }:
                raise ValueError(
                    "Unsupported normalization position: "
                    f"{normalizer_position!r}."
                )

        # ========================================================
        # 3. Normalization BEFORE resize
        # ========================================================
        #
        # Current experiment:
        #
        # raw image
        #     ↓
        # normalization
        #     ↓
        # resize / augmentation
        #

        if (
                self.normalizer is not None
                and normalizer_position == "before_resize"
        ):
            image = self.normalizer(
                image,
                row,
            )

        # ========================================================
        # 4. Transform pipeline
        # ========================================================
        #
        # Currently this may include:
        #
        # resize
        # flip
        # rotation
        # brightness
        # ...
        #

        if self.transform is not None:
            image = self.transform(image)

        # ========================================================
        # 5. after_resize is NOT applied here yet
        # ========================================================
        #
        # Important:
        #
        # self.transform may contain BOTH resize and augmentation.
        #
        # Therefore doing:
        #
        #     transform(image)
        #     normalization(image)
        #
        # would actually mean:
        #
        #     resize
        #     augmentation
        #     normalization
        #
        # which is NOT the intended:
        #
        #     resize
        #     normalization
        #     augmentation
        #
        # We deliberately reject it until resize and augmentation
        # are separated in the transform pipeline.
        #

        if (
                self.normalizer is not None
                and normalizer_position == "after_resize"
        ):
            raise NotImplementedError(
                "normalization.position='after_resize' requires "
                "the resize step to be separated from augmentation. "
                "The current Dataset receives one combined transform, "
                "so applying normalization here would incorrectly place "
                "it after augmentation."
            )

        # ========================================================
        # 6. Label
        # ========================================================

        label = self.label_to_index[
            row["sirna"]
        ]

        # ========================================================
        # 7. Return image + metadata
        # ========================================================

        cell_type_idx = encode_cell_type(
            row["cell_type"]
        )

        well_position = encode_well_position(
            row["well"]
        )

        return {
            "image": image,
            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),

            "experiment": row["experiment"],
            "cell_type": row["cell_type"],
            "plate": int(row["plate"]),
            "well": row["well"],
            "site": int(row["site"]),

            "cell_type_idx": torch.tensor(
                cell_type_idx,
                dtype=torch.long,
            ),
            "well_position": torch.tensor(
                well_position,
                dtype=torch.float32,
            ),
        }

if __name__ == "__main__":
    print(encode_cell_type("HEPG2"))
    print(encode_cell_type("U2OS"))

    print(encode_well_position("B02"))
    print(encode_well_position("O23"))
    print(encode_well_position("H12"))