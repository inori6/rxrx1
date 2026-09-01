from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RxRxDataset(Dataset):
    """Load one six-channel RxRx1 site from a manifest row."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        image_root: str | Path,
        label_to_index: dict,
        transform: Optional[Callable] = None,
        validate_paths: bool = False,
    ):
        self.manifest = manifest.copy().reset_index(drop=True)
        self.image_root = Path(image_root)
        self.label_to_index = label_to_index
        self.transform = transform

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
        image = self._load_six_channels(row)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(
                image.astype(np.float32)
            ).permute(2, 0, 1)

        label = self.label_to_index[row["sirna"]]

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "experiment": row["experiment"],
            "cell_type": row["cell_type"],
            "plate": int(row["plate"]),
            "well": row["well"],
            "site": int(row["site"]),
        }