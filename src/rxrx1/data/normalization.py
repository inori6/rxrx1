"""Normalization statistics, caching, and runtime normalizers for RxRx1.

Integration changes made with this module:

* ``scripts/train.py`` separates image-level and loader-batch normalizers by
  their ``apply_to`` attribute before constructing datasets and starting fit.
* ``src/rxrx1/training/trainer.py`` applies loader-batch normalization after a
  batch is moved to its training device.

Population selection, technical grouping, statistics geometry, and the
normalization formula intentionally remain independent concerns in this file.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from functools import partial

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


_GLOBAL_KEY = "global"
_CACHE_FORMAT_VERSION = 3

# normalization 使用的数学公式
_METHODS = {
    "none",
    "zscore",
    "channel_standard",  # 兼容旧版手动 mean/std
    # 以后：
    # "control_correction",
}

# mean/std 从哪里获得
_SOURCES = {
    "sample",
    "reference",
}

# reference statistics 按什么 scope/group 计算
_GROUPINGS = {
    "global",
    "experiment",
    "plate",
    "loader_batch",
}

_SPATIAL_GEOMETRIES = {
    "global",
    "pixel",
}

_CHANNEL_GEOMETRIES = {
    "shared",
    "per_channel",
}

_STD_TYPES = {
    "population",
    "sample",
}

_APPLICATION_POSITIONS = {
    "before_resize",
    "after_resize",
}


# ============================================================
# Data structures
# ============================================================


@dataclass
class NormStats:
    """
    Mean/std tensors and counts used to estimate them.

    count:
        Number of images.

    element_count:
        Number of scalar values contributing to each statistic.

        This is required when correctly pooling statistics from
        different reference populations.
    """

    mean: torch.Tensor
    std: torch.Tensor
    count: int
    element_count: int | None = None

    def __post_init__(self) -> None:
        if (
            not torch.is_tensor(self.mean)
            or not torch.is_tensor(self.std)
        ):
            raise TypeError(
                "NormStats mean and std must be torch tensors."
            )

        if self.mean.shape != self.std.shape:
            raise ValueError(
                "NormStats mean and std must have identical shapes, "
                f"got {tuple(self.mean.shape)} "
                f"and {tuple(self.std.shape)}."
            )

        if int(self.count) <= 0:
            raise ValueError(
                f"NormStats count must be positive, got {self.count}."
            )

        if (
            not torch.isfinite(self.mean).all()
            or not torch.isfinite(self.std).all()
        ):
            raise ValueError(
                "NormStats mean/std contain non-finite values."
            )

        if (self.std < 0).any():
            raise ValueError(
                "NormStats std cannot contain negative values."
            )

        self.count = int(self.count)

        if self.element_count is not None:
            self.element_count = int(
                self.element_count
            )

            if self.element_count <= 0:
                raise ValueError(
                    "NormStats element_count must be positive."
                )


@dataclass
class NormStatsStore:
    """Statistics indexed by global, experiment, or (experiment, plate) key."""

    stats: dict[Hashable, NormStats] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, key: Hashable, stats: NormStats) -> None:
        if not isinstance(stats, NormStats):
            raise TypeError("stats must be a NormStats instance.")
        self.stats[key] = stats

    def get(self, key: Hashable) -> NormStats:
        try:
            return self.stats[key]
        except KeyError as error:
            preview = list(self.stats)[:5]
            raise KeyError(
                f"No normalization statistics for group {key!r}. "
                f"Available key preview: {preview!r}."
            ) from error

    def __getitem__(self, key: Hashable) -> NormStats:
        return self.get(key)

    def __contains__(self, key: object) -> bool:
        return key in self.stats

    def __len__(self) -> int:
        return len(self.stats)

    def items(self):
        return self.stats.items()


# ============================================================
# Config helpers and validation
# ============================================================


def _normalization_section(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
    if not isinstance(config, Mapping):
        raise TypeError("Normalization config must be a mapping.")
    if "normalization" in config:
        section = config.get("normalization") or {}
        if not isinstance(section, Mapping):
            raise TypeError("config.normalization must be a mapping.")
        return section, True
    return config, False


def _is_enabled(section: Mapping[str, Any], embedded: bool) -> bool:
    default = not embedded
    return bool(section.get("switch", section.get("enabled", default)))


def _method(section: Mapping[str, Any]) -> str:
    method = section.get("method")
    mode = section.get("mode")
    if method is not None and mode is not None and str(method).lower() != str(mode).lower():
        raise ValueError("normalization.method and normalization.mode disagree.")
    return str(method if method is not None else mode if mode is not None else "none").lower()


def _statistics_options(section: Mapping[str, Any]) -> dict[str, Any]:
    nested = section.get("statistics") or {}
    if not isinstance(nested, Mapping):
        raise TypeError("normalization.statistics must be a mapping.")
    return {
        "grouping": str(nested.get("grouping", section.get("grouping", ""))).lower(),
        "spatial": str(nested.get("spatial", section.get("spatial", "global"))).lower(),
        "channel": str(
            nested.get("channel", section.get("channel", "per_channel"))
        ).lower(),
        "std_type": str(nested.get("std_type", section.get("std_type", "population"))).lower(),
        "source": str(nested.get("source", "reference")).lower(),
    }


def _reference_options(section: Mapping[str, Any]) -> Mapping[str, Any]:
    reference = section.get("reference") or {}
    if not isinstance(reference, Mapping):
        raise TypeError("normalization.reference must be a mapping.")
    return reference


def _cache_options(section: Mapping[str, Any]) -> Mapping[str, Any]:
    cache = section.get("cache") or {}
    if not isinstance(cache, Mapping):
        raise TypeError("normalization.cache must be a mapping.")
    return cache


def validate_normalization_config(config: Mapping[str, Any]) -> None:
    """
    Validate normalization configuration.

    The configuration intentionally separates four independent concepts:

    method:
        Which mathematical normalization formula is used.

    statistics.source:
        Where normalization statistics come from.

    statistics.grouping:
        Which images share one set of reference statistics.

    statistics.spatial / statistics.channel:
        Which tensor dimensions are aggregated when computing statistics.
    """

    section, embedded = _normalization_section(config)

    # --------------------------------------------------------
    # enabled
    # --------------------------------------------------------
    if not _is_enabled(section, embedded):
        return

    # --------------------------------------------------------
    # method
    # --------------------------------------------------------
    method = _method(section)

    if method not in _METHODS:
        raise ValueError(
            f"Unknown normalization method: {method!r}. "
            f"Expected one of {sorted(_METHODS)}."
        )

    if method == "none":
        return

    # --------------------------------------------------------
    # eps
    # --------------------------------------------------------
    eps = float(section.get("eps", 1.0e-6))

    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(
            "normalization.eps must be a finite value greater than zero."
        )

    # --------------------------------------------------------
    # application.position
    # --------------------------------------------------------
    application = _application_options(section)

    position = str(
        application.get("position", "before_resize")
    ).lower()

    if position not in _APPLICATION_POSITIONS:
        raise ValueError(
            "normalization.application.position must be one of "
            f"{sorted(_APPLICATION_POSITIONS)}, got {position!r}."
        )

    # --------------------------------------------------------
    # Backward-compatible manually configured normalization
    # --------------------------------------------------------
    if method == "channel_standard":
        params = section.get("params") or {}

        if not isinstance(params, Mapping):
            raise TypeError(
                "normalization.params must be a mapping."
            )

        mean = params.get("mean")
        std = params.get("std")

        if mean is None or std is None:
            raise ValueError(
                "channel_standard requires params.mean and params.std."
            )

        if len(mean) != len(std) or not mean:
            raise ValueError(
                "channel_standard mean/std must be non-empty "
                "and have equal length."
            )

        if any(float(value) <= 0 for value in std):
            raise ValueError(
                "channel_standard std values must all be greater than zero."
            )

        return

    # --------------------------------------------------------
    # z-score statistics configuration
    # --------------------------------------------------------
    if method != "zscore":
        raise NotImplementedError(
            f"Normalization method {method!r} is registered "
            "but is not implemented yet."
        )

    options = _statistics_options(section)

    source = options["source"]
    grouping = options["grouping"]
    spatial = options["spatial"]
    channel = options["channel"]
    std_type = options["std_type"]

    # --------------------------------------------------------
    # statistics.source
    # --------------------------------------------------------
    if source not in _SOURCES:
        raise ValueError(
            "normalization.statistics.source must be one of "
            f"{sorted(_SOURCES)}, got {source!r}."
        )

    # --------------------------------------------------------
    # statistics.spatial
    # --------------------------------------------------------
    if spatial not in _SPATIAL_GEOMETRIES:
        raise ValueError(
            "normalization.statistics.spatial must be one of "
            f"{sorted(_SPATIAL_GEOMETRIES)}, got {spatial!r}."
        )

    # --------------------------------------------------------
    # statistics.channel
    # --------------------------------------------------------
    if channel not in _CHANNEL_GEOMETRIES:
        raise ValueError(
            "normalization.statistics.channel must be one of "
            f"{sorted(_CHANNEL_GEOMETRIES)}, got {channel!r}."
        )

    # --------------------------------------------------------
    # statistics.std_type
    # --------------------------------------------------------
    if std_type not in _STD_TYPES:
        raise ValueError(
            "normalization.statistics.std_type must be one of "
            f"{sorted(_STD_TYPES)}, got {std_type!r}."
        )

    # ========================================================
    # source = sample
    # ========================================================
    #
    # 当前 image 自己计算 statistics。
    #
    # grouping / reference population 在这种情况下没有意义。
    #
    if source == "sample":

        # [1, C, H, W] 下如果只沿 B 求统计，
        # 每个 (C,H,W) 实际只有一个值：
        #
        # mean = x
        # std  = 0
        #
        # 因而该组合退化。
        if spatial == "pixel" and channel == "per_channel":
            raise ValueError(
                "statistics.source='sample' with "
                "spatial='pixel' and channel='per_channel' "
                "is degenerate because each statistic contains "
                "only one value."
            )

        return

    # ========================================================
    # source = reference
    # ========================================================

    # --------------------------------------------------------
    # grouping
    # --------------------------------------------------------
    if grouping not in _GROUPINGS:
        raise ValueError(
            "normalization.statistics.grouping must be one of "
            f"{sorted(_GROUPINGS)}, got {grouping!r}."
        )

    # loader_batch 是动态 statistics。
    # 后续在 factory/runtime integration 中单独处理。
    #
    # 对它来说 reference.population 没有实际意义，
    # 因此这里不强制要求 population。
    if grouping != "loader_batch":

        # ----------------------------------------------------
        # reference.population
        # ----------------------------------------------------
        reference = _reference_options(section)

        population = str(
            reference.get("population", "")
        ).strip().upper()

        valid_populations = {
            "N",
            "P",
            "NP",
            "TRAIN",
            "TRAIN_CONTROL",
        }

        if population not in valid_populations:
            raise ValueError(
                "normalization.reference.population must be one of "
                "N, P, NP, train, train_control."
            )

        # ----------------------------------------------------
        # reference.split_policy
        # ----------------------------------------------------
        split_policy = str(
            reference.get(
                "split_policy",
                "train_only",
            )
        ).lower()

        valid_split_policies = {
            "train_only",
            "val_only",
            "all",
        }

        if split_policy not in valid_split_policies:
            raise ValueError(
                "normalization.reference.split_policy must be one of "
                "'train_only', 'val_only', or 'all'."
            )

    # --------------------------------------------------------
    # cache
    # --------------------------------------------------------
    cache = _cache_options(section)

    missing = str(
        cache.get("missing", "compute")
    ).lower()

    if missing not in {"compute", "error"}:
        raise ValueError(
            "normalization.cache.missing must be "
            "'compute' or 'error'."
        )


def _application_options(section: Mapping[str, Any]) -> Mapping[str, Any]:
    application = section.get("application") or {}

    if not isinstance(application, Mapping):
        raise TypeError("normalization.application must be a mapping.")

    return application


# ============================================================
# Reference population
# ============================================================


def build_reference_manifest(
    population: str,
    train_manifest: pd.DataFrame,
    control_manifest: pd.DataFrame | None = None,
    *,
    control_column: str = "well_type",
) -> pd.DataFrame:
    """Select rows used to fit statistics without knowing grouping or geometry."""

    if not isinstance(train_manifest, pd.DataFrame):
        raise TypeError("train_manifest must be a pandas DataFrame.")
    population_key = str(population).strip().upper()

    if population_key == "TRAIN":
        result = train_manifest.copy()
    else:
        if control_manifest is None:
            raise ValueError(f"Reference population {population!r} requires control_manifest.")
        if not isinstance(control_manifest, pd.DataFrame):
            raise TypeError("control_manifest must be a pandas DataFrame.")

        if population_key == "NP":
            result = control_manifest.copy()
        elif population_key in {"N", "P"}:
            if control_column not in control_manifest.columns:
                raise ValueError(
                    f"Control manifest is missing required column {control_column!r}."
                )
            kinds = control_manifest[control_column].astype(str).str.strip().str.lower()
            expected = "negative_control" if population_key == "N" else "positive_control"
            result = control_manifest.loc[kinds.eq(expected)].copy()
        elif population_key == "TRAIN_CONTROL":
            result = pd.concat([train_manifest, control_manifest], ignore_index=True, sort=False)
        else:
            raise ValueError(f"Unknown reference population: {population!r}.")

    if result.empty:
        raise ValueError(f"Reference population {population!r} selected zero rows.")
    return result.reset_index(drop=True)


# ============================================================
# Grouping
# ============================================================


def _plain_scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("Expected scalar metadata tensor.")
        return value.item()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def get_group_key(metadata: Any, grouping: str) -> Hashable:
    """Map one image's metadata to its technical statistics group."""

    grouping = str(grouping).lower()
    if grouping == "global":
        return _GLOBAL_KEY
    if not isinstance(metadata, Mapping) and not hasattr(metadata, "__getitem__"):
        raise TypeError("Grouped normalization requires mapping-like metadata.")
    if "experiment" not in metadata:
        raise KeyError("Grouped normalization metadata is missing 'experiment'.")

    experiment = str(_plain_scalar(metadata["experiment"]))
    if grouping == "experiment":
        return experiment
    if grouping == "plate":
        if "plate" not in metadata:
            raise KeyError("Plate grouping metadata is missing 'plate'.")
        return experiment, int(_plain_scalar(metadata["plate"]))
    raise ValueError(f"Unsupported grouping: {grouping!r}.")


# ============================================================
# Statistics geometry
# ============================================================


def _get_reduce_dims(spatial: str, channel: str) -> tuple[int, ...]:
    """Return reduction dimensions for an image batch shaped [B, C, H, W]."""

    spatial, channel = str(spatial).lower(), str(channel).lower()
    if spatial not in _SPATIAL_GEOMETRIES or channel not in _CHANNEL_GEOMETRIES:
        raise ValueError(f"Unsupported statistics geometry: {spatial}/{channel}.")
    if spatial == "global" and channel == "shared":
        return 0, 1, 2, 3
    if spatial == "global" and channel == "per_channel":
        return 0, 2, 3
    if spatial == "pixel" and channel == "shared":
        return 0, 1
    return (0,)


def _validate_images(images: torch.Tensor) -> None:
    if not torch.is_tensor(images):
        raise TypeError("Expected images to be a torch.Tensor.")
    if images.ndim != 4:
        raise ValueError(f"Expected images shaped [B, C, H, W], got {tuple(images.shape)}.")
    if images.shape[0] == 0:
        raise ValueError("Cannot compute statistics from an empty image batch.")


def _sufficient_statistics(
    images: torch.Tensor,
    spatial: str,
    channel: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    _validate_images(images)
    values = images.detach().to(device="cpu", dtype=torch.float64)
    reduce_dims = _get_reduce_dims(spatial, channel)
    value_sum = values.sum(dim=reduce_dims, keepdim=True).squeeze(0)
    square_sum = values.square().sum(dim=reduce_dims, keepdim=True).squeeze(0)
    element_count = math.prod(values.shape[dim] for dim in reduce_dims)
    return value_sum, square_sum, int(element_count)


def _finalize_statistics(
    value_sum: torch.Tensor,
    square_sum: torch.Tensor,
    element_count: int,
    image_count: int,
    std_type: str,
) -> NormStats:
    correction = (
        0
        if std_type == "population"
        else 1
    )

    denominator = (
        element_count - correction
    )

    if denominator <= 0:
        raise ValueError(
            f"Cannot compute {std_type} std from "
            f"{element_count} value(s) per statistic."
        )

    mean = (
        value_sum
        / element_count
    )

    centered_square_sum = (
        square_sum
        - value_sum.square()
        / element_count
    )

    variance = (
        centered_square_sum
        / denominator
    ).clamp_min(0.0)

    return NormStats(
        mean=mean.to(torch.float32),
        std=variance.sqrt().to(torch.float32),
        count=image_count,

        # Required for later exact weighted pooling.
        element_count=element_count,
    )


def compute_batch_stats(
    images: torch.Tensor,
    spatial: str,
    channel: str,
    std_type: str = "population",
) -> NormStats:
    """Compute one batch's stats for any supported geometry."""

    std_type = str(std_type).lower()
    if std_type not in _STD_TYPES:
        raise ValueError(f"Unsupported std_type: {std_type!r}.")
    value_sum, square_sum, element_count = _sufficient_statistics(
        images, spatial, channel
    )
    return _finalize_statistics(
        value_sum, square_sum, element_count, int(images.shape[0]), std_type
    )


# ============================================================
# Streaming statistics fit
# ============================================================


@dataclass
class _Accumulator:
    value_sum: torch.Tensor
    square_sum: torch.Tensor
    element_count: int
    image_count: int


def _unpack_loader_batch(batch: Any) -> tuple[torch.Tensor, Mapping[str, Any] | None]:
    if isinstance(batch, Mapping):
        images = batch.get("image", batch.get("images"))
        if images is None:
            raise KeyError("Loader batch must contain 'image' or 'images'.")
        return images, batch
    if isinstance(batch, (tuple, list)) and batch:
        metadata = batch[1] if len(batch) > 1 and isinstance(batch[1], Mapping) else None
        return batch[0], metadata
    raise TypeError("Loader must yield a mapping or a non-empty tuple/list.")


def _metadata_item(batch: Mapping[str, Any], index: int, batch_size: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("experiment", "plate"):
        if name not in batch:
            continue
        values = batch[name]
        if torch.is_tensor(values):
            result[name] = values[index] if values.ndim > 0 else values
        elif isinstance(values, (list, tuple)):
            result[name] = values[index]
        elif hasattr(values, "__len__") and not isinstance(values, (str, bytes)):
            result[name] = values[index]
        elif batch_size == 1:
            result[name] = values
        else:
            raise ValueError(f"Cannot index batched metadata field {name!r}.")
    return result


def fit_normalization_stats(
    loader: Iterable[Any],
    grouping: str,
    spatial: str,
    channel: str,
    std_type: str = "population",
) -> NormStatsStore:
    """
    Fit reference statistics in float64.

    For experiment/plate grouping, both are saved:

        1. group-specific statistics
        2. pooled global statistics over ALL reference images

    Example:

        grouping = plate

    produces:

        stats["global"]
        stats[("HEPG2-05", 1)]
        stats[("HEPG2-05", 2)]
        ...
    """

    grouping = str(
        grouping
    ).lower()

    std_type = str(
        std_type
    ).lower()

    valid_fit_groupings = {
        "global",
        "experiment",
        "plate",
    }

    if grouping not in valid_fit_groupings:
        raise ValueError(
            "fit_normalization_stats supports only "
            "global / experiment / plate, "
            f"got {grouping!r}."
        )

    if std_type not in _STD_TYPES:
        raise ValueError(
            f"Unsupported std_type: {std_type!r}."
        )

    group_accumulators: dict[
        Hashable,
        _Accumulator,
    ] = {}

    global_accumulator: _Accumulator | None = None

    total_images = 0

    # ========================================================
    # Iterate reference images
    # ========================================================

    for batch in loader:

        images, metadata_batch = (
            _unpack_loader_batch(batch)
        )

        _validate_images(images)

        batch_size = int(
            images.shape[0]
        )

        total_images += batch_size

        # ====================================================
        # Always accumulate GLOBAL reference statistics
        # ====================================================

        (
            global_sum,
            global_square_sum,
            global_element_count,
        ) = _sufficient_statistics(
            images,
            spatial,
            channel,
        )

        if global_accumulator is None:

            global_accumulator = _Accumulator(
                value_sum=global_sum,
                square_sum=global_square_sum,
                element_count=global_element_count,
                image_count=batch_size,
            )

        else:

            if (
                global_accumulator.value_sum.shape
                != global_sum.shape
            ):
                raise ValueError(
                    "Statistics shape changed while "
                    "fitting global reference statistics."
                )

            global_accumulator.value_sum += (
                global_sum
            )

            global_accumulator.square_sum += (
                global_square_sum
            )

            global_accumulator.element_count += (
                global_element_count
            )

            global_accumulator.image_count += (
                batch_size
            )

        # grouping=global needs nothing else.
        if grouping == "global":
            continue

        # ====================================================
        # Group-specific statistics
        # ========================================================

        if metadata_batch is None:
            raise ValueError(
                f"{grouping} grouping requires loader metadata."
            )

        grouped_indices: dict[
            Hashable,
            list[int],
        ] = {}

        for index in range(batch_size):

            metadata = _metadata_item(
                metadata_batch,
                index,
                batch_size,
            )

            key = get_group_key(
                metadata,
                grouping,
            )

            grouped_indices.setdefault(
                key,
                [],
            ).append(index)

        for key, indices in grouped_indices.items():

            subset = images[
                indices
            ]

            (
                value_sum,
                square_sum,
                element_count,
            ) = _sufficient_statistics(
                subset,
                spatial,
                channel,
            )

            if key not in group_accumulators:

                group_accumulators[key] = (
                    _Accumulator(
                        value_sum=value_sum,
                        square_sum=square_sum,
                        element_count=element_count,
                        image_count=len(indices),
                    )
                )

                continue

            accumulator = (
                group_accumulators[key]
            )

            if (
                accumulator.value_sum.shape
                != value_sum.shape
            ):
                raise ValueError(
                    "Statistics shape changed within "
                    f"group {key!r}."
                )

            accumulator.value_sum += (
                value_sum
            )

            accumulator.square_sum += (
                square_sum
            )

            accumulator.element_count += (
                element_count
            )

            accumulator.image_count += (
                len(indices)
            )

    # ========================================================
    # Finalize
    # ========================================================

    if (
        total_images == 0
        or global_accumulator is None
    ):
        raise ValueError(
            "Reference loader is empty; "
            "cannot fit normalization statistics."
        )

    store = NormStatsStore()

    # ALWAYS save pooled global statistics.
    store.add(
        _GLOBAL_KEY,
        _finalize_statistics(
            global_accumulator.value_sum,
            global_accumulator.square_sum,
            global_accumulator.element_count,
            global_accumulator.image_count,
            std_type,
        ),
    )

    # Save plate / experiment statistics.
    if grouping != "global":

        for key, accumulator in (
            group_accumulators.items()
        ):

            store.add(
                key,
                _finalize_statistics(
                    accumulator.value_sum,
                    accumulator.square_sum,
                    accumulator.element_count,
                    accumulator.image_count,
                    std_type,
                ),
            )

    return store


def merge_norm_stats(
    first: NormStats,
    second: NormStats,
    std_type: str,
) -> NormStats:
    """
    Correctly pool two independently fitted statistics.

    This is NOT:

        mean = (mean1 + mean2) / 2
        std  = (std1 + std2) / 2

    Instead, statistics are weighted by the number of
    contributing scalar values.
    """

    std_type = str(
        std_type
    ).lower()

    if std_type not in _STD_TYPES:
        raise ValueError(
            f"Unsupported std_type: {std_type!r}."
        )

    if (
        first.element_count is None
        or second.element_count is None
    ):
        raise ValueError(
            "Pooling statistics requires element_count."
        )

    if first.mean.shape != second.mean.shape:
        raise ValueError(
            "Cannot merge statistics with different shapes: "
            f"{tuple(first.mean.shape)} vs "
            f"{tuple(second.mean.shape)}."
        )

    n1 = int(
        first.element_count
    )

    n2 = int(
        second.element_count
    )

    n = n1 + n2

    mean1 = first.mean.to(
        dtype=torch.float64
    )

    mean2 = second.mean.to(
        dtype=torch.float64
    )

    std1 = first.std.to(
        dtype=torch.float64
    )

    std2 = second.std.to(
        dtype=torch.float64
    )

    correction = (
        0
        if std_type == "population"
        else 1
    )

    m2_1 = (
        std1.square()
        * (n1 - correction)
    )

    m2_2 = (
        std2.square()
        * (n2 - correction)
    )

    delta = (
        mean2 - mean1
    )

    mean = (
        mean1
        + delta
        * (n2 / n)
    )

    m2 = (
        m2_1
        + m2_2
        + delta.square()
        * (n1 * n2 / n)
    )

    denominator = (
        n - correction
    )

    if denominator <= 0:
        raise ValueError(
            "Not enough values to merge statistics."
        )

    variance = (
        m2 / denominator
    ).clamp_min(0.0)

    return NormStats(
        mean=mean.to(torch.float32),
        std=variance.sqrt().to(torch.float32),
        count=(
            first.count
            + second.count
        ),
        element_count=n,
    )


def merge_train_global_with_val_stats(
    train_stats: NormStatsStore,
    val_stats: NormStatsStore,
    grouping: str,
    std_type: str,
) -> NormStatsStore:
    """
    Build validation statistics for split_policy='all'.

    train:
        use pooled train-global reference statistics

    val:
        use its own group-specific reference statistics

    resulting val statistics:
        train global + corresponding val group
    """

    if _GLOBAL_KEY not in train_stats:
        raise KeyError(
            "Train reference statistics are missing "
            "pooled global statistics."
        )

    if _GLOBAL_KEY not in val_stats:
        raise KeyError(
            "Val reference statistics are missing "
            "pooled global statistics."
        )

    grouping = str(
        grouping
    ).lower()

    result = NormStatsStore()

    train_global = train_stats.get(
        _GLOBAL_KEY
    )

    # Always provide merged global statistics.
    result.add(
        _GLOBAL_KEY,
        merge_norm_stats(
            train_global,
            val_stats.get(_GLOBAL_KEY),
            std_type,
        ),
    )

    if grouping != "global":

        for key, stats in val_stats.items():

            if key == _GLOBAL_KEY:
                continue

            result.add(
                key,
                merge_norm_stats(
                    train_global,
                    stats,
                    std_type,
                ),
            )

    result.metadata = {
        "policy": "all",
        "train": train_stats.metadata,
        "val": val_stats.metadata,
    }

    return result


# ============================================================
# Metadata
# ============================================================


def _plain_metadata(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_metadata(item) for item in value]
    value = _plain_scalar(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_stats_metadata(
    config: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    """
    Build provenance metadata for fitted normalization statistics.

    The metadata is stored together with mean/std/count so that a cached
    statistics file can only be reused when its statistical assumptions
    match the current normalization configuration.

    Important:
        method, statistics source, grouping, geometry, reference population,
        split policy, and application position are treated as independent
        parts of the cache identity.
    """

    validate_normalization_config(config)

    section, _ = _normalization_section(config)

    method = _method(section)
    options = _statistics_options(section)
    reference = _reference_options(section)
    application = _application_options(section)

    split = str(split).lower()

    # --------------------------------------------------------
    # Core normalization/statistics identity
    # --------------------------------------------------------

    metadata: dict[str, Any] = {
        "schema_version": _CACHE_FORMAT_VERSION,

        # Mathematical normalization formula
        #
        # Example:
        #   zscore
        "method": method,

        # Where mean/std come from
        #
        # sample:
        #   current image
        #
        # reference:
        #   external reference image population
        "source": options["source"],

        # Which technical unit shares one set of statistics
        #
        # global / experiment / plate / loader_batch
        "grouping": options["grouping"] or None,

        # Statistics geometry
        #
        # spatial:
        #   global / pixel
        #
        # channel:
        #   shared / per_channel
        "spatial": options["spatial"],
        "channel": options["channel"],

        # population / sample std
        "std_type": options["std_type"],

        # Which split was actually used to fit this cache.
        #
        # Note:
        # build_normalizer() should pass the already resolved reference split
        # here. For split_policy=train_only this will normally be "train".
        "split": split,

        # Whether normalization happens before or after resize.
        #
        # This is critical for spatial=pixel statistics because the tensor
        # geometry depends on the image resolution at statistics-fit time.
        "application_position": str(
            application.get(
                "position",
                "before_resize",
            )
        ).lower(),
    }

    # --------------------------------------------------------
    # Reference-specific metadata
    # --------------------------------------------------------

    if options["source"] == "reference" and options["grouping"] != "loader_batch":

        population = str(
            reference.get(
                "population",
                "",
            )
        ).strip().upper()

        split_policy = str(
            reference.get(
                "split_policy",
                "train_only",
            )
        ).lower()

        metadata.update(
            {
                # N / P / NP / TRAIN / TRAIN_CONTROL
                "population": population,

                # train_only / same_split
                "split_policy": split_policy,
            }
        )

    else:
        # Keep the schema explicit instead of silently omitting the fields.
        metadata.update(
            {
                "population": None,
                "split_policy": None,
            }
        )

    # --------------------------------------------------------
    # Dataset provenance
    # --------------------------------------------------------
    #
    # These are not used for the mathematics directly, but they help prevent
    # accidentally loading statistics fitted from a different manifest.
    #

    if "data" in config and isinstance(config["data"], Mapping):
        data = config["data"]

        metadata["manifest"] = data.get(
            f"{split}_manifest"
        )

        metadata["control_manifest"] = data.get(
            f"{split}_control_manifest",
            data.get("control_manifest"),
        )

        # image_size only affects the fitted statistics when normalization
        # statistics are computed after resize.
        #
        # For before_resize we deliberately store None here because the
        # configured model input size does not define the statistics geometry.
        if metadata["application_position"] == "after_resize":
            metadata["image_size"] = data.get(
                "image_size"
            )
        else:
            metadata["image_size"] = None

    # --------------------------------------------------------
    # Return only simple serializable metadata
    # --------------------------------------------------------

    return _plain_metadata(metadata)

def _metadata_mismatches(actual: Any, expected: Any, path: str = "metadata") -> list[str]:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return [f"{path}: expected mapping, got {type(actual).__name__}"]
        mismatches: list[str] = []
        for key, expected_value in expected.items():
            child_path = f"{path}.{key}"
            if key not in actual:
                mismatches.append(f"{child_path}: missing")
            else:
                mismatches.extend(
                    _metadata_mismatches(actual[key], expected_value, child_path)
                )
        return mismatches
    if actual != expected:
        return [f"{path}: cached={actual!r}, expected={expected!r}"]
    return []


def validate_stats_metadata(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Validate every expected field while allowing future extra cache metadata."""

    mismatches = _metadata_mismatches(_plain_metadata(actual), _plain_metadata(expected))
    if mismatches:
        details = "\n".join(f"- {item}" for item in mismatches)
        raise ValueError(f"Cached normalization statistics do not match config:\n{details}")


# ============================================================
# Cache
# ============================================================


def _safe_cache_component(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-_")
    return text or "none"


def build_stats_cache_path(
    config: Mapping[str, Any],
    cache_dir: str | Path | None,
    split: str,
) -> Path:
    """
    Build a human-readable cache path from the normalization assumptions.

    The filename is only for readability and reducing accidental collisions.
    The final source of truth is still the metadata stored inside the cache.
    """

    validate_normalization_config(config)

    section, _ = _normalization_section(config)

    method = _method(section)
    options = _statistics_options(section)
    reference = _reference_options(section)
    application = _application_options(section)

    split = str(split).lower()

    if cache_dir is None:
        cache_dir = _cache_options(section).get(
            "dir",
            "outputs/normalization_stats",
        )

    source = options["source"]

    position = str(
        application.get(
            "position",
            "before_resize",
        )
    ).lower()

    # --------------------------------------------------------
    # Reference-specific identity
    # --------------------------------------------------------

    if source == "reference" and options["grouping"] != "loader_batch":
        population = str(
            reference.get(
                "population",
                "none",
            )
        ).upper()

        split_policy = str(
            reference.get(
                "split_policy",
                "train_only",
            )
        ).lower()

    else:
        population = "none"
        split_policy = "none"

    # --------------------------------------------------------
    # Filename components
    # --------------------------------------------------------

    components = (
        split,
        method,
        source,
        population,
        split_policy,
        options["grouping"] or "none",
        options["spatial"],
        options["channel"],
        options["std_type"],
        position,
    )

    filename = (
        "__".join(
            _safe_cache_component(component)
            for component in components
        )
        + ".pt"
    )

    return Path(cache_dir) / filename


def save_norm_stats(
    stats_store: NormStatsStore,
    path: str | Path,
) -> None:
    """Atomically save normalization statistics."""

    if (
        not isinstance(
            stats_store,
            NormStatsStore,
        )
        or len(stats_store) == 0
    ):
        raise ValueError(
            "Cannot save an empty NormStatsStore."
        )

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "format_version": _CACHE_FORMAT_VERSION,
        "metadata": _plain_metadata(
            stats_store.metadata
        ),
        "stats": {
            key: {
                "mean": (
                    stats.mean
                    .detach()
                    .cpu()
                    .contiguous()
                ),
                "std": (
                    stats.std
                    .detach()
                    .cpu()
                    .contiguous()
                ),
                "count": stats.count,
                "element_count": stats.element_count,
            }
            for key, stats
            in stats_store.items()
        },
    }

    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )

    try:
        torch.save(
            payload,
            temporary_path,
        )

        temporary_path.replace(
            path
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def load_norm_stats(
    path: str | Path,
    expected_metadata: Mapping[str, Any] | None = None,
) -> NormStatsStore:
    """Load cached normalization statistics."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            "Normalization statistics cache "
            f"not found: {path}"
        )

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(
        payload,
        Mapping,
    ):
        raise ValueError(
            f"Invalid normalization cache payload: {path}"
        )

    if (
        payload.get("format_version")
        != _CACHE_FORMAT_VERSION
    ):
        raise ValueError(
            "Unsupported normalization cache format: "
            f"{payload.get('format_version')!r}. "
            f"Expected {_CACHE_FORMAT_VERSION}."
        )

    metadata = (
        payload.get("metadata")
        or {}
    )

    if expected_metadata is not None:
        validate_stats_metadata(
            metadata,
            expected_metadata,
        )

    serialized_stats = payload.get(
        "stats"
    )

    if (
        not isinstance(
            serialized_stats,
            Mapping,
        )
        or not serialized_stats
    ):
        raise ValueError(
            "Normalization cache contains "
            f"no statistics: {path}"
        )

    store = NormStatsStore(
        metadata=dict(metadata)
    )

    for key, values in (
        serialized_stats.items()
    ):

        if not isinstance(
            values,
            Mapping,
        ):
            raise ValueError(
                "Invalid statistics entry "
                f"for group {key!r}."
            )

        store.add(
            key,
            NormStats(
                mean=values["mean"],
                std=values["std"],
                count=int(
                    values["count"]
                ),
                element_count=int(
                    values["element_count"]
                ),
            ),
        )

    return store

def get_or_fit_normalization_stats(
    loader: Iterable[Any] | None,
    config: Mapping[str, Any],
    cache_path: str | Path | None = None,
    *,
    split: str = "train",
    logger: Any = None,
) -> NormStatsStore:
    """
    Load matching cached normalization statistics or fit new statistics.

    Cache behavior:

    cache.load = true
        Try to load an existing cache.

    cache.save = true
        Save newly fitted statistics.

    cache.missing = "compute"
        Missing or incompatible cache -> recompute statistics.

    cache.missing = "error"
        Missing or incompatible cache -> raise an error.
    """

    validate_normalization_config(config)

    section, _ = _normalization_section(config)

    options = _statistics_options(section)
    cache = _cache_options(section)

    split = str(split).lower()

    expected_metadata = build_stats_metadata(
        config,
        split,
    )

    load_cache = bool(
        cache.get(
            "load",
            False,
        )
    )

    save_cache = bool(
        cache.get(
            "save",
            False,
        )
    )

    missing_policy = str(
        cache.get(
            "missing",
            "compute",
        )
    ).lower()

    # --------------------------------------------------------
    # Resolve cache path
    # --------------------------------------------------------

    if cache_path is None and (load_cache or save_cache):
        cache_path = build_stats_cache_path(
            config=config,
            cache_dir=cache.get("dir"),
            split=split,
        )

    resolved_path = (
        Path(cache_path)
        if cache_path is not None
        else None
    )

    # ========================================================
    # 1. Try cache
    # ========================================================

    if load_cache and resolved_path is not None:

        if resolved_path.exists():

            try:
                store = load_norm_stats(
                    resolved_path,
                    expected_metadata=expected_metadata,
                )

            except (ValueError, KeyError, TypeError) as error:

                # ------------------------------------------------
                # Existing cache is stale / incompatible / invalid
                # ------------------------------------------------

                if missing_policy == "error":
                    raise ValueError(
                        "Normalization statistics cache exists but "
                        "does not match the current configuration: "
                        f"{resolved_path}"
                    ) from error

                if logger is not None:
                    logger.warning(
                        "Ignoring incompatible normalization cache "
                        "and recomputing statistics | path=%s | reason=%s",
                        resolved_path,
                        error,
                    )

            else:

                if logger is not None:
                    logger.info(
                        "Loaded normalization statistics | path=%s",
                        resolved_path,
                    )

                return store

        else:

            # ----------------------------------------------------
            # Cache file does not exist
            # ----------------------------------------------------

            if missing_policy == "error":
                raise FileNotFoundError(
                    "Normalization statistics cache is required "
                    f"but missing: {resolved_path}"
                )

            if logger is not None:
                logger.info(
                    "Normalization cache not found; "
                    "statistics will be computed | path=%s",
                    resolved_path,
                )

    # ========================================================
    # 2. Fit statistics
    # ========================================================

    if loader is None:
        raise ValueError(
            "A reference loader is required because matching "
            "cached normalization statistics are unavailable."
        )

    if logger is not None:
        logger.info(
            "Computing normalization statistics | "
            "split=%s | grouping=%s | spatial=%s | "
            "channel=%s | std_type=%s",
            split,
            options["grouping"],
            options["spatial"],
            options["channel"],
            options["std_type"],
        )

    store = fit_normalization_stats(
        loader=loader,
        grouping=options["grouping"],
        spatial=options["spatial"],
        channel=options["channel"],
        std_type=options["std_type"],
    )

    # Attach provenance information before saving.
    store.metadata = expected_metadata

    # ========================================================
    # 3. Save newly fitted statistics
    # ========================================================

    if save_cache:

        if resolved_path is None:
            raise ValueError(
                "cache.save=true requires cache.dir "
                "or an explicit cache_path."
            )

        save_norm_stats(
            store,
            resolved_path,
        )

        if logger is not None:
            logger.info(
                "Saved normalization statistics | path=%s",
                resolved_path,
            )

    return store


# ============================================================
# Normalizers
# ============================================================


def _normalize_with_stats(
    image: torch.Tensor,
    stats: NormStats,
    eps: float,
) -> torch.Tensor:
    """
    Apply z-score normalization to one image.

    Formula:

        x_norm = (x - mean) / (std + eps)

    Expected image shape:

        [C, H, W]

    Supported statistics shapes include:

        [1, 1, 1]
        [C, 1, 1]
        [1, H, W]
        [C, H, W]

    PyTorch broadcasting is used to apply the statistics.
    """

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if not torch.is_tensor(image) or image.ndim != 3:
        shape = (
            tuple(image.shape)
            if torch.is_tensor(image)
            else type(image).__name__
        )

        raise ValueError(
            "Expected image shaped [C, H, W], "
            f"got {shape}."
        )

    # --------------------------------------------------------
    # Move statistics to image device / dtype
    # --------------------------------------------------------

    mean = stats.mean.to(
        device=image.device,
        dtype=image.dtype,
    )

    std = stats.std.to(
        device=image.device,
        dtype=image.dtype,
    )

    # --------------------------------------------------------
    # Validate broadcasting
    # --------------------------------------------------------

    try:
        output_shape = torch.broadcast_shapes(
            tuple(image.shape),
            tuple(mean.shape),
        )

    except RuntimeError as error:
        raise ValueError(
            f"Statistics shape {tuple(mean.shape)} "
            f"cannot broadcast to image "
            f"{tuple(image.shape)}."
        ) from error

    if output_shape != tuple(image.shape):
        raise ValueError(
            "Statistics broadcast would change image shape "
            f"from {tuple(image.shape)} "
            f"to {output_shape}."
        )

    # --------------------------------------------------------
    # Z-score
    # --------------------------------------------------------
    #
    # YAML definition:
    #
    #     x_norm = (x - mean) / (std + eps)
    #

    return (
        image - mean
    ) / (
        std + eps
    )

class SampleZScoreNormalizer:
    apply_to = "image"

    def __init__(
        self,
        spatial: str = "global",
        channel: str = "per_channel",
        std_type: str = "population",
        eps: float = 1.0e-6,
    ) -> None:
        self.spatial = spatial
        self.channel = channel
        self.std_type = std_type
        self.eps = float(eps)

    def __call__(self, image: torch.Tensor, metadata: Any = None) -> torch.Tensor:
        if not torch.is_tensor(image) or image.ndim != 3:
            raise ValueError("SampleZScoreNormalizer expects image shaped [C, H, W].")
        stats = compute_batch_stats(
            image.unsqueeze(0), self.spatial, self.channel, self.std_type
        )
        return _normalize_with_stats(image, stats, self.eps)


class ReferenceZScoreNormalizer:
    apply_to = "image"

    def __init__(
        self,
        stats: NormStatsStore,
        grouping: str,
        eps: float = 1.0e-6,
        missing_group: str = "error",
    ) -> None:
        if not isinstance(stats, NormStatsStore) or len(stats) == 0:
            raise ValueError("ReferenceZScoreNormalizer requires non-empty fitted statistics.")
        self.stats = stats
        self.grouping = str(grouping).lower()
        self.eps = float(eps)
        self.missing_group = str(missing_group).lower()
        if self.grouping not in _GROUPINGS:
            raise ValueError(f"Unsupported grouping: {self.grouping!r}.")
        if self.missing_group not in {"error", "global"}:
            raise ValueError("missing_group must be 'error' or 'global'.")

    def __call__(self, image: torch.Tensor, metadata: Any) -> torch.Tensor:
        key = get_group_key(metadata, self.grouping)
        if key not in self.stats and self.missing_group == "global":
            key = _GLOBAL_KEY
        return _normalize_with_stats(image, self.stats.get(key), self.eps)


class LoaderBatchZScoreNormalizer:
    apply_to = "batch"

    def __init__(
        self,
        spatial: str = "global",
        channel: str = "per_channel",
        std_type: str = "population",
        eps: float = 1.0e-6,
    ) -> None:
        self.spatial = spatial
        self.channel = channel
        self.std_type = std_type
        self.eps = float(eps)

    def __call__(
            self,
            images: torch.Tensor,
            metadata: Any = None,
    ) -> torch.Tensor:
        """
        Normalize the current DataLoader mini-batch using statistics
        computed from that same mini-batch.

        Input:

            images.shape = [B, C, H, W]

        Formula:

            x_norm = (x - mean) / (std + eps)
        """

        _validate_images(images)

        reduce_dims = _get_reduce_dims(
            self.spatial,
            self.channel,
        )

        correction = (
            0
            if self.std_type == "population"
            else 1
        )

        values_per_statistic = math.prod(
            images.shape[dim]
            for dim in reduce_dims
        )

        if values_per_statistic <= correction:
            raise ValueError(
                f"Cannot compute {self.std_type} "
                "loader-batch std from "
                f"{values_per_statistic} value(s) "
                "per statistic."
            )

        # --------------------------------------------------------
        # Compute current mini-batch statistics
        # --------------------------------------------------------

        mean = images.mean(
            dim=reduce_dims,
            keepdim=True,
        )

        std = images.std(
            dim=reduce_dims,
            correction=correction,
            keepdim=True,
        )

        # --------------------------------------------------------
        # Z-score
        # --------------------------------------------------------

        return (
                images - mean
        ) / (
                std + self.eps
        )

class ChannelStandardNormalizer:
    """Backward-compatible configured per-channel normalization."""

    apply_to = "image"

    def __init__(self, mean: list[float], std: list[float]) -> None:
        if len(mean) != len(std) or not mean:
            raise ValueError("mean and std must be non-empty and have equal length.")
        if any(float(value) <= 0 for value in std):
            raise ValueError("All std values must be greater than zero.")
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]

    def __call__(self, image: torch.Tensor, metadata: Any = None) -> torch.Tensor:
        stats = NormStats(self.mean, self.std, count=1)
        return _normalize_with_stats(image, stats, eps=torch.finfo(image.dtype).eps)


# ============================================================
# Factory and current-project integration
# ============================================================


def _resolve_project_path(path: str | Path, project_root: str | Path | None) -> Path:
    result = Path(path)
    if not result.is_absolute() and project_root is not None:
        result = Path(project_root) / result
    return result


def _extract_reference_image(item: Any) -> torch.Tensor:
    """
    Extract the image tensor from one RxRxDataset item.

    This adapter allows normalization-statistics fitting to work with
    different dataset return styles without changing the normal training
    dataset interface.

    Supported examples:

        Tensor[C, H, W]

        (image, label)

        {
            "image": image,
            ...
        }
    """

    if torch.is_tensor(item):
        image = item

    elif isinstance(item, Mapping):
        if "image" in item:
            image = item["image"]
        elif "images" in item:
            image = item["images"]
        else:
            raise KeyError(
                "Reference dataset mapping must contain "
                "'image' or 'images'."
            )

    elif isinstance(item, (tuple, list)) and item:
        image = item[0]

    else:
        raise TypeError(
            "Unsupported dataset item type while fitting "
            f"normalization statistics: {type(item).__name__}."
        )

    if not torch.is_tensor(image):
        raise TypeError(
            "Reference dataset image must be a torch.Tensor."
        )

    if image.ndim != 3:
        raise ValueError(
            "Reference dataset image must have shape [C, H, W], "
            f"got {tuple(image.shape)}."
        )

    return image


class _ReferenceStatsDataset(Dataset):
    """
    Adapter around RxRxDataset used only for fitting normalization statistics.

    The normal training dataset does not need to change its return format.

    This adapter adds the technical metadata required by grouped statistics:

        global:
            image only

        experiment:
            image + experiment

        plate:
            image + experiment + plate

    DataLoader will collate these items into a mapping compatible with
    fit_normalization_stats().
    """

    def __init__(
        self,
        base_dataset: Dataset,
        manifest: pd.DataFrame,
        grouping: str,
    ) -> None:
        self.base_dataset = base_dataset
        self.manifest = manifest.reset_index(drop=True)
        self.grouping = str(grouping).lower()

        if len(self.base_dataset) != len(self.manifest):
            raise ValueError(
                "Reference base dataset and manifest must contain "
                "the same number of rows."
            )

        if self.grouping not in {
            "global",
            "experiment",
            "plate",
        }:
            raise ValueError(
                "_ReferenceStatsDataset only supports "
                "global / experiment / plate grouping, "
                f"got {self.grouping!r}."
            )

        required_columns: set[str] = set()

        if self.grouping in {
            "experiment",
            "plate",
        }:
            required_columns.add("experiment")

        if self.grouping == "plate":
            required_columns.add("plate")

        missing_columns = (
            required_columns
            - set(self.manifest.columns)
        )

        if missing_columns:
            raise ValueError(
                "Reference manifest is missing metadata columns "
                "required for normalization grouping: "
                f"{sorted(missing_columns)}."
            )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base_dataset[index]

        image = _extract_reference_image(item)

        result: dict[str, Any] = {
            "image": image,
        }

        if self.grouping in {
            "experiment",
            "plate",
        }:
            result["experiment"] = str(
                self.manifest.iloc[index]["experiment"]
            )

        if self.grouping == "plate":
            result["plate"] = int(
                self.manifest.iloc[index]["plate"]
            )

        return result




def _val_reference_policy(
    section: Mapping[str, Any],
) -> str:

    reference = _reference_options(
        section
    )

    policy = str(
        reference.get(
            "split_policy",
            "train_only",
        )
    ).lower()

    valid = {
        "train_only",
        "val_only",
        "all",
    }

    if policy not in valid:
        raise ValueError(
            "reference.split_policy must be "
            "'train_only', 'val_only', or 'all', "
            f"got {policy!r}."
        )

    return policy

def _build_reference_loader(
    config: Mapping[str, Any],
    reference_split: str,
    project_root: str | Path | None,
) -> DataLoader:
    """
    Build the DataLoader used only for fitting normalization statistics.

    Responsibilities:

    1. Load the correct treatment/control manifests.
    2. Build the requested reference population.
    3. Respect normalization.application.position.
    4. Never apply random augmentation while fitting statistics.
    5. Attach experiment/plate metadata required by grouped statistics.
    """

    # ========================================================
    # Full project config is required
    # ========================================================

    if "data" not in config or not isinstance(
        config["data"],
        Mapping,
    ):
        raise ValueError(
            "Automatic reference fitting requires "
            "the full config with data settings."
        )

    data = config["data"]

    section, _ = _normalization_section(config)
    options = _statistics_options(section)
    reference = _reference_options(section)
    application = _application_options(section)

    grouping = options["grouping"]

    if grouping == "loader_batch":
        raise ValueError(
            "loader_batch statistics are computed dynamically "
            "during training and must not build a reference loader."
        )

    reference_split = str(
        reference_split
    ).lower()

    # ========================================================
    # Load treatment manifest
    # ========================================================

    manifest_key = (
        f"{reference_split}_manifest"
    )

    manifest_path = data.get(
        manifest_key
    )

    if manifest_path is None:
        raise ValueError(
            f"config.data.{manifest_key} is required "
            "for reference fitting."
        )

    train_manifest = pd.read_csv(
        _resolve_project_path(
            manifest_path,
            project_root,
        )
    )

    # ========================================================
    # Load control manifest when required
    # ========================================================

    population = str(
        reference.get(
            "population",
            "",
        )
    )

    control_manifest = None

    if population.upper() != "TRAIN":

        control_key = (
            f"{reference_split}_control_manifest"
        )

        control_path = data.get(
            control_key,
            data.get(
                "control_manifest"
            ),
        )

        if control_path is None:
            raise ValueError(
                f"Reference population {population!r} requires "
                f"config.data.{control_key}."
            )

        control_manifest = pd.read_csv(
            _resolve_project_path(
                control_path,
                project_root,
            )
        )

    # ========================================================
    # Build biological reference population
    # ========================================================

    manifest = build_reference_manifest(
        population=population,
        train_manifest=train_manifest,
        control_manifest=control_manifest,
    )

    # ========================================================
    # Resolve image root
    # ========================================================

    from rxrx1.data.dataset import RxRxDataset
    from rxrx1.data.transforms import resize_image
    from rxrx1.utils.paths import get_image_root

    configured_image_root = data.get(
        "image_root"
    )

    if configured_image_root is not None:

        image_root = _resolve_project_path(
            configured_image_root,
            project_root,
        )

    else:

        # Current d1_train / d1_val manifests both originate
        # from the RxRx1 training image set.
        #
        # Future test support can resolve to the test root.
        image_source_split = (
            "test"
            if reference_split == "test"
            else "train"
        )

        image_root = get_image_root(
            image_source_split
        )

    # ========================================================
    # Statistics-fit transform
    # ========================================================
    #
    # This must NOT use prepare_transforms(config).
    #
    # Otherwise random augmentation such as rotation,
    # brightness, blur, etc. could contaminate mean/std.
    #

    position = str(
        application.get(
            "position",
            "before_resize",
        )
    ).lower()

    if position == "before_resize":

        # Your current experiment:
        #
        # RAW image
        #     ↓
        # fit normalization statistics
        #     ↓
        # resize later
        #
        fit_transform = None

    elif position == "after_resize":

        image_size = data.get(
            "image_size"
        )

        if image_size is None:
            raise ValueError(
                "normalization.application.position='after_resize' "
                "requires config.data.image_size."
            )

        # Only deterministic resize is allowed here.
        #
        # Do NOT include augmentation.
        fit_transform = partial(
            resize_image,
            size=int(image_size),
        )

    else:
        # Normally caught by validate_normalization_config(),
        # retained here for defensive programming.
        raise ValueError(
            "normalization.application.position must be "
            "'before_resize' or 'after_resize'."
        )

    # ========================================================
    # Build label mapping required by current RxRxDataset API
    # ========================================================
    #
    # Labels are irrelevant for normalization fitting,
    # but RxRxDataset currently requires label_to_index.
    #

    if "sirna" not in manifest.columns:
        raise ValueError(
            "Reference manifest is missing 'sirna', which is "
            "required by the current RxRxDataset interface."
        )

    manifest = manifest.copy()

    manifest["sirna"] = (
        manifest["sirna"]
        .astype(str)
    )

    labels = sorted(
        manifest["sirna"].unique()
    )

    label_to_index = {
        label: index
        for index, label in enumerate(labels)
    }

    # ========================================================
    # Base image dataset
    # ========================================================

    base_dataset = RxRxDataset(
        manifest=manifest,
        image_root=image_root,
        label_to_index=label_to_index,

        # IMPORTANT:
        #
        # before_resize -> None
        #
        # after_resize  -> resize only
        #
        # Never random augmentation.
        transform=fit_transform,

        # Statistics must be fitted from the reference images
        # themselves, not from already-normalized images.
        normalizer=None,
    )

    # ========================================================
    # Attach technical metadata
    # ========================================================

    stats_dataset = _ReferenceStatsDataset(
        base_dataset=base_dataset,
        manifest=manifest,
        grouping=grouping,
    )

    # ========================================================
    # Fit DataLoader
    # ========================================================

    fit_config = (
        section.get("fit")
        or {}
    )

    if not isinstance(
        fit_config,
        Mapping,
    ):
        raise TypeError(
            "normalization.fit must be a mapping."
        )

    batch_size = int(
        fit_config.get(
            "batch_size",
            data.get(
                "batch_size",
                16,
            ),
        )
    )

    num_workers = int(
        fit_config.get(
            "num_workers",
            data.get(
                "num_workers",
                0,
            ),
        )
    )

    if batch_size <= 0:
        raise ValueError(
            "normalization.fit.batch_size "
            "must be greater than zero."
        )

    if num_workers < 0:
        raise ValueError(
            "normalization.fit.num_workers "
            "cannot be negative."
        )

    return DataLoader(
        stats_dataset,
        batch_size=batch_size,

        # Statistics fitting must be deterministic with
        # respect to membership. Order does not matter.
        shuffle=False,

        num_workers=num_workers,

        # Statistics are accumulated on CPU in float64.
        pin_memory=False,
    )


def build_normalizer(
    config: Mapping[str, Any],
    stats: NormStatsStore | None = None,
    *,
    split: str = "train",
    project_root: str | Path | None = None,
    logger: Any = None,
    loader: Iterable[Any] | None = None,
):
    """
    Build the runtime normalizer from the normalization configuration.

    Configuration responsibilities are intentionally separated:

    method:
        Which mathematical transformation is used.

        Example:
            zscore

    statistics.source:
        Where mean/std come from.

        sample:
            Statistics are computed from the current image.

        reference:
            Statistics come from an external reference population
            or from the current DataLoader mini-batch when
            grouping='loader_batch'.

    statistics.grouping:
        Which technical scope shares one set of statistics.

        global
        experiment
        plate
        loader_batch

    statistics.spatial / statistics.channel:
        Define the geometry of the statistics.

    This factory only chooses and prepares the correct normalizer.
    """

    # ========================================================
    # Validate configuration
    # ========================================================

    validate_normalization_config(config)

    section, embedded = _normalization_section(config)

    enabled = _is_enabled(
        section,
        embedded,
    )

    method = _method(section)

    # --------------------------------------------------------
    # Disabled / no normalization
    # --------------------------------------------------------

    if not enabled or method == "none":
        if logger is not None:
            logger.info(
                "Normalization disabled | split=%s",
                split,
            )

        return None

    # ========================================================
    # Common configuration
    # ========================================================

    options = _statistics_options(section)
    application = _application_options(section)

    source = options["source"]
    grouping = options["grouping"]
    spatial = options["spatial"]
    channel = options["channel"]
    std_type = options["std_type"]
    reference_policy = "none"
    reference_scope = "none"

    eps = float(
        section.get(
            "eps",
            1.0e-6,
        )
    )

    position = str(
        application.get(
            "position",
            "before_resize",
        )
    ).lower()

    # ========================================================
    # Method: channel_standard
    # ========================================================
    #
    # Backward-compatible manually configured normalization.
    #
    # This does not use fitted statistics.
    #

    if method == "channel_standard":

        params = section.get(
            "params"
        ) or {}

        normalizer = ChannelStandardNormalizer(
            mean=params["mean"],
            std=params["std"],
        )

    # ========================================================
    # Method: zscore
    # ========================================================

    elif method == "zscore":

        # ====================================================
        # Source: sample
        # ====================================================
        #
        # Current image computes its own statistics.
        #
        # Example: first-place RxRx1 normalization
        #
        # source: sample
        # spatial: global
        # channel: per_channel
        #

        if source == "sample":

            normalizer = SampleZScoreNormalizer(
                spatial=spatial,
                channel=channel,
                std_type=std_type,
                eps=eps,
            )

        # ====================================================
        # Source: reference
        # ====================================================

        elif source == "reference":

            # ------------------------------------------------
            # Dynamic DataLoader mini-batch statistics
            # ------------------------------------------------
            #
            # Nothing is fitted or cached.
            #
            # Current [B,C,H,W] batch:
            #
            #     calculate mean/std
            #         ↓
            #     normalize same batch
            #

            if grouping == "loader_batch":

                normalizer = LoaderBatchZScoreNormalizer(
                    spatial=spatial,
                    channel=channel,
                    std_type=std_type,
                    eps=eps,
                )

            # ------------------------------------------------
            # Pre-fitted reference statistics
            # ------------------------------------------------
            #
            # global / experiment / plate
            #

            else:

                requested_split = str(
                    split
                ).lower()

                policy = _val_reference_policy(
                    section
                )
                reference_policy = policy


                cache = _cache_options(
                    section
                )

                cache_dir = Path(
                    cache.get(
                        "dir",
                        "outputs/normalization_stats",
                    )
                )

                if (
                        not cache_dir.is_absolute()
                        and project_root is not None
                ):
                    cache_dir = (

                            Path(project_root)

                            / cache_dir

                    )

                # ====================================================

                # Helper: load/fit one split's reference stats

                # ====================================================

                def get_reference_stats(

                        reference_split: str,

                        supplied_stats: NormStatsStore | None = None,

                ) -> NormStatsStore:

                    if supplied_stats is not None:
                        return supplied_stats

                    cache_path = build_stats_cache_path(

                        config=config,

                        cache_dir=cache_dir,

                        split=reference_split,

                    )

                    reference_loader = _build_reference_loader(

                        config=config,

                        reference_split=reference_split,

                        project_root=project_root,

                    )

                    return get_or_fit_normalization_stats(

                        loader=reference_loader,

                        config=config,

                        cache_path=cache_path,

                        split=reference_split,

                        logger=logger,

                    )

                # ====================================================

                # TRAIN

                # ====================================================

                if requested_split == "train":

                    runtime_stats = get_reference_stats(
                        "train",
                        supplied_stats=stats,
                    )

                    runtime_grouping = grouping

                    reference_scope = "train_group"


                # ====================================================

                # VALIDATION

                # ====================================================

                elif requested_split == "val":

                    # ------------------------------------------------

                    # train_only

                    # ------------------------------------------------

                    if policy == "train_only":

                        train_stats = get_reference_stats(
                            "train",
                            supplied_stats=stats,
                        )

                        if _GLOBAL_KEY not in train_stats:
                            raise KeyError(
                                "train_only validation requires "
                                "pooled train global statistics."
                            )

                        runtime_stats = train_stats
                        runtime_grouping = "global"

                        reference_scope = "train_global"


                    # ------------------------------------------------

                    # val_only

                    # ------------------------------------------------

                    elif policy == "val_only":

                        runtime_stats = get_reference_stats(

                            "val"

                        )

                        runtime_grouping = grouping

                        reference_scope = "val_group"

                    # ------------------------------------------------

                    # all

                    # ------------------------------------------------

                    elif policy == "all":

                        train_stats = get_reference_stats(

                            "train",

                            supplied_stats=stats,

                        )

                        val_stats = get_reference_stats(

                            "val"

                        )

                        runtime_stats = merge_train_global_with_val_stats(

                            train_stats=train_stats,

                            val_stats=val_stats,

                            grouping=grouping,

                            std_type=std_type,

                        )

                        runtime_grouping = grouping

                        reference_scope = "train_global+val_group"


                    else:

                        raise ValueError(

                            f"Unknown validation reference policy: {policy!r}."

                        )


                else:

                    raise ValueError(

                        "Reference normalization currently supports "

                        f"train/val splits, got {requested_split!r}."

                    )

                normalizer = ReferenceZScoreNormalizer(

                    stats=runtime_stats,

                    grouping=runtime_grouping,

                    eps=eps,

                    missing_group=str(

                        section.get(

                            "missing_group",

                            "error",

                        )

                    ),

                )
        else:

            # Should normally already be caught by
            # validate_normalization_config().
            raise ValueError(
                "Unsupported normalization statistics source: "
                f"{source!r}."
            )

    # ========================================================
    # Future methods
    # ========================================================

    else:
        raise NotImplementedError(
            f"Normalization method {method!r} "
            "is not implemented."
        )

    # ========================================================
    # Runtime metadata attached to normalizer
    # ========================================================
    #
    # application.position will be used by the image pipeline
    # integration step.
    #
    # Keeping it on the normalizer avoids reparsing YAML inside
    # Dataset / trainer.
    #

    normalizer.position = position

    # ========================================================
    # Logging
    # ========================================================

    if logger is not None:
        logger.info(
            "Normalization | split=%s | policy=%s | "
            "method=%s | source=%s | reference=%s | "
            "grouping=%s | spatial=%s | channel=%s | "
            "position=%s | apply_to=%s",
            split,
            reference_policy,
            method,
            source,
            reference_scope,
            getattr(
                normalizer,
                "grouping",
                grouping,
            ) or "none",
            spatial,
            channel,
            position,
            getattr(
                normalizer,
                "apply_to",
                "none",
            ),
        )

    return normalizer

__all__ = [
    "NormStats",
    "NormStatsStore",
    "validate_normalization_config",
    "build_reference_manifest",
    "get_group_key",
    "compute_batch_stats",
    "fit_normalization_stats",
    "build_stats_metadata",
    "validate_stats_metadata",
    "build_stats_cache_path",
    "save_norm_stats",
    "load_norm_stats",
    "get_or_fit_normalization_stats",
    "SampleZScoreNormalizer",
    "ReferenceZScoreNormalizer",
    "LoaderBatchZScoreNormalizer",
    "ChannelStandardNormalizer",
    "build_normalizer",
]

