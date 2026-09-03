import math
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


# ============================================================
# Basic utilities
# ============================================================


def resize_image(
    image: torch.Tensor,
    size: int,
) -> torch.Tensor:
    image = image.unsqueeze(0)

    image = F.interpolate(
        image,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )

    return image.squeeze(0)


def compute_channel_stats(
    dataset,
    batch_size: int = 16,
    num_workers: int = 4,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    channel_sum = None
    channel_sq_sum = None
    pixel_count = 0

    for batch in tqdm(
        loader,
        desc="Computing mean/std",
    ):
        images = batch["image"]

        if channel_sum is None:
            num_channels = images.shape[1]

            channel_sum = torch.zeros(
                num_channels,
                dtype=torch.float64,
            )

            channel_sq_sum = torch.zeros(
                num_channels,
                dtype=torch.float64,
            )

        channel_sum += images.sum(
            dim=(0, 2, 3)
        ).double()

        channel_sq_sum += (
            images ** 2
        ).sum(
            dim=(0, 2, 3)
        ).double()

        pixel_count += (
            images.shape[0]
            * images.shape[2]
            * images.shape[3]
        )

    if pixel_count == 0:
        raise ValueError(
            "Dataset is empty. Cannot compute channel statistics."
        )

    mean = channel_sum / pixel_count

    variance = (
        channel_sq_sum / pixel_count
        - mean ** 2
    )

    # Avoid tiny negative values caused by floating-point error.
    variance = torch.clamp(
        variance,
        min=0.0,
    )

    std = torch.sqrt(variance)

    return mean.tolist(), std.tolist()


# ============================================================
# Geometric augmentation
# ============================================================


class Random90Rotation:
    """
    Random rotation by multiples of 90 degrees.

    Possible rotations:
        0°
        90°
        180°
        270°

    torch.rot90 is used, so no interpolation is introduced.
    """

    def __init__(
        self,
        p: float = 1.0,
    ):
        self.p = p

    def __call__(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if random.random() >= self.p:
            return image

        k = random.randint(0, 3)

        return torch.rot90(
            image,
            k=k,
            dims=(-2, -1),
        )


# ============================================================
# Intensity augmentation
# ============================================================


class GaussianNoise:
    def __init__(
        self,
        std: float = 0.01,
        p: float = 0.5,
    ):
        self.std = std
        self.p = p

    def __call__(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if random.random() >= self.p:
            return image

        noise = (
            torch.randn_like(image)
            * self.std
        )

        return image + noise


class RandomBrightness:
    """
    Brightness augmentation for arbitrary channel count.

    x' = a * x
    """

    def __init__(
        self,
        factor_min: float = 0.9,
        factor_max: float = 1.1,
        p: float = 0.5,
    ):
        self.factor_min = factor_min
        self.factor_max = factor_max
        self.p = p

    def __call__(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if random.random() >= self.p:
            return image

        factor = random.uniform(
            self.factor_min,
            self.factor_max,
        )

        return image * factor


class RandomContrast:
    """
    Contrast augmentation for arbitrary channel count.

    Contrast is adjusted independently around each channel's
    spatial mean:

        x'_c = mean_c + a * (x_c - mean_c)
    """

    def __init__(
        self,
        factor_min: float = 0.9,
        factor_max: float = 1.1,
        p: float = 0.5,
    ):
        self.factor_min = factor_min
        self.factor_max = factor_max
        self.p = p

    def __call__(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if random.random() >= self.p:
            return image

        factor = random.uniform(
            self.factor_min,
            self.factor_max,
        )

        channel_mean = image.mean(
            dim=(-2, -1),
            keepdim=True,
        )

        return (
            channel_mean
            + factor
            * (image - channel_mean)
        )


class ChannelIntensityJitter:
    """
    Per-channel intensity jitter used by the RxRx1 1st-place solution.

    x'_c = a_c * x_c + b_c

    a_c ~ N(scale_mean, scale_std)
    b_c ~ N(shift_mean, shift_std)
    """

    def __init__(
        self,
        scale_mean: float = 1.0,
        scale_std: float = 0.1,
        shift_mean: float = 0.0,
        shift_std: float = 0.1,
        p: float = 1.0,
    ):
        self.scale_mean = scale_mean
        self.scale_std = scale_std
        self.shift_mean = shift_mean
        self.shift_std = shift_std
        self.p = p

    def __call__(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if random.random() >= self.p:
            return image

        num_channels = image.shape[0]

        scales = torch.empty(
            num_channels,
            1,
            1,
            device=image.device,
            dtype=image.dtype,
        ).normal_(
            mean=self.scale_mean,
            std=self.scale_std,
        )

        shifts = torch.empty(
            num_channels,
            1,
            1,
            device=image.device,
            dtype=image.dtype,
        ).normal_(
            mean=self.shift_mean,
            std=self.shift_std,
        )

        return image * scales + shifts


# ============================================================
# Legacy / additional intensity augmentation
# ============================================================


class GlobalIntensityScale:
    """
    Kept for compatibility with previous YAML files.

        x' = a * x
    """

    def __init__(
        self,
        scale_min: float = 0.9,
        scale_max: float = 1.1,
        p: float = 0.5,
    ):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.p = p

    def __call__(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if random.random() >= self.p:
            return image

        scale = random.uniform(
            self.scale_min,
            self.scale_max,
        )

        return image * scale


class ChannelIntensityScale:
    """
    Kept for compatibility.

        x'_c = a_c * x_c

    For the RxRx1 backbone, use ChannelIntensityJitter instead.
    """

    def __init__(
        self,
        scale_min: float = 0.9,
        scale_max: float = 1.1,
        p: float = 0.5,
    ):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.p = p

    def __call__(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if random.random() >= self.p:
            return image

        scales = torch.empty(
            image.shape[0],
            1,
            1,
            device=image.device,
            dtype=image.dtype,
        ).uniform_(
            self.scale_min,
            self.scale_max,
        )

        return image * scales


class RandomGamma:
    def __init__(
        self,
        gamma_min: float = 0.9,
        gamma_max: float = 1.1,
        p: float = 0.5,
    ):
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.p = p

    def __call__(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if random.random() >= self.p:
            return image

        gamma = random.uniform(
            self.gamma_min,
            self.gamma_max,
        )

        image_min = image.amin()
        image_max = image.amax()

        if image_max <= image_min:
            return image

        normalized = (
            image - image_min
        ) / (
            image_max - image_min
        )

        normalized = normalized.pow(
            gamma
        )

        return (
            normalized
            * (image_max - image_min)
            + image_min
        )


# ============================================================
# Channel dropout
# ============================================================


class ChannelDropout:
    """
    Randomly zero one or more microscopy channels.

    Kept as an interface for later long-epoch experiments.
    """

    def __init__(
        self,
        p: float = 0.2,
        max_channels: int = 1,
    ):
        self.p = p
        self.max_channels = max_channels

    def __call__(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if random.random() >= self.p:
            return image

        image = image.clone()

        num_channels = random.randint(
            1,
            min(
                self.max_channels,
                image.shape[0],
            ),
        )

        channels = random.sample(
            range(image.shape[0]),
            num_channels,
        )

        image[channels] = 0

        return image


# ============================================================
# Interpolation helper
# ============================================================


def get_interpolation_mode(
    name: str,
) -> InterpolationMode:
    interpolation_modes = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
    }

    name = name.lower()

    if name not in interpolation_modes:
        raise ValueError(
            f"Unknown interpolation mode: {name}"
        )

    return interpolation_modes[name]


# ============================================================
# Batch-level augmentation
# MixUp / CutMix
# ============================================================


def _to_soft_targets(
    targets: torch.Tensor,
    num_classes: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Convert integer labels to one-hot / soft labels.

    Also accepts targets that are already soft labels.
    """

    if targets.ndim == 1:
        return F.one_hot(
            targets,
            num_classes=num_classes,
        ).to(dtype=dtype)

    if (
        targets.ndim == 2
        and targets.shape[1] == num_classes
    ):
        return targets.to(dtype=dtype)

    raise ValueError(
        "Targets must be class indices "
        "or soft targets."
    )


class MixUp:
    """
    Batch-level MixUp.

    This is intentionally separate from image-level transforms.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.2,
        p: float = 1.0,
    ):
        self.num_classes = num_classes
        self.alpha = alpha
        self.p = p

    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ):
        targets = _to_soft_targets(
            targets,
            num_classes=self.num_classes,
            dtype=images.dtype,
        )

        if random.random() >= self.p:
            return images, targets

        distribution = torch.distributions.Beta(
            self.alpha,
            self.alpha,
        )

        lam = distribution.sample().to(
            device=images.device,
            dtype=images.dtype,
        )

        permutation = torch.randperm(
            images.shape[0],
            device=images.device,
        )

        mixed_images = (
            lam * images
            + (1.0 - lam)
            * images[permutation]
        )

        mixed_targets = (
            lam * targets
            + (1.0 - lam)
            * targets[permutation]
        )

        return (
            mixed_images,
            mixed_targets,
        )


class CutMix:
    """
    Batch-level CutMix.

    Kept ready for later long-epoch experiments.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 1.0,
        p: float = 1.0,
    ):
        self.num_classes = num_classes
        self.alpha = alpha
        self.p = p

    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ):
        targets = _to_soft_targets(
            targets,
            num_classes=self.num_classes,
            dtype=images.dtype,
        )

        if random.random() >= self.p:
            return images, targets

        distribution = torch.distributions.Beta(
            self.alpha,
            self.alpha,
        )

        lam = float(
            distribution.sample()
        )

        batch_size, _, height, width = (
            images.shape
        )

        permutation = torch.randperm(
            batch_size,
            device=images.device,
        )

        cut_ratio = math.sqrt(
            1.0 - lam
        )

        cut_width = int(
            width * cut_ratio
        )

        cut_height = int(
            height * cut_ratio
        )

        center_x = random.randint(
            0,
            width - 1,
        )

        center_y = random.randint(
            0,
            height - 1,
        )

        x1 = max(
            center_x - cut_width // 2,
            0,
        )

        x2 = min(
            center_x + cut_width // 2,
            width,
        )

        y1 = max(
            center_y - cut_height // 2,
            0,
        )

        y2 = min(
            center_y + cut_height // 2,
            height,
        )

        mixed_images = images.clone()

        mixed_images[
            :,
            :,
            y1:y2,
            x1:x2,
        ] = images[
            permutation,
            :,
            y1:y2,
            x1:x2,
        ]

        patch_area = (
            (x2 - x1)
            * (y2 - y1)
        )

        lam_adjusted = (
            1.0
            - patch_area
            / (width * height)
        )

        mixed_targets = (
            lam_adjusted
            * targets
            + (
                1.0
                - lam_adjusted
            )
            * targets[permutation]
        )

        return (
            mixed_images,
            mixed_targets,
        )


class BatchCompose:
    def __init__(
        self,
        transforms_list,
    ):
        self.transforms = transforms_list

    def __call__(
        self,
        images,
        targets,
    ):
        for transform in self.transforms:
            images, targets = transform(
                images,
                targets,
            )

        return images, targets


def build_batch_transform(
    transform_config: list[dict],
    num_classes: int,
):
    """
    Build batch-level augmentation pipeline.

    Currently supports:
        MixUp
        CutMix

    Example future YAML:

    batch:
      - name: mixup
        alpha: 0.2
        p: 1.0
    """

    pipeline = []

    for config in transform_config:
        name = config["name"]

        if name == "mixup":
            pipeline.append(
                MixUp(
                    num_classes=num_classes,
                    alpha=config.get(
                        "alpha",
                        0.2,
                    ),
                    p=config.get(
                        "p",
                        1.0,
                    ),
                )
            )

        elif name == "cutmix":
            pipeline.append(
                CutMix(
                    num_classes=num_classes,
                    alpha=config.get(
                        "alpha",
                        1.0,
                    ),
                    p=config.get(
                        "p",
                        1.0,
                    ),
                )
            )

        else:
            raise ValueError(
                "Unknown batch transform: "
                f"{name}"
            )

    if not pipeline:
        return None

    return BatchCompose(
        pipeline
    )

SIZE_TRANSFORMS = {
    "resize",
    "random_resized_crop",
    "crop",
    "center_crop",
}


def build_transform(transform_config: list[dict]):
    pipeline = []

    for config in transform_config:
        name = str(config["name"]).lower()

        # Resize
        if name == "resize":
            size = config["size"]
            interpolation = get_interpolation_mode(
                config.get("interpolation", "bilinear")
            )
            pipeline.append(
                transforms.Resize(
                    (size, size),
                    interpolation=interpolation,
                )
            )

        # Random resized crop
        elif name in {"random_resized_crop", "crop"}:
            pipeline.append(
                transforms.RandomResizedCrop(
                    size=config["size"],
                    scale=tuple(config.get("scale", [0.5, 1.0])),
                    ratio=tuple(config.get("ratio", [1.0, 1.0])),
                    interpolation=get_interpolation_mode(
                        config.get("interpolation", "nearest")
                    ),
                )
            )

        # Center crop
        elif name == "center_crop":
            pipeline.append(
                transforms.CenterCrop(
                    size=config["size"]
                )
            )

        # Flip
        elif name == "horizontal_flip":
            pipeline.append(
                transforms.RandomHorizontalFlip(
                    p=config.get("p", 0.5)
                )
            )

        elif name == "vertical_flip":
            pipeline.append(
                transforms.RandomVerticalFlip(
                    p=config.get("p", 0.5)
                )
            )

        # Exact 90-degree rotation
        elif name == "random_90_rotation":
            pipeline.append(
                Random90Rotation(
                    p=config.get("p", 1.0)
                )
            )

        # Continuous rotation
        elif name == "rotation":
            pipeline.append(
                transforms.RandomRotation(
                    degrees=config.get("degrees", 15),
                    interpolation=get_interpolation_mode(
                        config.get("interpolation", "bilinear")
                    ),
                )
            )

        # Affine
        elif name == "affine":
            pipeline.append(
                transforms.RandomAffine(
                    degrees=config.get("degrees", 10),
                    translate=tuple(config.get("translate", [0.05, 0.05])),
                    scale=tuple(config.get("scale", [0.95, 1.05])),
                    shear=config.get("shear", 5),
                )
            )

        # Winner-style per-channel intensity jitter
        elif name == "channel_intensity_jitter":
            pipeline.append(
                ChannelIntensityJitter(
                    scale_mean=config.get("scale_mean", 1.0),
                    scale_std=config.get("scale_std", 0.1),
                    shift_mean=config.get("shift_mean", 0.0),
                    shift_std=config.get("shift_std", 0.1),
                    p=config.get("p", 1.0),
                )
            )

        # Brightness
        elif name == "brightness":
            pipeline.append(
                RandomBrightness(
                    factor_min=config.get("factor_min", 0.9),
                    factor_max=config.get("factor_max", 1.1),
                    p=config.get("p", 0.5),
                )
            )

        # Contrast
        elif name == "contrast":
            pipeline.append(
                RandomContrast(
                    factor_min=config.get("factor_min", 0.9),
                    factor_max=config.get("factor_max", 1.1),
                    p=config.get("p", 0.5),
                )
            )

        # Gaussian noise
        elif name == "gaussian_noise":
            pipeline.append(
                GaussianNoise(
                    std=config.get("std", 0.01),
                    p=config.get("p", 0.5),
                )
            )

        # Gaussian blur
        elif name == "gaussian_blur":
            pipeline.append(
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            kernel_size=config.get("kernel_size", 3),
                            sigma=tuple(config.get("sigma", [0.1, 1.0])),
                        )
                    ],
                    p=config.get("p", 0.5),
                )
            )

        # Legacy intensity transforms
        elif name == "global_intensity_scale":
            pipeline.append(
                GlobalIntensityScale(
                    scale_min=config.get("scale_min", 0.9),
                    scale_max=config.get("scale_max", 1.1),
                    p=config.get("p", 0.5),
                )
            )

        elif name == "channel_intensity_scale":
            pipeline.append(
                ChannelIntensityScale(
                    scale_min=config.get("scale_min", 0.9),
                    scale_max=config.get("scale_max", 1.1),
                    p=config.get("p", 0.5),
                )
            )

        elif name == "gamma":
            pipeline.append(
                RandomGamma(
                    gamma_min=config.get("gamma_min", 0.9),
                    gamma_max=config.get("gamma_max", 1.1),
                    p=config.get("p", 0.5),
                )
            )

        # Channel dropout
        elif name == "channel_dropout":
            pipeline.append(
                ChannelDropout(
                    p=config.get("p", 0.2),
                    max_channels=config.get("max_channels", 1),
                )
            )

        # Random erasing
        elif name == "random_erasing":
            pipeline.append(
                transforms.RandomErasing(
                    p=config.get("p", 0.25),
                    scale=tuple(config.get("scale", [0.02, 0.15])),
                    ratio=tuple(config.get("ratio", [0.5, 2.0])),
                    value=config.get("value", 0),
                )
            )

        # Batch-level augmentation
        elif name in {"mixup", "cutmix"}:
            raise ValueError(
                f"{name} is a batch-level augmentation. "
                "Use build_batch_transform() instead of build_transform()."
            )

        else:
            raise ValueError(f"Unknown transform: {name}")

    return transforms.Compose(pipeline)


def resolve_transform_config(
    transform_config: list[dict],
    image_size: int,
) -> list[dict]:
    resolved = []

    for config in transform_config:
        config = config.copy()
        name = str(config.get("name", "")).lower()

        if name in SIZE_TRANSFORMS:
            config.setdefault("size", image_size)

        resolved.append(config)

    return resolved


def prepare_transforms(config):
    """Build image transforms using data.image_size as the default size."""
    transform_config = config.get("transform") or {}

    if not transform_config.get("switch", False):
        return None, None

    image_size = config["data"]["image_size"]

    train_config = resolve_transform_config(
        transform_config.get("train", []),
        image_size,
    )
    val_config = resolve_transform_config(
        transform_config.get("val", []),
        image_size,
    )

    return (
        build_transform(train_config),
        build_transform(val_config),
    )
