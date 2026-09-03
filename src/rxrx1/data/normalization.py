import torch


class ChannelStandardNormalizer:
    """Per-channel standardization: (x - mean) / std."""

    def __init__(
        self,
        mean: list[float],
        std: list[float],
    ):
        if len(mean) != len(std):
            raise ValueError("mean and std must have the same length.")

        if any(value <= 0 for value in std):
            raise ValueError("All std values must be > 0.")

        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, image: torch.Tensor, row=None) -> torch.Tensor:
        if not torch.is_tensor(image):
            raise TypeError(
                "Normalizer expects a torch.Tensor. "
                "Make sure ToTensor is applied first."
            )

        if image.ndim != 3:
            raise ValueError(
                f"Expected image shape [C, H, W], got {tuple(image.shape)}."
            )

        if image.shape[0] != len(self.mean):
            raise ValueError(
                f"Image has {image.shape[0]} channels, "
                f"but normalization has {len(self.mean)} channels."
            )

        mean = self.mean.to(
            device=image.device,
            dtype=image.dtype,
        )[:, None, None]

        std = self.std.to(
            device=image.device,
            dtype=image.dtype,
        )[:, None, None]

        return (image - mean) / std


def build_normalizer(
    config,
    split: str,
    project_root=None,
    logger=None,
):
    """
    Build normalization strategy.

    `split`, `project_root`, and `logger` are intentionally part of the
    interface so future methods such as control_plate can use different
    train/val control manifests and metadata.
    """
    normalization_config = config.get("normalization") or {}

    if not normalization_config.get("switch", False):
        return None

    method = str(
        normalization_config.get("method", "none")
    ).lower()

    if method == "none":
        return None

    if method == "channel_standard":
        params = normalization_config.get("params") or {}

        mean = params.get("mean")
        std = params.get("std")

        if mean is None or std is None:
            raise ValueError(
                "channel_standard requires normalization.params.mean "
                "and normalization.params.std."
            )

        normalizer = ChannelStandardNormalizer(
            mean=mean,
            std=std,
        )

        if logger is not None:
            logger.info(
                "Normalization | split=%s | method=%s",
                split,
                method,
            )

        return normalizer

    raise ValueError(
        f"Unknown normalization method: {method}"
    )